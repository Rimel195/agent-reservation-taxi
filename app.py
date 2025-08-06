import streamlit as st
import json
import gc
import torch
import os
import logging
from transformers import logging as transformers_logging
import json
from typing import Dict, Any, Tuple
from datetime import datetime
import re
import torch
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import json
from huggingface_hub import login , whoami



gc.collect()
torch.cuda.empty_cache()
#  Désactive le parallélisme des tokenizers (prétraitement des textes) de Hugging Face ,cela évite certains avertissements ou comportements indésirables
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Configurer le logging pour afficher que les erreurs
logging.basicConfig(level=logging.ERROR)   #configure le système de logging standard de Python
transformers_logging.set_verbosity_error() # configure le logging spécifique à la bibliothèque transformers de Hugging Face.


HF_TOKEN = st.secrets["HF_TOKEN"]
# 1. Nettoyage des variables conflictuelles
os.environ.pop("HF_TOKEN", None)  # Supprime la variable d'environnement
os.environ.pop("HUGGINGFACE_TOKEN", None)  # Nettoie d'autres variables possibles

# 2. Vérification stricte du token
if "HF_TOKEN" not in st.secrets:
    st.error("""
    🔐 Token HF manquant. Ajoutez-le dans :
    `.streamlit/secrets.toml` sous la forme :
    HF_TOKEN = "votre_token_hf"
    """)
    st.stop()

# 3. Authentification explicite
try:
    login(token=st.secrets["HF_TOKEN"], add_to_git_credential=False)
    st.success("Authentification HF réussie !")
except Exception as e:
    st.error(f"Échec de l'authentification : {str(e)}")
    st.stop()



# 1. Initialisation du modèle Mistral
model_name = "mistralai/Mistral-7B-Instruct-v0.3"
tokenizer = AutoTokenizer.from_pretrained(model_name , trust_remote_code=True, use_fast=False,token=HF_TOKEN)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",          # Utilise le GPU automatiquement
    load_in_4bit=True,          # Quantification 4-bit
    torch_dtype=torch.float16,   # Optimisation pour GPU
    token=HF_TOKEN,
    trust_remote_code=True
)

# Création du pipeline HF
mistral_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=100,
    do_sample=True,
    temperature=0.5,
    top_p=0.95,
    repetition_penalty=1.1  # Évite les répétitions
)


st.write("Tokenizer chargé avec succès ✅")


# Adaptation pour LangChain
llm = HuggingFacePipeline(pipeline=mistral_pipeline)
# 2. Définition des prompts
CONVERSATION_PROMPT = PromptTemplate(
    input_variables=["history", "next_question"],
    template="""<s>[INST]
Vous êtes un assistant de réservation de taxi en France. Posez la question suivante AU CLIENT de manière naturelle.

Historique:
{history}

Prochaine question à poser: {next_question}

Instructions:
1.Répondez sans montionner l'historique de conversation.
2.Répondez avec des phrases courtes et claires, rien de trop compliqué ,sans des commentaires.
3.Parlez en francais uniquement sans des commentaires en anglais .
4.Ne resaluez pas ensuite.

Règles STRICTES:
1. Grammaire et syntaxe parfaites
2. Aucune hallucination


Formulez cette question de manière conversationnelle: [/INST]"""
)

EXTRACTION_PROMPT = PromptTemplate(
    input_variables=["user_input", "current_info", "current_field"],
    template="""<s>[INST]
Extrayez uniquement la valeur COMPLÈTE (sans format JSON) pour le champ '{current_field}' à partir de cette réponse utilisateur:

Réponse: "{user_input}"

Informations déjà collectées (ne pas modifier):
{current_info}

Instructions:
1. Extrayez seulement la valeur COMPLÈTE pour '{current_field}'
2. Ne retournez pas de JSON, juste la valeur brute
3. Pas de commentaires ou explications
4. Règles de formatage STRICTES:
   - Dates: CONVERTISSEZ TOUJOURS en format jour/mois/année (ex: 05/12/2025)
     * Si l'année absente: retourner la VALEUR DE L'année ACTUELLE
   - Heures: CONVERTISSEZ TOUJOURS en format heures:minutes (ex: 14:30)
   - Numéros: UNIQUEMENT les chiffres (supprimez espaces, tirets, etc.)
5. Exemples de conversion:
   - "3 avril 2025" → "03/04/2025"
   - "14h30" → "14:30"
   - "01-02-2025" → "01/02/2025"
   - "5:5" → "05:05"
6. Complétez avec des zéros si nécessaire (ex: 5 → 05)
7. Interdictions:
   - Ne pas inventer de données manquantes
   - Ne modifiez pas les autres champs déjà collectés

Valeur extraite (FORMAT FINAL APPLIQUÉ): [/INST]"""
)


VALIDATION_PROMPT = PromptTemplate(
    input_variables=["field", "error_msg", "template_question"],
    template="""<s>[INST]
Vous êtes un assistant de réservation de taxi. L'information fournie pour {field} est invalide.

Message d'erreur: {error_msg}

Reposez la question suivante de manière naturelle et polie:
{template_question}

Instructions:
1. Incorporez l'erreur naturellement dans la question
2. Restez professionnel et amical
3. Fournissez un exemple si pertinent
4. Formulez en français clair [/INST]"""
)



# 3. Création des chaînes
conversation_chain = LLMChain(llm=llm, prompt=CONVERSATION_PROMPT)
extraction_chain = LLMChain(llm=llm, prompt=EXTRACTION_PROMPT)
validation_chain = LLMChain(llm=llm, prompt=VALIDATION_PROMPT)






# 4. Classe pour gérer la réservation
class TaxiReservationSystem:
    def __init__(self):
        self.reservation_info = {
            "nom et prenom": "",
            "date": "",
            "heure": "",
            "telephone": "",
            "adresse_depart": "",
            "adresse_destination": ""
        }
        self.conversation_history = []
        self.questions_flow = [
            ("nom et prenom", "Pour commencer, pourriez-vous me donner votre prénom et votre nom ?"),
            ("date", "Quelle date souhaitez-vous pour la réservation ? Merci d'indiquer la date comme suit jour/mois/année"),
            ("heure", "À quelle heure avez-vous besoin du taxi ? heures précise "),
            ("telephone", "Peux-tu me donner ton numéro de téléphone sur lequel vous etes joignable s'il te plaît ? Assure-toi qu'il comporte bien 10 chiffres. "),
            ("adresse_depart", "Adresse exacte de départ  (adresse complète) ?"),
            ("adresse_destination", "Adresse de destination ?")
        ]

    def clean_response(self, response):
        """Nettoyer la réponse du modèle"""
        return response.split("[/INST]")[-1].strip().strip('"').strip()

    def ask_question(self, field, template_question):
        """Poser une question de manière naturelle"""
        question = conversation_chain.run({
            "history": "\n".join(self.conversation_history),
            "next_question": template_question
        })
        return self.clean_response(question)    # question propre

    def extract_info(self, field, user_input):
        """Extraire l'information de la réponse utilisateur"""
        extracted = extraction_chain.run({
            "user_input": user_input,
            "current_info": json.dumps(self.reservation_info, ensure_ascii=False),
            "current_field": field
        })
        return self.clean_response(extracted)


    #fonctions de validation
    def validate_field(self, field, value ) -> Tuple[bool, str]:
        """Valide le champ selon des règles spécifiques et retourne un message d'erreur si invalide"""
        if field == "telephone":
            if not re.match(r'^0\d{9}$', value):
                return False, "Le numéro ne doit commencer par 0 et contenir exactement 10 chiffres (ex: 0612345678)"
            return True, ""

        elif field == "heure":
            if not re.match(r'^([01]?[0-9]|2[0-3])[^0-9]([0-5][0-9])$', value):
                return False, "L'heure doit être au format HH:MM "
            return True, ""
        elif field == "date":
            if not re.match(r'^(0[1-9]|[12][0-9]|3[01])[^0-9](0[1-9]|1[0-2])[^0-9](202[5-9]|20[3-9]\d|2[1-9]\d{2})$', value):
                return False, "La date doit être au format jj/mm/aaaa"
            return True, ""
        return True, ""

    def ask_validation_feedback(self, field: str, error_msg: str, template_question: str) -> str:
        """Génère un message d'erreur personnalisé en utilisant le LLM"""
        feedback = validation_chain.run({
            "field": field,
            "error_msg": error_msg,
            "template_question": template_question
        })
        return self.clean_response(feedback)




### confirm_info a supprimer
    def confirm_info(self, field, value):
        """Demander confirmation pour une information"""
        prompt = f"""<s>[INST]
Vous êtes un assistant de réservation de taxi. Demandez confirmation que {field} est bien "{value}".
Formulez cette demande de manière polie et naturelle. [/INST]"""
        return self.clean_response(llm(prompt))



    def run_reservation(self):
        """Exécuter le processus de réservation"""
        st.title(" Réservation de Taxi")
        # Afficher l'historique
        for msg in self.conversation_history:
            st.write(msg)


        for field, template_question in self.questions_flow:
            valid_field = True   #  dans le cas unvalide  en evite la repition du question
            while not self.reservation_info[field]:
                if valid_field  :
                  # Poser la question
                  question = self.ask_question(field, template_question)
                  st.write(f"**Assistant**: {question}")

                user_input = st.chat_input("Votre réponse...")

                # Extraire l'information
                extracted = self.extract_info(field, user_input)


                # Valider le champ
                is_valid, error_msg = self.validate_field(field, extracted)

                if not is_valid:
                    feedback = self.ask_validation_feedback(field, error_msg, template_question)
                    st.error(feedback)
                    valid_field = False
                    continue

                #Enregistrer si valide
                self.reservation_info[field] = extracted
                # Enregistrer dans l'historique
                self.conversation_history.append(f"Assistant: {question}\nClient: {user_input}")
                valid_field = True





        # Affichage final
        if all(self.reservation_info.values()):
            st.success("✅ Réservation confirmée !")
            st.json(self.reservation_info)





# 5. Exécution du code
if __name__ == "__main__":
    agent = TaxiReservationSystem()
    agent.run_reservation()

