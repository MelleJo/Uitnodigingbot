import streamlit as st
from openai import OpenAI
from streamlit_mic_recorder import mic_recorder
import os
import time
import pytz
import tempfile
from datetime import datetime
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import AnalyzeDocumentChain
from langchain_community.callbacks import get_openai_callback
from langchain.chains.question_answering import load_qa_chain
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from fuzzywuzzy import process
from docx import Document
#from pydub import AudioSegment
import streamlit.components.v1 as components


st.title("Uitnodigingsbot")

# Definieer variabelen op een hoger niveau om scope-problemen te voorkomen
uploaded_audio = None
uploaded_text = None


client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

def split_audio(file_path, max_size=24000000):
    audio = AudioSegment.from_file(file_path)
    duration = len(audio)
    chunks_count = max(1, duration // (max_size / (len(audio.raw_data) / duration)))

    # Als chunks_count 1 is, retourneer de hele audio in één stuk
    if chunks_count == 1:
        return [audio]

    # Anders, splits de audio in de berekende aantal chunks
    return [audio[i:i + duration // chunks_count] for i in range(0, duration, duration // int(chunks_count))]

def transcribe_audio(file_path):
    with st.spinner("Transcriptie maken..."): 
        transcript_text = ""
        try:
            audio_segments = split_audio(file_path)
            for segment in audio_segments:
                with tempfile.NamedTemporaryFile(delete=True, suffix='.wav') as temp_file:
                    segment.export(temp_file.name, format="wav")
                    with open(temp_file.name, "rb") as audio_file:
                        transcription_response = client.audio.transcriptions.create(file=audio_file, model="whisper-1")
                        if hasattr(transcription_response, 'text'):
                            transcript_text += transcription_response.text + " "
            return transcript_text.strip()
        except Exception as e:
            st.error(f"Transcription failed: {str(e)}")
            return "Transcription mislukt."
        
def generate_uitnodiging(text):
    with st.spinner("Een uitnodigingsverzoek maken..."):
        prompt = "Schrijf een verzoek tot uitnodiging voor mijn collega, je vult op basis van de gebruikers input {text} de volgende velden in: wanneer, welke bedrijfsarts, soort afspraak, vraagstelling, extra vraag, welke taken maken in XS, toevoegen aan spreekuren overzicht (ja/nee), is het consult inclusief/exclusief, is facturatie nodig (ja/nee)"

        chat_model = ChatOpenAI(api_key=st.secrets["OPENAI_API_KEY"], model="gpt-4-0125-preview", temperature=0)
        prompt_template = ChatPromptTemplate.from_template(prompt)
        llm_chain = prompt_template | chat_model | StrOutputParser()
        
        try:
            uitnodiging_text = llm_chain.invoke({})
            if not uitnodiging_text:
                uitnodiging_text = "Mislukt om een uitnodiging te genereren."
        except Exception as e:
            st.error(f"Fout bij het genereren van uitnodiging: {e}")
            uitnodiging_text = "Mislukt om een uitnodiging te genereren."

        return uitnodiging_text

        
input_method = st.radio("Hoe wil je de uitnodiging genereren?", ["Audio", "Tekst"])

if input_method == "Audio":
    audio_data = mic_recorder(
            key="recorder",
            start_prompt="Start opname",
            stop_prompt="Stop opname",
            use_container_width=True,
            format="webm"
        )
    if audio_data and 'bytes' in audio_data:
            uploaded_audio = audio_data['bytes']
elif input_method == "Tekst":
    uploaded_text = st.text_area("Vul hieronder je tekst in:", height=200)

if uploaded_audio or uploaded_text:
    if uploaded_text:
        uitnodiging = generate_uitnodiging(uploaded_text)
    else:
        uitnodiging = generate_uitnodiging(uploaded_audio)
    st.write(uitnodiging)


        



