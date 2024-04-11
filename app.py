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
#from pydub import AudioSegment
import streamlit as st
from openai import OpenAI
from streamlit_mic_recorder import mic_recorder
import os
import tempfile
from datetime import datetime
import pytz
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Initialisatie
st.title("Uitnodigingsbot")
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

def transcribe_audio(audio_bytes):
    with tempfile.NamedTemporaryFile(delete=True, suffix='.wav') as tmp_file:
        tmp_file.write(audio_bytes)
        tmp_file.flush()
        transcription_response = client.audio.transcriptions.create(
            file=open(tmp_file.name, "rb"), 
            model="whisper-1"
        )
        return transcription_response['text'] if transcription_response else "Transcriptie mislukt."

def generate_uitnodiging(input_text):
    prompt_text = f"Schrijf een uitnodiging op basis van de input: {input_text}"
    chat_model = ChatOpenAI(api_key=st.secrets["OPENAI_API_KEY"], model="gpt-4-0125-preview", temperature=0)
    prompt_template = ChatPromptTemplate.from_template(prompt_text)
    llm_chain = prompt_template | chat_model | StrOutputParser()
    return llm_chain.invoke({})

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
        transcript = transcribe_audio(audio_data['bytes'])
        uitnodiging = generate_uitnodiging(transcript)
        st.text_area("Gegenereerde uitnodiging:", uitnodiging, height=250)
elif input_method == "Tekst":
    uploaded_text = st.text_area("Vul hieronder je tekst in:", height=200)
    if st.button("Genereer Uitnodiging"):
        uitnodiging = generate_uitnodiging(uploaded_text)
        st.text_area("Gegenereerde uitnodiging:", uitnodiging, height=250)
