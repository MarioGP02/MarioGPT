# --- EN UN ARCHIVO DE UTILIDADES (ej. image_utils.py) ---
import streamlit as st
import requests
import io
from PIL import Image

# Usamos FLUX.1-schnell, que es el modelo gratuito más rápido y potente actualmente
API_URL = "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-schnell"
headers = {"Authorization": f"Bearer {st.secrets['HUGGINGFACE_API_KEY']}"}

def generar_imagen(prompt):
    payload = {"inputs": prompt}
    response = requests.post(API_URL, headers=headers, json=payload)
    
    # Si la API está cargando el modelo, Hugging Face devuelve un JSON con el tiempo estimado
    if response.status_type != "image/png" and response.status_type != "image/jpeg":
        raise Exception(f"La API está cargando o hubo un error: {response.text}")
        
    return response.content # Devuelve los bytes de la imagen