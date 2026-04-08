import streamlit as st
import requests
import time

# URL actualizada
API_URL = "https://router.huggingface.co/hf-inference/models/black-forest-labs/FLUX.1-schnell"
headers = {"Authorization": f"Bearer {st.secrets['HUGGINGFACE_API_KEY']}"}

def pedir_imagen_a_api(prompt):
    """Hace la petición simple y devuelve la respuesta pura"""
    response = requests.post(API_URL, headers=headers, json={"inputs": prompt})
    return response

def generar_imagen(prompt):
    payload = {"inputs": prompt}
    response = requests.post(API_URL, headers=headers, json=payload)
    
    # --- LA CORRECCIÓN ESTÁ AQUÍ ---
    # Revisamos el 'content-type' en los headers
    content_type = response.headers.get("content-type", "")
    
    if "image" not in content_type:
        # Si no es una imagen, es que Hugging Face nos ha mandado un JSON con un error o aviso
        error_info = response.json()
        
        # A veces el modelo está cargando ("loading"), podemos dar un mensaje más específico
        if "estimated_time" in error_info:
            tiempo = round(error_info["estimated_time"], 1)
            raise Exception(f"El modelo se está despertando. Estará listo en unos {tiempo} segundos. ¡Inténtalo de nuevo ahora!")
        
        raise Exception(f"Error de la API: {error_info.get('error', 'Desconocido')}")
        
    return response.content # Si todo está bien, devuelve los bytes de la imagen