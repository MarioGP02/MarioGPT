# --- EN TU ARCHIVO DE UTILIDADES O APP ---
import streamlit as st
from tavily import TavilyClient

tavily = TavilyClient(api_key=st.secrets["TU_TAVILY_API_KEY"])

def buscar_en_internet(query):
    # Realiza la búsqueda y devuelve un resumen del contenido
    respuesta = tavily.search(query=query, search_depth="basic")
    contexto_web = "\n".join([f"Fuente: {r['url']}\nContenido: {r['content']}" for r in respuesta['results']])
    return contexto_web