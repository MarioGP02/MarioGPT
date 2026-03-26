import streamlit as st
from groq import Groq

# Configuración de la página
st.set_page_config(page_title="MarioGPT", page_icon="🤖")
st.title("MarioGPT con Llama 3 (Vía Groq)")

# Inicializar el cliente (usando secretos de Streamlit para seguridad)
client = Groq(api_key=st.secrets["GROQ_API_KEY"])

# Inicializar el historial de chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Mostrar mensajes anteriores
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input del usuario
if prompt := st.chat_input("¿En qué puedo ayudarte?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Respuesta del LLM
    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            messages=[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages],
            model="llama-3.3-70b-versatile", # Modelo potente y rápido
            stream=True,
        )
        response = st.write_stream(stream)
    st.session_state.messages.append({"role": "assistant", "content": response})