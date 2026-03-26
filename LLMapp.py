import streamlit as st
from groq import Groq

st.set_page_config(page_title="MarioGPT", page_icon="🤖")
st.title("🤖 MarioGPT con Llama 3.1 8b instant")

client = Groq(api_key=st.secrets["GROQ_API_KEY"])

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("En que puedo ayudarte hoy?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # Creamos un contenedor vacío para ir "escribiendo" el texto
        response_placeholder = st.empty()
        full_response = ""
        
        # Llamada a la API
        completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "Eres MarioGPT, un asistente inteligente. Responde siempre de forma amigable y en español."},
                *[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
            ],
            model="llama-3.1-8b-instant",
            stream=True,
        )

        # Iteramos sobre el stream extrayendo ÚNICAMENTE el texto
        for chunk in completion:
            # Esta es la clave: accedemos al contenido exacto del delta
            content = chunk.choices[0].delta.content
            if content:
                full_response += content
                # Actualizamos el contenedor con el texto acumulado
                response_placeholder.markdown(full_response + "▌")
        
        # Al terminar, quitamos el cursor "▌" y guardamos
        response_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})