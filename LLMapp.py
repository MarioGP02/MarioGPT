import streamlit as st
from groq import Groq

# 1. Configuración de la página
st.set_page_config(page_title="MarioGPT", page_icon="🤖")
st.title("🤖 MarioGPT con Llama 3.1 8b instant")

# 2. Inicializar el cliente (Asegúrate de tener GROQ_API_KEY en los Secrets de Streamlit)
client = Groq(api_key=st.secrets["GROQ_API_KEY"])

# 3. Inicializar el historial de chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# 4. Mostrar mensajes anteriores de forma limpia
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 5. Input del usuario
if prompt := st.chat_input("En que puedo ayudarte hoy?"):
    # Guardar y mostrar mensaje del usuario
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 6. Respuesta del LLM (Procesamiento del Stream)
    with st.chat_message("assistant"):
        try:
            # Preparamos los mensajes con el System Prompt
            messages_to_send = [
                {"role": "system", "content": "Eres MarioGPT, un asistente inteligente. Responde siempre de forma amigable y en español."}
            ] + [
                {"role": m["role"], "content": m["content"]} for m in st.session_state.messages
            ]

            # Llamada a la API
            completion = client.chat.completions.create(
                messages=messages_to_send,
                model="llama-3.1-8b-instant",
                stream=True,
            )
            
            # st.write_stream se encarga de extraer el 'content' de esos JSON que veías
            # y mostrar solo el texto letra a letra.
            full_response = st.write_stream(completion)
            
            # Guardamos la respuesta limpia en el historial
            st.session_state.messages.append({"role": "assistant", "content": full_response})

        except Exception as e:
            st.error(f"Error de conexión: {str(e)}")