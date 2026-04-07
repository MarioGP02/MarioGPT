import streamlit as st
from groq import Groq
import PyPDF2
from supabase_utils import *

st.set_page_config(page_title="MarioGPT", page_icon="🤖")
st.title("🤖 MarioGPT con Llama 3.1 8b instant")

# --- AUTH ---
if "user" not in st.session_state:
    st.session_state.user = None

st.sidebar.title("🔐 Cuenta")
mode = st.sidebar.selectbox("Acceso", ["Login", "Registro"])
email = st.sidebar.text_input("Email")
password = st.sidebar.text_input("Contraseña", type="password")

if mode == "Registro":
    if st.sidebar.button("Crear cuenta"):
        res = register(email, password)
        if res.user:
            st.sidebar.success("Cuenta creada")
        else:
            st.sidebar.error("Error")

elif mode == "Login":
    if st.sidebar.button("Entrar"):
        try:
            res = login(email, password)
            if res.user:
                st.session_state.user = res.user
                st.sidebar.success("Login correcto")
        except Exception as e:
            st.sidebar.error(f"Error: {e}")

# Logout
if st.session_state.user:
    if st.sidebar.button("Cerrar sesión"):
        st.session_state.user = None
        st.session_state.messages = []
        if "messages_loaded" in st.session_state:
            del st.session_state.messages_loaded

# Bloquear acceso
if not st.session_state.user:
    st.warning("🔒 Inicia sesión para usar MarioGPT")
    st.stop()

# --- CLIENTE LLM ---
client = Groq(api_key=st.secrets["GROQ_API_KEY"])

# --- SESSION STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if "file_context" not in st.session_state:
    st.session_state.file_context = ""

# --- CARGAR HISTORIAL ---
if "last_user_id" not in st.session_state or st.session_state.last_user_id != st.session_state.user.id:
    st.session_state.messages = load_messages(st.session_state.user.id)
    st.session_state.last_user_id = st.session_state.user.id
    st.session_state.messages_loaded = True

# --- SIDEBAR ARCHIVOS ---
with st.sidebar:
    st.header("📂 Análisis de Documentos")
    uploaded_file = st.file_uploader("Soporta: TXT, PDF", type=["txt", "pdf"])

    if uploaded_file:
        text_content = ""
        if uploaded_file.name.endswith(".pdf"):
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            for page in pdf_reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text_content += extracted
        else:
            text_content = uploaded_file.getvalue().decode("utf-8")

        st.session_state.file_context = text_content[:20000]
        st.success("Archivo procesado")
    else:
        st.session_state.file_context = ""

# --- MOSTRAR HISTORIAL ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- CHAT ---
if prompt := st.chat_input("¿En qué puedo ayudarte hoy?"):

    # USER
    if prompt and prompt.strip():
        st.session_state.messages.append({"role": "user", "content": prompt})
        save_message(st.session_state.user.id, "user", prompt)

    if prompt and prompt.strip():
        with st.chat_message("user"):
            st.markdown(prompt)

    # ASSISTANT
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""

        base_system_prompt = """
                    Eres MarioGPT, un asistente inteligente.
                    No tienes nada que ver con el reconocido videojuego Mario Bros.
                    Si alguien te pregunta por tu creador, responde que se llama Mario, quien parametrizó tu modelo LLM descargado de forma gratuita de Meta.
                    - Da respuestas estructuradas
                    - Usa ejemplos
                    - Sé claro y práctico
                    - Evita respuestas largas innecesarias si el usuario no lo pide
                    - Si no conoces la respuesta, admítelo de forma honesta y ofrece buscar información adicional en internet (aunque no puedas hacerlo realmente)

                    Responde siempre:
                    - De forma amigable
                    - En español
                    - De manera clara y útil
        """

        if st.session_state.file_context:
            base_system_prompt += f"\n\n{st.session_state.file_context}"

        try:
            # 🔥 FILTRAR MENSAJES ANTES DE ENVIAR
            valid_messages = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
                if m["content"] and m["content"].strip() != ""
            ]

            completion = client.chat.completions.create(
                messages=[
                {"role": "system", "content": base_system_prompt},
                *valid_messages
                ],
                model="llama-3.1-8b-instant",
                stream=True,
            )

            for chunk in completion:
                content = chunk.choices[0].delta.content
                if content:
                    full_response += content
                    response_placeholder.markdown(full_response + "▌")

            response_placeholder.markdown(full_response)

            # GUARDAR RESPUESTA
            if full_response and full_response.strip():
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                save_message(st.session_state.user.id, "assistant", full_response)

        except Exception as e:
            error_msg = f"Error: {e}"
            response_placeholder.error(error_msg)

            st.session_state.messages.append({"role": "assistant", "content": error_msg})
            save_message(st.session_state.user.id, "assistant", error_msg)

# LIMPIAR SESIÓN SI LOGOUT
if "messages_loaded" in st.session_state and not st.session_state.user:
    st.session_state.messages = []
    del st.session_state.messages_loaded
