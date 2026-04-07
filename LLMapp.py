import streamlit as st
from groq import Groq
import PyPDF2
from supabase_utils import *

st.set_page_config(page_title="MarioGPT", page_icon="🤖")
st.title("🤖 MarioGPT con Llama 3.1 8b instant")

# --- 0.Loggin y Registro de Usuarios ---
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
        res = login(email, password)
        if res.user:
            st.session_state.user = res.user
            st.sidebar.success("Login correcto")
        else:
            st.sidebar.error("Credenciales incorrectas")

# Logout
if st.session_state.user:
    if st.sidebar.button("Cerrar sesión"):
        st.session_state.user = None

if not st.session_state.user:
    st.warning("🔒 Inicia sesión")
    st.stop()


if "messages_loaded" not in st.session_state:
    st.session_state.messages = load_messages(st.session_state.user.id)
    st.session_state.messages_loaded = True

# --- 1. Inicialización del Cliente ---
client = Groq(api_key=st.secrets["GROQ_API_KEY"])

# --- 2. Variables de Estado (Session State) ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "file_context" not in st.session_state:
    st.session_state.file_context = ""

# --- 3. Barra Lateral: Subida y Procesamiento de Archivos ---
with st.sidebar:
    st.header("📂 Análisis de Documentos")
    st.info("Sube un archivo para que MarioGPT lo lea y te ayude con su contenido.")
    
    # Widget de subida de archivos
    uploaded_file = st.file_uploader("Soporta: TXT, PDF", type=["txt", "pdf"])
    
    if uploaded_file is not None:
        try:
            text_content = ""
            
            # Lógica para extraer texto de PDFs
            if uploaded_file.name.lower().endswith(".pdf"):
                pdf_reader = PyPDF2.PdfReader(uploaded_file)
                for page in pdf_reader.pages:
                    extracted = page.extract_text()
                    if extracted:
                        text_content += extracted + "\n"
                        
            # Lógica para extraer texto de TXT u otros archivos de texto plano
            else:
                text_content = uploaded_file.getvalue().decode("utf-8")
            
            # Guardamos el texto en el estado. 
            # IMPORTANTE: Limitamos a ~20000 caracteres para evitar excepciones de Límite de Tokens en la API de Groq.
            st.session_state.file_context = text_content[:20000]
            st.success("✅ Archivo procesado correctamente. ¡Ya puedes hacer preguntas sobre él!")
            
        except Exception as e:
            st.error(f"Error al procesar el archivo. Asegúrate de que no esté corrupto. Detalle: {e}")
    else:
        # Si el usuario cierra o quita el archivo, limpiamos la memoria de contexto
        st.session_state.file_context = ""

# --- 4. Renderizado del Historial de Chat ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 5. Lógica Principal de Interacción ---
if prompt := st.chat_input("¿En qué puedo ayudarte hoy?"):
    # Guardar y mostrar el mensaje del usuario
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        
        # --- Construcción Dinámica del System Prompt ---
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
        
        # Si hay un archivo subido, inyectamos su contenido en las instrucciones del sistema
        if st.session_state.file_context:
            base_system_prompt += f"\n\n--- INICIO DEL DOCUMENTO PROPORCIONADO POR EL USUARIO ---\n{st.session_state.file_context}\n--- FIN DEL DOCUMENTO ---\n\nUtiliza la información de este documento para responder a las preguntas del usuario si es relevante."

        try:
            # Llamada a la API de Groq
            completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": base_system_prompt},
                    *[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
                ],
                model="llama-3.1-8b-instant",
                stream=True,
            )

            # Iteramos sobre el stream extrayendo ÚNICAMENTE el texto
            for chunk in completion:
                content = chunk.choices[0].delta.content
                if content:
                    full_response += content
                    response_placeholder.markdown(full_response + "▌")
            
            # Al terminar, quitamos el cursor y guardamos en el historial
            response_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
        except Exception as e:
            # Manejo de excepciones de la API (ej. caídas de servidor, errores de red)
            error_msg = f"Lo siento, hubo un problema al conectar con mi cerebro (API Error: {e})"
            response_placeholder.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})

st.session_state.messages.append({"role": "user", "content": prompt})
save_message(st.session_state.user.id, "user", prompt)

st.session_state.messages.append({"role": "assistant", "content": full_response})
save_message(st.session_state.user.id, "assistant", full_response)

if "messages_loaded" in st.session_state and not st.session_state.user:
    st.session_state.messages = []
    del st.session_state.messages_loaded