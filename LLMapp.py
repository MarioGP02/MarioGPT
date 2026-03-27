import streamlit as st
from groq import Groq
import PyPDF2

st.set_page_config(page_title="MarioGPT", page_icon="🤖")
st.title("🤖 MarioGPT con Llama 3.1 8b instant")

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
        base_system_prompt = "Eres MarioGPT, un asistente inteligente, no tienes nada que ver con el reconocido videojuego Mario bros. Responde siempre de forma amigable y en español."
        
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