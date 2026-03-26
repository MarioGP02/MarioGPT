import streamlit as st
from groq import Groq

# 1. Configuración de la página
st.set_page_config(page_title="MarioGPT", page_icon="🤖")
st.title("MarioGPT con Llama 3.1-8B Instant")

# 2. Inicializar el cliente
client = Groq(api_key=st.secrets["GROQ_API_KEY"])

# 3. Inicializar el historial de chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# 4. Mostrar mensajes anteriores
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 5. Input del usuario
if prompt := st.chat_input("¿En qué puedo ayudarte?"):
    # Guardar y mostrar mensaje del usuario
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 6. Respuesta del LLM
    with st.chat_message("assistant"):
        response_placeholder = st.empty() # Espacio para la respuesta
        full_response = ""
        
        try:
            # Preparamos los mensajes incluyendo el System Prompt
            messages_to_send = [
                {"role": "system", "content": "Eres MarioGPT, un asistente inteligente experto en Sevilla y el barrio de Montequinto. Eres amable, directo y conoces bien la zona."}
            ] + [
                {"role": m["role"], "content": m["content"]} for m in st.session_state.messages
            ]

            # Llamada a la API (He bajado al 8B para evitar el BadRequest por límites de saturación)
            stream = client.chat.completions.create(
                messages=messages_to_send,
                model="llama-3.1-8b-instant", 
                stream=True,
            )
            
            # Usamos write_stream para el efecto de escritura en tiempo real
            full_response = st.write_stream(stream)
            
            # Guardamos la respuesta en el historial solo si hubo éxito
            st.session_state.messages.append({"role": "assistant", "content": full_response})

        except Exception as e:
            st.error(f"Vaya, algo ha fallado en la conexión: {str(e)}")
            # Opcional: imprimir el error en consola para debug
            print(f"DEBUG ERROR: {e}")