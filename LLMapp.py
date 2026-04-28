import streamlit as st
from groq import Groq
import PyPDF2
import io
from PIL import Image
import requests
import base64
from huggingface_hub import hf_hub_download
from supabase_utils import *
from tavily_utils import *
from imagenes_utils import *
from marioGPT_core import *

st.set_page_config(page_title="MarioGPT", page_icon="🤖")
st.title("🤖 MarioGPT")

# --- AUTH ---
if "user" not in st.session_state:
    st.session_state.user = None

# Si el usuario no está logueado, lo dejamos abierto por defecto. Si ya entró, lo cerramos.
auth_expanded = True if not st.session_state.user else False

with st.sidebar.expander("🔐 Cuenta y Acceso", expanded=auth_expanded):
    if not st.session_state.user:
        # Añadimos "Recuperar" a las opciones
        mode = st.selectbox("Acceso", ["Login", "Registro", "Recuperar contraseña"])
        
        email = st.text_input("Email")
        
        # Ocultamos el campo de contraseña si solo quiere recuperar la cuenta
        if mode != "Recuperar contraseña":
            password = st.text_input("Contraseña", type="password")

        if mode == "Registro":
            if st.button("Crear cuenta"):
                res = register(email, password)
                if res.user:
                    st.success("Cuenta creada. ¡Ya puedes entrar!")
                else:
                    st.error("Error al crear cuenta")

        elif mode == "Login":
            if st.button("Entrar"):
                try:
                    res = login(email, password)
                    if res.user:
                        st.session_state.user = res.user
                        st.success("Login correcto")
                        st.rerun()
                except Exception as e:
                    st.error(f"Error de credenciales. ¿Te has equivocado?")
                    
        elif mode == "Recuperar contraseña":
            if st.button("Enviar enlace de recuperación"):
                try:
                    # Llamamos a la nueva función
                    enviar_recuperacion(email)
                    st.success("📩 Revisa tu correo. Te hemos enviado un enlace para cambiar tu contraseña.")
                except Exception as e:
                    st.error(f"Error: {e}")
    else:
        st.write(f"Conectado como: **{st.session_state.user.email}**")
        
        # --- NUEVO: FORMULARIO PARA CAMBIAR CONTRASEÑA ---
        st.divider()
        nueva_pass = st.text_input("Nueva contraseña", type="password", key="new_pass")
        if st.button("Actualizar contraseña"):
            try:
                actualizar_contraseña(nueva_pass)
                st.success("✅ ¡Contraseña actualizada con éxito!")
            except Exception as e:
                st.error(f"Error al actualizar: {e}")
        # Logout       
        st.divider()
        if st.button("Cerrar sesión"):
            st.session_state.user = None
            st.session_state.messages = []
            if "messages_loaded" in st.session_state:
                del st.session_state.messages_loaded
            st.rerun()

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

if "image_context" not in st.session_state:
    st.session_state.image_context = None

# --- CARGAR HISTORIAL Y CARGAR MODELO MarioGPT---
@st.cache_resource(show_spinner="Cargando cerebro de MarioGPT...")
def cargar_mariogpt_local():
    path_modelo = hf_hub_download(repo_id="MarioGP/MarioGPTitan", filename="MarioGPTitan_Int8.pth")
    
    # 1. Instanciamos el modelo normal
    model_base = MarioLLM()
    
    # 2. Aplicamos la cuantización DINÁMICA al modelo vacío 
    # (Esto crea la estructura necesaria para los pesos Int8)
    model_quantized = torch.quantization.quantize_dynamic(
        model_base, 
        {torch.nn.Linear}, 
        dtype=torch.qint8
    )
    
    # 3. Cargamos los pesos comprimidos
    state_dict = torch.load(path_modelo, map_location='cpu')
    model_quantized.load_state_dict(state_dict)
    model_quantized.eval()
    
    enc = tiktoken.get_encoding("gpt2")
    return model_quantized, enc, 'cpu'



if "last_user_id" not in st.session_state or st.session_state.last_user_id != st.session_state.user.id:
    st.session_state.messages = load_messages(st.session_state.user.id)
    st.session_state.last_user_id = st.session_state.user.id
    st.session_state.messages_loaded = True


# Diccionario de modelos disponibles en Groq
modelos_disponibles = {
    # 📝 --- MODELOS DE TEXTO PÚRO Y RAZONAMIENTO ---
    "MarioGPT 3.0(Local, sin conexión)": "mariogpt_local", # <--- ID ESPECIAL para usar tu modelo local
    "Llama 3.3 70B (Máxima Inteligencia)": "llama-3.3-70b-versatile",
    "Llama 3.1 8B (Rápido y eficaz)": "llama-3.1-8b-instant",
    "Mixtral 8x7B (Equilibrado/Contexto largo)": "mixtral-8x7b-32768",
    "Gemma 2 9B (Modelo de Google)": "gemma2-9b-it",
    
    # 👁️ --- MODELOS MULTIMODALES (VISIÓN) ---
    "Llama 3.2 90B (Análisis profundo de imágenes)": "llama-3.2-90b-vision-preview",
    "Llama 3.2 11B (Análisis rápido de imágenes)": "llama-3.2-11b-vision-preview",
    
    # 🎨 --- MODELO DE CREACIÓN (Usa Hugging Face) ---
    "FLUX.1 [schnell] (Crear imágenes desde texto)": "generador_imagenes" # <--- ID ESPECIAL
}

st.sidebar.divider()
st.sidebar.subheader("🤖 Configuración del Modelo")

# El usuario elige el nombre amigable
modelo_nombre = st.sidebar.selectbox(
    "Selecciona el modelo LLM:",
    options=list(modelos_disponibles.keys()),
    index=0 # Por defecto el primero
)

# Obtenemos el ID técnico para la API
modelo_actual = modelos_disponibles[modelo_nombre]

# Pequeña alerta visual si el usuario elige un modelo de visión
if "Análisis" in modelo_nombre:
    st.sidebar.info("📷 Has seleccionado un modelo de Visión. ¡Asegúrate de subir una imagen para aprovecharlo!")
else:
    st.sidebar.caption("Modelo de texto puro seleccionado.")
# Info visual
if "FLUX" in modelo_nombre:
    st.sidebar.info("🎨 Has seleccionado el Generador de Imágenes. MarioGPT creará una imagen basada en lo que escribas.")

# --- SIDEBAR ARCHIVOS ---
with st.sidebar.expander("📂 Análisis de Documentos e Imágenes", expanded=False):
    uploaded_file = st.file_uploader("Soporta: TXT, PDF, JPG, PNG, JPEG", type=["txt", "pdf", "jpg", "jpeg", "png"])

    if uploaded_file:
        # Reiniciamos contextos por si cambia de archivo
        st.session_state.file_context = ""
        st.session_state.image_context = None

        # Si es una IMAGEN
        if uploaded_file.name.lower().endswith((".jpg", ".jpeg", ".png")):
            image_bytes = uploaded_file.getvalue()
            base64_image = base64.b64encode(image_bytes).decode("utf-8")
            mime_type = uploaded_file.type
            
            # Guardamos la imagen en el formato que exige Groq/OpenAI
            st.session_state.image_context = f"data:{mime_type};base64,{base64_image}"
            
            st.sidebar.image(uploaded_file, caption="Imagen lista para analizar", use_container_width=True)
            st.success("¡Imagen procesada! Recuerda usar un modelo de Visión.")

        # Si es un PDF
        elif uploaded_file.name.endswith(".pdf"):
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            text_content = ""
            for page in pdf_reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text_content += extracted
            st.session_state.file_context = text_content[:20000]
            st.success("Archivo PDF procesado")

        # Si es un TXT
        else:
            text_content = uploaded_file.getvalue().decode("utf-8")
            st.session_state.file_context = text_content[:20000]
            st.success("Archivo TXT procesado")
    else:
        st.session_state.file_context = ""
        st.session_state.image_context = None

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
                    - Si no conoces la respuesta, admítelo de forma honesta

                    Responde siempre:
                    - De forma amigable
                    - En español
                    - De manera clara y útil
        """

        if st.session_state.file_context:
            base_system_prompt += f"\n\n{st.session_state.file_context}"

        try:
            if modelo_actual == "generador_imagenes":
                with st.status("🎨 Preparando pinceles...", expanded=True) as status:
                    intentos = 0
                    max_intentos = 15 # Máximo de reintentos (aprox 1-2 minutos total)
                    exito = False
                
                    while intentos < max_intentos and not exito:
                        res = pedir_imagen_a_api(prompt)
                        content_type = res.headers.get("content-type", "")

                        if "image" in content_type:
                            # ¡ÉXITO! Tenemos la imagen
                            st.image(res.content, caption=f"MarioGPT generó: {prompt}", use_container_width=True)
                            status.update(label="✅ ¡Imagen terminada!", state="complete")
                            full_response = f"He terminado tu imagen de: '{prompt}'"
                            exito = True
                        else:
                            # EL MODELO ESTÁ CARGANDO O HAY ERROR
                            try:
                                error_data = res.json()
                                if "estimated_time" in error_data:
                                    tiempo_espera = error_data["estimated_time"]
                                    status.update(label=f"⏳ El modelo está despertando... (Espera {round(tiempo_espera)}s)", state="running")
                                    # Esperamos 10 segundos antes de reintentar para no saturar
                                    time.sleep(10)
                                    intentos += 1
                                else:
                                    raise Exception(error_data.get("error", "Error desconocido"))
                            except:
                                status.update(label="❌ Error en la API", state="error")
                                break
                
                if not exito:
                    st.error("Lo siento, el modelo tardó demasiado en responder. Prueba de nuevo en unos segundos.")
                    full_response = "Error por tiempo de espera agotado."

            elif modelo_actual == "mariogpt_local":
                with st.status("🧠 MarioGPT revisando su memoria...", expanded=True) as status:
                    # 1. Carga de recursos
                    modelo_local, enc_local, device_local = cargar_mariogpt_local()
        
                    # 2. Construcción de la memoria (Contexto)
                    contexto_memoria = ""
                    for m in st.session_state.messages[-6:]:
                        role_label = "Usuario" if m["role"] == "user" else "Asistente"
                        contexto_memoria += f"{role_label}: {m['content']}\n"
        
                    texto_entrada = f"{contexto_memoria}Asistente:"
        
                    # 3. Tokenización (De Español a Integers)
                    tokens_input = enc_local.encode(texto_entrada)
        
                    # Recorte de seguridad para no exceder los 512 tokens del MarioGPT
                    limite_tokens = 512 - 150 
                    if len(tokens_input) > limite_tokens:
                        tokens_input = tokens_input[-limite_tokens:]
            
                    context_tensor = torch.tensor(tokens_input, dtype=torch.long, device=device_local).unsqueeze(0)
        
                    status.update(label="Escribiendo respuesta basada en el historial...", state="running")
        
                    # 4. Generación (El modelo produce nuevos Integers)
                    with torch.no_grad():
                        generado_idx_completo = modelo_local.generate(
                            context_tensor, 
                            max_new_tokens=150, 
                            temperature=0.8, 
                            top_p=0.85
                        )[0].tolist()
        
                    # 5. DECODIFICACIÓN "QUIRÚRGICA" (De Integers a Español)
                    # En lugar de decodificar todo y cortar el texto, cortamos la lista de números primero.
                   # Esto es más eficiente y evita errores con espacios en blanco.
                    tokens_nuevos = generado_idx_completo[len(tokens_input):]
                    full_response = enc_local.decode(tokens_nuevos).strip()
        
                    # 6. Limpieza final de alucinaciones
                    # Si el modelo intenta seguir la conversación solo, cortamos por lo sano.
                    full_response = full_response.split("Usuario:")[0].split("Asistente:")[0].strip()
        
                    # 7. Renderizado en la App
                    response_placeholder.markdown(full_response)
                    status.update(label="✅ Memoria procesada con éxito", state="complete")
            else:

                # 1. EJECUTAR LA BÚSQUEDA (Aquí usas tu función)
                with st.status("🌐 Buscando información actualizada...", expanded=False) as status:
                    informacion_web = buscar_en_internet(prompt) # 👈 Llamas a tu función
                    status.update(label="✅ Información web obtenida", state="complete")

                # 2. PREPARAR LOS MENSAJES (El "Contexto")
                # Empezamos con el prompt de sistema y la info de internet
                messages_to_send = [
                    {"role": "system", "content": base_system_prompt},
                    {"role": "system", "content": f"DATOS DE INTERNET ACTUALIZADOS:\n{informacion_web}"}
                ]
                # 🔥 FILTRAR MENSAJES ANTES DE ENVIAR
                valid_messages = [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                    if m["content"] and m["content"].strip() != ""
                ]

                # 1. Preparamos una lista única para enviar a la API
                #messages_to_send = [
                #    {"role": "system", "content": base_system_prompt}
                #]

                # 2. Aplicamos la lógica del límite de memoria (ej. 30 mensajes)
                if len(valid_messages) > 30:
                    resumen = "Resumen: el usuario ha estado hablando sobre diversos temas en el pasado."
                    messages_to_send.append({"role": "system", "content": resumen})
                    messages_to_send.extend(valid_messages[-30:]) # Solo recordamos los últimos 30
                else:
                    messages_to_send.extend(valid_messages) # Recordamos todo el historial corto

                # Comprobamos si el modelo seleccionado soporta visión y si hay una imagen subida
                if "vision" in modelo_actual and st.session_state.image_context:
                    # Recorremos la lista al revés para encontrar el mensaje más reciente del usuario
                    for i in range(len(messages_to_send) - 1, -1, -1):
                        if messages_to_send[i]["role"] == "user":
                            texto_original = messages_to_send[i]["content"]
                            messages_to_send[i]["content"] = [
                                {"type": "text", "text": texto_original},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": st.session_state.image_context
                                    }
                                }
                            ]
                            break

                # 3. Enviamos la variable correcta a Groq
                completion = client.chat.completions.create(
                    messages=messages_to_send, #AHORA SÍ ENVIAMOS EL CONTEXTO CORRECTO
                    model=modelo_actual, # El modelo seleccionado dinámicamente
                    stream=True,
                )

                for chunk in completion:
                    content = chunk.choices[0].delta.content
                    if content:
                        full_response += content
                        response_placeholder.markdown(full_response + "▌")

                response_placeholder.markdown(full_response)

                #Mostrar contexto
                #st.write(len(valid_messages))
                #st.write(valid_messages[:2])  # primeros mensajes
                #st.write(valid_messages[-2:]) # últimos mensajes
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
