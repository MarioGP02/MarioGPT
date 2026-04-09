import streamlit as st
from supabase import create_client

SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# ---------------- AUTH ---------------- #

def login(email, password):
    return supabase.auth.sign_in_with_password({
        "email": email,
        "password": password
    })

def register(email, password):
    return supabase.auth.sign_up({
        "email": email,
        "password": password
    })

# ---------------- RECOVERY ---------------- #

def enviar_recuperacion(email):
    # Supabase enviará un email con un enlace a este usuario
    # Requiere que el usuario exista en la base de datos
    return supabase.auth.reset_password_for_email(email)

def actualizar_contraseña(nueva_contraseña):
    # Esto actualiza la contraseña del usuario que está actualmente logueado
    return supabase.auth.update_user({
        "password": nueva_contraseña
    })

# ---------------- DB ---------------- #

def save_message(user_id, role, content):
    supabase.table("messages").insert({
        "user_id": user_id,
        "role": role,
        "content": content
    }).execute()

def load_messages(user_id):
    res = supabase.table("messages") \
        .select("role, content") \
        .eq("user_id", user_id) \
        .order("created_at") \
        .execute()

    # 🔥 FILTRO CLAVE
    clean_messages = [
        {"role": m["role"], "content": m["content"]}
        for m in res.data
        if m["content"] is not None and m["content"] != ""
    ]

    return clean_messages

