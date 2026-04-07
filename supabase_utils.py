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

    return res.data if res.data else []