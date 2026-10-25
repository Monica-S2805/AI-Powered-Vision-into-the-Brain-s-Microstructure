import requests
import streamlit as st

st.set_page_config(page_title="NeuroBot", layout="wide", page_icon="🤖")
st.title("NeuroBot")
st.markdown("##### Welcome to NeuroBot AI Assistant, your companion for brain disease information!")


# OpenRouter API setup
API_URL = "https://openrouter.ai/api/v1/chat/completions"
API_KEY = "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"  # Replace this with your actual key
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# Function to call OpenRouter API
def query_openrouter(prompt):
    payload = {
        "model": "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",  # Or any other available model
        "messages": [
            {"role": "system", "content": "You are a helpful assistant knowledgeable in brain diseases."},
            {"role": "user", "content": prompt}
        ]
    }
    response = requests.post(API_URL, headers=HEADERS, json=payload)
    print("DEBUG:", response.status_code, response.text)

    try:
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return "OOOOOOOPPPPPSSSSSS! Sorry, I'm having trouble right now. Try again after sometime."

# Session state setup
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Hi! How can I help you? "):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = query_openrouter(prompt)
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

