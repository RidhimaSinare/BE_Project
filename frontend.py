import streamlit as st
import requests

# Streamlit UI setup
st.set_page_config(page_title="Marathi Sentence Classifier", layout="centered")

# Title and description
st.title("🔍 Marathi Sentence Classification")
st.markdown("Enter a **Marathi sentence** below, and the model will classify it into different categories.")

# Define specific task names
task_names = [
    "🚨 Hate Speech Detection",
    "😊 Sentiment Analysis",
    "📰 News Category",
    "🎭 Topic Classification",
    "⚠️ Offensive Content Detection"
]

# Input field
user_input = st.text_area("📝 Enter a Marathi sentence:", height=100)

# Button to trigger classification
if st.button("🚀 Classify"):
    if user_input.strip():
        with st.spinner("🔄 Analyzing... Please wait"):
            # Send request to FastAPI backend
            response = requests.post("http://127.0.0.1:8000/predict", json={"text": user_input})

            if response.status_code == 200:
                predictions = response.json()["predictions"]

                # Display predictions with task names
                st.subheader("📌 Predictions:")
                for idx, pred in enumerate(predictions):
                    st.markdown(f"**{task_names[idx]}**: `{pred}`")

            else:
                st.error("❌ Error fetching predictions.")
    else:
        st.warning("⚠️ Please enter a sentence.")
