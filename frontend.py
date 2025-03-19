import streamlit as st
import requests

# Streamlit UI
st.title("Marathi Sentence Classification")

# Input field
user_input = st.text_area("Enter a Marathi sentence:")

# Button to trigger classification
if st.button("Classify"):
    if user_input.strip():
        # Send request to FastAPI backend
        response = requests.post("http://127.0.0.1:8000/predict", json={"text": user_input})
        
        if response.status_code == 200:
            predictions = response.json()["predictions"]
            
            # Display predictions in a structured way
            st.write("### Predictions:")
            task_names = ["Hate Speech 4 Class", "Sentiment", "SHC", "LPC", "Hate Speech 2 Class"]  
            
            for idx, pred in enumerate(predictions):
                st.write(f"**{task_names[idx]}:** {pred}")
        
        else:
            st.error("Error fetching predictions from the backend.")
    else:
        st.warning("Please enter a sentence.")
