import requests
import streamlit as st

st.title("USA Jobs Classifier Demo")
st.caption("Use the USA Jobs Location Negotiable Classifier to predict whether a listed job's \n location is negotiable. Paste the job description below and click 'Predict'.")

job_description = st.text_area("Enter a Job Description:", "", 300, None)

if st.button("Predict"):
    if job_description != "":
        # GET URL is "Public IPv4 DNS" of running EC2 instance, then normal api url formatting 
        response = requests.get(f"http://127.0.0.1:3400/predict/{job_description}")
        st.write("Prediction: ", response.text)