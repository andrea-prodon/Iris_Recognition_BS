import os
import time
import json
import httpx
import streamlit as st



PREDICTION_SERVICE_HOSTNAME = '127.0.0.1'#os.environ["PREDICTION_SERVICE_HOSTNAME"]
PREDICTION_SERVICE_PORT = '8000'#os.environ["PREDICTION_SERVICE_PORT"]

st.set_page_config(
     page_title="Sign in with your iris?",
     layout="centered"
 )

md_text = """
            <style>
            #MainMenu {visibility: hidden;}
            </style>
            # Sign in with your iris.
          """
st.markdown(md_text, unsafe_allow_html=True)

uploaded_image = st.file_uploader("Upload image")

if uploaded_image is not None:
    st.image(uploaded_image)

submitted = st.button("Submit")
if submitted and uploaded_image is not None:
    with st.spinner("Processing image..."):
                start_time = time.time()
                response = httpx.post(
                                f'http://{PREDICTION_SERVICE_HOSTNAME}:{PREDICTION_SERVICE_PORT}/recognition',
                                files={'file': uploaded_image},
                                timeout=60.0)
                if response.status_code == 200:
                    response_dict = response.json()
                    st.success(f'Welcome {response_dict["subject"]}')
                elif response.status_code == 422:
                    st.error("No iris detected")
                elapsed_time = time.time() - start_time
    st.success('Done in %.2fs' % elapsed_time)