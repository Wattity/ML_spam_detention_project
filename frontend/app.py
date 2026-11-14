import streamlit as st
import requests


st.set_page_config(page_title="Spam Detection", page_icon=":email:", layout="wide")
st.title("This is an Email Spam Detection App ")
st.write("This app categories your input message to spam or not spam")

input_message=st.text_area(label="Message",height='stretch',
placeholder="Enter your message here",key="input_message",on_change='ignore',
width="stretch",label_visibility="visible",disabled=False,help="Enter the message you want to classify as spam or not spam",max_chars=None,)

predict_button=st.button("Predict Message",type="secondary",  icon_position="left", width="content", wrap=True)

if predict_button:
    data={"message": input_message}
    if data["message"].strip() == "":
        st.warning("Please Enter your message")
    else:
        try:
            response=requests.post("http://127.0.0.1:8000/predict",json=data)

            if response.status_code==422:
                st.warning("Error")

            if response.status_code==200:
                result=response.json()
                massg_prediction=result["prediction"]
                st.success(massg_prediction)
        except :
            pass




sentiment_mapping = ["one", "two", "three", "four", "five"]
selected=st.feedback("stars")
if selected is not None:
    st.write(f"You selected {sentiment_mapping[selected]} stars")

    
