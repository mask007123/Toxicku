import numpy as np
import pandas as pd
import pickle
import streamlit as st
from PIL import Image

with open('toxic_vect.pkl', 'rb') as f:
    toxic = pickle.load(f)
with open('toxic_model.pkl', 'rb') as f:
    toxic_model = pickle.load(f)
with open('severe_toxic_vect.pkl', 'rb') as f:
    severe_toxic = pickle.load(f)
with open('severe_toxic_model.pkl', 'rb') as f:
    severe_toxic_model = pickle.load(f)
with open('threat_vect.pkl', 'rb') as f:
    threat = pickle.load(f)
with open('threat_model.pkl', 'rb') as f:
    threat_model = pickle.load(f)
with open('obscene_vect.pkl', 'rb') as f:
    obscene = pickle.load(f)
with open('obscene_model.pkl', 'rb') as f:
    obscene_model = pickle.load(f)
with open('insult_vect.pkl', 'rb') as f:
    insult = pickle.load(f)
with open('insult_model.pkl', 'rb') as f:
    insult_model = pickle.load(f)
with open('identity_hate_vect.pkl', 'rb') as f:
    identity_hate = pickle.load(f)
with open('identity_hate_model.pkl', 'rb') as f:
    identity_hate_model = pickle.load(f)

def main():
    st.title("Toxic Comment Classifier Using NLP")
    image = Image.open('Image.jpg')
    st.image(image, use_column_width=True)

    Input = st.text_input("Don't Hold Back - Be as rude as you can [in english only]")
    if len(Input) == 0:
        return
    # toxic
    vect = toxic.transform([Input])
    zero = toxic_model.predict_proba(vect)[:, 0][0]
    one = toxic_model.predict_proba(vect)[:, 1][0]
    if (zero >= 0.41 and one <= 0.59) and (zero <= 0.59 and one >= 0.41):
        st.write('neutral')
    elif one > 0.59:
        st.write('Toxic')
    else:
        st.write('Non Toxic')

    #severe_toxic
    vect = severe_toxic.transform([Input])
    zero = severe_toxic_model.predict_proba(vect)[:, 0][0]
    one = severe_toxic_model.predict_proba(vect)[:, 1][0]
    if (zero >= 0.41 and one <= 0.59) and (zero <= 0.59 and one >= 0.41):
        st.write('neutral')
    elif one > 0.59:
        st.write('Severe_toxic')
    else:
        st.write('Non Severe_toxic')

    #threat
    vect = threat.transform([Input])
    zero = threat_model.predict_proba(vect)[:, 0][0]
    one = threat_model.predict_proba(vect)[:, 1][0]
    if (zero >= 0.41 and one <= 0.59) and (zero <= 0.59 and one >= 0.41):
        st.write('neutral')
    elif one > 0.59:
        st.write('Threat')
    else:
        st.write('Non Threat')

    #obscene
    vect = obscene.transform([Input])
    zero = obscene_model.predict_proba(vect)[:, 0][0]
    one = obscene_model.predict_proba(vect)[:, 1][0]
    if (zero >= 0.41 and one <= 0.59) and (zero <= 0.59 and one >= 0.41):
        st.write('neutral')
    elif one > 0.59:
        st.write('obscene')
    else:
        st.write('Non obscene')

    #insult
    vect = insult.transform([Input])
    zero = insult_model.predict_proba(vect)[:, 0][0]
    one = insult_model.predict_proba(vect)[:, 1][0]
    if (zero >= 0.41 and one <= 0.59) and (zero <= 0.59 and one >= 0.41):
        st.write('neutral')
    elif one > 0.59:
        st.write('Insult')
    else:
        st.write('Non Insult')

    #identity_hate
    vect = identity_hate.transform([Input])
    zero = identity_hate_model.predict_proba(vect)[:, 0][0]
    one = identity_hate_model.predict_proba(vect)[:, 1][0]
    if (zero >= 0.41 and one <= 0.59) and (zero <= 0.59 and one >= 0.41):
        st.write('neutral')
    elif one > 0.59:
        st.write('Identity hate')
    else:
        st.write('Non Identity hate')


if __name__ == '__main__':
    main()
