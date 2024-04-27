import streamlit as st
import pandas as pd
import joblib

model = joblib.load("parkinson_predictor_model.joblib")



MDVP_Fo = st.number_input("MDVP:Fo(Hz)", value=148.79)
MDVP_Fhi = st.number_input("MDVP:Fhi(Hz)", value=175.829)
MDVP_Flo = st.number_input("MDVP:Flo(Hz)", value=104.31)
MDVP_Jitter_Percent = st.number_input("MDVP:Jitter(%)", value=0.004940)
MDVP_Jitter_Abs = st.number_input("MDVP:Jitter(Abs)", value=0.000030)
MDVP_RAP = st.number_input("MDVP:RAP", value=0.002500)
MDVP_PPQ = st.number_input("MDVP:PPQ", value=0.002690)
Jitter_DDP = st.number_input("Jitter:DDP", value=0.007490)
MDVP_Shimmer = st.number_input("MDVP:Shimmer", value=0.022970)
MDVP_Shimmer_dB = st.number_input("MDVP:Shimmer(dB)", value=0.221)
Shimmer_APQ3 = st.number_input("Shimmer:APQ3", value=0.012790)
Shimmer_APQ5 = st.number_input("Shimmer:APQ5", value=0.013470)
MDVP_APQ = st.number_input("MDVP:APQ", value=0.01826)
Shimmer_DDA = st.number_input("Shimmer:DDA", value=0.038360)
NHR = st.number_input("NHR", value=0.01166)
HNR = st.number_input("HNR", value=22.085)
RPDE = st.number_input("RPDE", value=0.49)
DFA = st.number_input("DFA", value=0.722254)
spread1 = st.number_input("spread1", value=-5.720868)
spread2 = st.number_input("spread2", value=0.218885)
D2 = st.number_input("D2", value=2.361532)
PPE = st.number_input("PPE", value=0.194052)



input_data = pd.DataFrame({
    'MDVP:Fo(Hz)': [MDVP_Fo],
    'MDVP:Fhi(Hz)': [MDVP_Fhi],
    'MDVP:Flo(Hz)': [MDVP_Flo],
    'MDVP:Jitter(%)': [MDVP_Jitter_Percent],
    'MDVP:Jitter(Abs)': [MDVP_Jitter_Abs],
    'MDVP:RAP': [MDVP_RAP],
    'MDVP:PPQ': [MDVP_PPQ],
    'Jitter:DDP': [Jitter_DDP],
    'MDVP:Shimmer': [MDVP_Shimmer],
    'MDVP:Shimmer(dB)': [MDVP_Shimmer_dB],
    'Shimmer:APQ3': [Shimmer_APQ3],
    'Shimmer:APQ5': [Shimmer_APQ5],
    'MDVP:APQ': [MDVP_APQ],
    'Shimmer:DDA': [Shimmer_DDA],
    'NHR': [NHR],
    'HNR': [HNR],
    'RPDE': [RPDE],
    'DFA': [DFA],
    'spread1': [spread1],
    'spread2': [spread2],
    'D2': [D2],
    'PPE': [PPE]
})


if st.button("Predict"):
    prediction = model.predict(input_data)
    if prediction == 1:
        st.write("The model predicts that the person has Parkinson's disease.")
    else:
        st.write("The model predicts that the person does not have Parkinson's disease.")
