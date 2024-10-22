import pickle
import numpy as np
import streamlit as st

# Load the model and scaler
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Title of the web app
st.title('Network Intrusion Detection System')

# Add inputs for all features used in the model
rate = st.text_input('Rate', '')
sttl = st.text_input('STTL', '')
dload = st.text_input('Dload', '')
swin = st.text_input('Swin', '')
dwin = st.text_input('Dwin', '')
dmean = st.text_input('Dmean', '')
ct_state_ttl = st.text_input('CT_State_TTL', '')
ct_src_dport_ltm = st.text_input('CT_Src_Dport_LTM', '')
ct_dst_sport_ltm = st.text_input('CT_Dst_Sport_LTM', '')
ct_dst_src_ltm = st.text_input('CT_Dst_Src_LTM', '')

# Button to make predictions
if st.button('Predict'):
    try:
        # Collect input from the form
        features = [float(rate), float(sttl), float(dload), float(swin), float(dwin),
                    float(dmean), float(ct_state_ttl), float(ct_src_dport_ltm),
                    float(ct_dst_sport_ltm), float(ct_dst_src_ltm)]
        
        features_array = np.array(features).reshape(1, -1)

        # Scale the input data using the loaded scaler
        scaled_features = scaler.transform(features_array)

        # Make prediction using the trained model
        prediction = model.predict(scaled_features)

        # Interpret the result (adjust as needed based on label encoding)
        if prediction[0] == 0:
            result = "Normal"
        else:
            result = "Miscellaneous"

        # Display the result
        st.success(f'Prediction: {result}')
    
    except Exception as e:
        st.error(f'Error: {str(e)}')
