
import numpy as np
import streamlit as st
import pickle
import pandas as pd

# Load the trained model
model = pickle.load(open('LRmodel1.pkl', 'rb'))

holderArray = np.zeros(18)#array for hoding binary inputs

# Reshape input features array outside the prediction function
def preprocess_features(features):
    
    
   # Convert categorical input to numerical value efficiently
   
   df = pd.DataFrame(np.array(features).reshape(1, -1))
   #holderArray = np.zeros(18)
    
   return pd.get_dummies(df)

# Predict diabetes based on features
def predict_diabetes(features):
    return model.predict(features)

# Create a Streamlit web app
def main():
    st.title("Drug Recommendation System")
    st.image('diabetic_pic.jpg',use_column_width='right')
    st.write("Enter the required information to get a recommendation for your BP Drug ")

    # Input fields for user information
    input_features = {
    
        
        "Age": st.number_input("Enter your age", min_value=1, max_value=100, value=30),
        "Sex": st.selectbox("Sex" , ("F", "M")),
        "BP":  st.selectbox("Blood Pressure", ("High", "Low", "Normal")),
        "Cholesterol": st.selectbox("Cholesterol Levels", ("HIGH", "NORMAL")),
        "Na_to_K": st.number_input("Enter your Sodium to Pottasium", min_value=1, max_value=40, value=8 )
    
        
    }
    #Sex_F	Sex_M	BP_HIGH	BP_LOW	BP_NORMAL	Cholesterol_HIGH	Cholesterol_NORMAL	Age_binned_<20s	Age_binned_20s	Age_binned_30s	
    #Age_binned_40s	Age_binned_50s	Age_binned_60s	Age_binned_>60s	Na_to_K_binned_<10	Na_to_K_binned_10-20	
    #Na_to_K_binned_20-30	Na_to_K_binned_>30

    #Convert categorical input to numerical value efficiently
    holderArray[0] = 1 if input_features["Sex"] == "F" else 0
    holderArray[1] = 1 if input_features["Sex"] == "M" else 0
    holderArray[2] = 1 if input_features["BP"] == "High" else 0
    holderArray[3] = 1 if input_features["BP"] == "Low" else 0
    holderArray[4] = 1 if input_features["BP"] == "Normal" else 0
    holderArray[5] = 1 if input_features["Cholesterol"] == "HIGH" else 0
    holderArray[6] = 1 if input_features["Cholesterol"] == "NORMAL" else 0
    holderArray[7] = 1 if input_features["Age"] < 20 else 0
    holderArray[8] = 1 if ((input_features["Age"] == 20) and (input_features["Age"] <30)) else 0
    holderArray[9] = 1 if ((input_features["Age"] == 30) and (input_features["Age"] <40)) else 0
    holderArray[10] = 1 if ((input_features["Age"] == 40) and (input_features["Age"] <50)) else 0
    holderArray[11] = 1 if ((input_features["Age"] == 50) and (input_features["Age"] <60)) else 0
    holderArray[12] = 1 if ((input_features["Age"] == 60) and (input_features["Age"] <70)) else 0
    holderArray[13] = 1 if input_features["Age"] > 70  else 0
    holderArray[14] = 1 if input_features["Na_to_K"] < 10 else 0
    holderArray[15] = 1 if ((input_features["Na_to_K"] == 10) and (input_features["Na_to_K"] <20)) else 0
    holderArray[16] = 1 if ((input_features["Na_to_K"] == 20) and (input_features["Na_to_K"] <30)) else 0
    holderArray[17] = 1 if input_features["Na_to_K"] > 30 else 0
    

    if st.button("PREDICT"):
        # Gather input features and preprocess
        features = preprocess_features(list(input_features.values()))

        # Predict diabetes
        prediction = predict_diabetes(holderArray.reshape(1,-1))

        # Display the prediction
        if prediction == 0:
            st.success('Congratulations! You have a low diabetes risk')
            st.write(
                "Based on the provided information, it seems there is no immediate recommendation for the stated case, it is recommended that you counsult  a specialist or re-enter some accurate data.")
        else:
            st.success(prediction[0])
            st.write(
                "Based on the provided information, the recommended drug for this particular case is stated above, It is strongly recommended to consult with a healthcare professional for a thorough evaluation and guidance on preventive measures.")

if __name__ == '__main__':
    main()
