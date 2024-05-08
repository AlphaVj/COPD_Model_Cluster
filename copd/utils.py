import os
import numpy as np
import pickle



# Function to preprocess input data and make predictions
def Copd_predict_output(input_data):
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Update paths to point to the correct location inside the project
    ann_model_path = os.path.join(BASE_DIR,'copd', 'models', 'COPD_ann_model.pickle')
    scaler_path = os.path.join(BASE_DIR,'copd','models', 'COPD_scaler.pickle')
    # Load the ANN model
    with open(ann_model_path, 'rb') as f:
        ann_model = pickle.load(f)
    
    # Load the scaler
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    # Preprocess input data (scaling)
    input_data_scaled = scaler.transform([input_data])
    
    # Predict using the preprocessed input data
    prediction = ann_model.predict(input_data_scaled)
    predicted_class = np.argmax(prediction)
    
    # Define labels
    labels = ['Mild', 'Moderate', 'Severe', 'Very Severe']
    
    # Get the label for the predicted class
    predicted_label = labels[predicted_class]
    
    # Return the predicted label
    return predicted_label

# chatbot_api/openai.py


# Set your OpenAI API key




