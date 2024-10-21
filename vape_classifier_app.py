import os
import streamlit as st
import torch
import torch.nn as nn
import requests  # Import requests for downloading files
from transformers import ViTForImageClassification

# Define the CustomEnsemble class using ViT model
class CustomEnsemble(nn.Module):
    def __init__(self):
        super(CustomEnsemble, self).__init__()
        self.model_vit = ViTForImageClassification.from_pretrained(
            'google/vit-large-patch16-224',
            num_labels=1,
            ignore_mismatched_sizes=True
        )
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        outputs = self.model_vit(x).logits
        outputs = self.dropout(outputs)
        return outputs

# Function to download the model from Google Drive
def download_file_from_google_drive(file_id, destination):
    URL = f'https://drive.google.com/uc?id={file_id}'
    response = requests.get(URL)
    
    if response.status_code == 200:
        with open(destination, 'wb') as f:
            f.write(response.content)
        st.success("Model downloaded successfully.")
    else:
        st.error("Error downloading the file. Please check the file ID or permissions.")

# Load the trained model
@st.cache_data
def load_model():
    model = CustomEnsemble()
    
    # Check if the model file already exists
    if not os.path.exists('custom_ensemble_model.pth'):
        st.info("Model file not found. Downloading...")
        download_file_from_google_drive('1aeY1XyB1SgVhP4iHKUHBRVHRtPDK1TFe', 'custom_ensemble_model.pth')
        
        # Check again if the model was downloaded
        if not os.path.exists('custom_ensemble_model.pth'):
            st.error("Failed to download the model. Please check the file ID and permissions.")
            return None

    try:
        # Load the model state dictionary
        model.load_state_dict(torch.load('custom_ensemble_model.pth', map_location=torch.device('cpu')))
        model.eval()  # Set the model to evaluation mode
        st.success("Model loaded successfully.")
    except FileNotFoundError:
        st.error("Model file not found after download. Please check your download logic.")
    except Exception as e:
        st.error(f"An error occurred while loading the model: {str(e)}")

    return model

# Load the model
model = load_model()

# Streamlit interface code continues...
