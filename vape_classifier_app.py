import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from transformers import ViTForImageClassification

# Define the CustomEnsemble class using ViT model
class CustomEnsemble(nn.Module):
    def __init__(self):
        super(CustomEnsemble, self).__init__()

        # Load ViT model with pretrained weights from JFT-300M
        self.model_vit = ViTForImageClassification.from_pretrained(
            'google/vit-large-patch16-224',
            num_labels=1,  # Set number of labels for binary classification
            ignore_mismatched_sizes=True
        )

        # Add Dropout layer with a probability of 0.3 (as in training)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        # Get predictions from ViT model
        outputs = self.model_vit(x).logits
        outputs = self.dropout(outputs)
        return outputs

# Define the same transformations used during training
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to match model's input size
    transforms.ToTensor(),           # Convert image to tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize for ViT
])

# Load the trained model
@st.cache(allow_output_mutation=True)
def load_model():
    model = CustomEnsemble()
    model.load_state_dict(torch.load('custom_ensemble_model.pth', map_location=torch.device('cpu')))
    model.eval()  # Set the model to evaluation mode
    return model

# Load the model
model = load_model()

# Define the Streamlit interface
st.title("Vape Image Classifier")
st.write("Upload an image and the model will classify it as either **Vape** or **Not Vape**.")

# Allow user to upload an image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Load the image and display it
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Preprocess the image
    image = image.convert('RGB')  # Ensure it's in RGB mode
    image = transform(image).unsqueeze(0)  # Apply transformations and add batch dimension

    # Perform the prediction
    with torch.no_grad():
        output = model(image)
        prediction = torch.sigmoid(output).item()

    # Display the result
    threshold = 0.5  # Adjust this if needed
    if prediction > threshold:
        st.write("The image is classified as: **Vape**")
    else:
        st.write("The image is classified as: **Not Vape**")