import streamlit as st
from PIL import Image
import soundfile as sf
import numpy as np
from transformers import ViTFeatureExtractor, ViTForImageClassification
import librosa
import tensorflow as tf
from io import BytesIO
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
import tempfile
import os
from labels import label_encoder
import requests
# Load ViT model for image classification
image_feature_extractor = ViTFeatureExtractor.from_pretrained('Dewa/dog_emotion_v2')
image_model = ViTForImageClassification.from_pretrained('Dewa/dog_emotion_v2')

# Load sound model
saved_model_path = "C:/Users/kuber vajpayee/models/newtry1.h5"
sound_model = tf.keras.models.load_model(saved_model_path)

# Function to preprocess a new image
def preprocess_image(file_content, target_size=(256, 256)):
    try:
        img = Image.open(BytesIO(file_content))
        img = img.convert("L")  # Convert image to grayscale
        img = img.resize(target_size)
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

# Function to generate mel spectrogram
def generate_mel_spectrogram(input_data, sample_rate):
    h_length = 512
    n_mels = 90
    n_fft = 2048
    mel_spectrogram = librosa.feature.melspectrogram(y=input_data, sr=sample_rate, n_fft=n_fft, hop_length=h_length, n_mels=n_mels)
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)
    return log_mel_spectrogram

# Function to process sound
def process_sound(sound_file):
    audio_data, sample_rate = sf.read(sound_file)

    # Generate mel spectrogram
    mel_spectrogram = generate_mel_spectrogram(audio_data, sample_rate)

    # Save mel spectrogram as a temporary image
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp:
        temp_image_path = temp.name
        plt.figure(figsize=(25, 10))
        librosa.display.specshow(mel_spectrogram, x_axis='time', y_axis='mel', sr=sample_rate)
        plt.colorbar(format="%+2.f")
        plt.title('Mel Spectrogram')
        plt.savefig(temp_image_path)
        plt.show()
        plt.close()

    # Load and preprocess the temporary image
    temp_image = load_img(temp_image_path, color_mode='grayscale', target_size=(256, 256))
    temp_image_array = img_to_array(temp_image)
    temp_image_array = np.expand_dims(temp_image_array, axis=0)

    # Make predictions using the loaded model
    predictions = sound_model.predict(temp_image_array)
    predicted_class = np.argmax(predictions)
    predicted_label = label_encoder.classes_[predicted_class]

    # Remove the temporary image file
    os.remove(temp_image_path)

    return predicted_label

# Streamlit layout
st.title("Image and Sound Input Form")
st.write("Please upload an image and a sound file.")

# Image upload
uploaded_image = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

# Sound upload
uploaded_sound = st.file_uploader("Upload Sound", type=["wav"])

# If both image and sound are uploaded
if uploaded_image is not None and uploaded_sound is not None:
    # Display uploaded image
    image = Image.open(uploaded_image)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Play uploaded sound
    with st.audio(uploaded_sound, format='audio/wav') as sound:
        st.write('Uploaded Sound')

    # Process the image using ViT model
    image_inputs = image_feature_extractor(images=image, return_tensors="pt")
    image_outputs = image_model(**image_inputs)
    image_logits = image_outputs.logits
    image_predicted_class_idx = image_logits.argmax(-1).item()
    image_predicted_label = image_model.config.id2label[image_predicted_class_idx]

    # Process the sound file and generate mel spectrogram
    audio_data, sample_rate = sf.read(uploaded_sound)
    mel_spectrogram = generate_mel_spectrogram(audio_data, sample_rate)
    
    # Save mel spectrogram as a temporary image
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp:
        temp_image_path = temp.name
        plt.figure(figsize=(25, 10))
        librosa.display.specshow(mel_spectrogram, x_axis='time', y_axis='mel', sr=sample_rate)
        plt.colorbar(format="%+2.f")
        plt.title('Mel Spectrogram')
        plt.savefig(temp_image_path)
        plt.close()

    # Display the generated mel spectrogram image
    mel_spectrogram_image = Image.open(temp_image_path)
    st.image(mel_spectrogram_image, caption='Mel Spectrogram', use_column_width=True)

    # Send the mel spectrogram image to the FastAPI server
    with open(temp_image_path, 'rb') as image_file:
        files = {'file': image_file}
        response = requests.post('http://127.0.0.1:8000/upload/', files=files) 
     
     # Print the response from the server
    if response.status_code == 200:
        # st.write("Server Response:", response.json())
        
        sound_predicted_label = response.json()
    else:
        st.write("Error:", response.text) 
    
    # Remove the temporary image file
    os.remove(temp_image_path)    
    # # Process the sound using the sound model
    # file_content = uploaded_sound.read()
    # new_img = preprocess_image(file_content)
    # if new_img is not None:
    #     sound_predictions = sound_model.predict(new_img)
    #     sound_predicted_class = np.argmax(sound_predictions)
    #     sound_predicted_label = label_encoder.classes_[sound_predicted_class]
    # else:
    #     sound_predicted_label = "Error processing sound file"
    

    # Print the output from both models
    st.write("Predicted class from Image Model:", image_predicted_label)
    st.write("Predicted class from Sound Model:", sound_predicted_label)
