# importing necessary libraries
from fastapi import FastAPI, File, UploadFile
from PIL import Image
from io import BytesIO
import numpy as np
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os 
import tensorflow as tf
from typing import Union

# creating app for fast api
app = FastAPI()

# loading model 
saved_model_path = "C:/Users/kuber vajpayee/models/newtry1.h5"
model = tf.keras.models.load_model(saved_model_path)


# for loading labels 
def load_data(data_folder, height, width):
    spectrograms = []
    labels = []
    
    for label in os.listdir(data_folder):
        label_folder = os.path.join(data_folder, label)
        for file in os.listdir(label_folder):
            if file.endswith(".png"):  # Assuming all spectrogram images are in PNG format
                file_path = os.path.join(label_folder, file)
                
                # Load the spectrogram image
                spectrogram = load_img(file_path, color_mode='grayscale', target_size=(height, width))
                spectrogram = img_to_array(spectrogram)
                
                spectrograms.append(spectrogram)
                labels.append(label)

    return np.array(spectrograms), np.array(labels)

# loading data 
data_folder = "C:/Users/kuber vajpayee/Downloads/dataset"
height = 256
width = 256
spectrograms, labels = load_data(data_folder, height, width)

# putting label encoders 
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)


# i have put the new preprocess_image function for now 
# Function to preprocess a new image
def preprocess_image(file_content, target_size=(256, 256)):
    # add the BytesIO to this 
    img = load_img(BytesIO(file_content), color_mode='grayscale', target_size=target_size)
    # img = load_img(image_path, color_mode='grayscale', target_size=target_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.get("/")
def read_root():
    return {"Hello": "World"}

#  creating a post method for prediction 
@app.post("/upload/")
async def create_file_upload(file: UploadFile = File(...)):
    try:
        #  Read the file content 
        file_content = await file.read()
        
        # Preprocess the uploaded image
        new_img = preprocess_image(file_content)
        
        # Make predictions using the loaded model
        predictions = model.predict(new_img)
        
        # Decode the predictions to get class labels
        # Modify this part based on your model's output
        predicted_class = np.argmax(predictions)
        
        # Convert numpy.int64 to regular Python int
        # predicted_class = int(predicted_class)
        predicted_class_label = label_encoder.classes_[predicted_class]
        
        return {"predicted_class": predicted_class_label}
    except Exception as e:
        # Log any errors that occur
        print(f"Error uploading file: {e}")
        return {"error": "An error occurred"}
    