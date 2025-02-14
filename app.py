import numpy as np
import pickle as pkl
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPool2D

from sklearn.neighbors import NearestNeighbors
import os
from numpy.linalg import norm
import streamlit as st
import pandas as pd  # For handling feedback storage

st.header('Fashion Recommendation System with Evaluation')

# Load precomputed image features and filenames
Image_features = pkl.load(open('Images_features.pkl', 'rb'))
filenames = pkl.load(open('filenames.pkl', 'rb'))

def extract_features_from_images(image_path, model):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_expand_dim = np.expand_dims(img_array, axis=0)
    img_preprocess = preprocess_input(img_expand_dim)
    result = model.predict(img_preprocess).flatten()
    norm_result = result / norm(result)
    return norm_result

# Load the pre-trained model
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = tf.keras.models.Sequential([
    model,
    GlobalMaxPool2D()
])

# Initialize the Nearest Neighbors model
neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
neighbors.fit(Image_features)

# Upload image
upload_file = st.file_uploader("Upload an image to find similar products")

if upload_file is not None:
    # Save uploaded image
    with open(os.path.join('upload', upload_file.name), 'wb') as f:
        f.write(upload_file.getbuffer())
    
    # Display uploaded image
    st.subheader('Uploaded Image')
    st.image(upload_file)

    # Extract features and find similar images
    input_img_features = extract_features_from_images(upload_file, model)
    distance, indices = neighbors.kneighbors([input_img_features])

    st.subheader('Recommended Images')
    
    # Create columns for displaying images and collecting feedback
    cols = st.columns(5)
    feedback = {}  # Dictionary to store user feedback

    for i, col in enumerate(cols):
        if i+1 < len(indices[0]):
            # Get the image filename
            image_path = filenames[indices[0][i+1]]
            
            # Display the image
            with col:
                st.image(image_path, use_container_width=True)  # Updated parameter
                
                # Add a select box for rating
                rating = st.selectbox(
                    f'Rate this recommendation (Image {i+1})',
                    options=[1, 2, 3, 4, 5],
                    index=4,  # Default rating is 5
                    key=f'rating_{i}'
                )
                
                # Add an optional text input for comments
                comment = st.text_input(
                    f'Comments for Image {i+1}',
                    key=f'comment_{i}'
                )
                
                # Store the feedback
                feedback[image_path] = {
                    'rating': rating,
                    'comment': comment
                }

    # Add an overall satisfaction rating
    st.subheader('Overall Satisfaction')
    overall_satisfaction = st.slider(
        'Rate your overall satisfaction with these recommendations:',
        min_value=1,
        max_value=5,
        value=5  # Default value
    )

    # Submit button to save feedback
    if st.button('Submit Feedback'):
        # Save feedback to a CSV file
        feedback_df = pd.DataFrame.from_dict(feedback, orient='index')
        feedback_df.reset_index(inplace=True)
        feedback_df.rename(columns={'index': 'image_path'}, inplace=True)
        
        # Add the overall satisfaction rating to the DataFrame
        overall_feedback = pd.DataFrame([{
            'image_path': 'Overall',
            'rating': overall_satisfaction,
            'comment': 'Overall satisfaction rating'
        }])
        feedback_df = pd.concat([feedback_df, overall_feedback], ignore_index=True)
        
        # Append feedback to the CSV file
        feedback_file = 'feedback.csv'
        if os.path.exists(feedback_file):
            feedback_df.to_csv(feedback_file, mode='a', header=False, index=False)
        else:
            feedback_df.to_csv(feedback_file, index=False)
        
        st.success('Thank you for your feedback!')
