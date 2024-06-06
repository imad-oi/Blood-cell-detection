import streamlit as st
import cv2
import numpy as np
import pickle
from PIL import Image
from collections import Counter
from features import process_images, prepare_for_prediction, scale_data
import numpy as np
from scipy import ndimage as ndi
from skimage import morphology
from sklearn.cluster import KMeans
import os
import random


# Charger le modèle pré-entraîné
with open('../models/svm_model.pkl', 'rb') as f:
    model = pickle.load(f)

st.title('Détection Automatisée des Cellules Cancéreuses dans le Sang')

uploaded_file = st.file_uploader("Choisissez une image...", type="jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image = np.array(image)
    i= cv2.resize(image,(224,224))
    #-------- Segmentation --------- 
    # Convert the image from BGR to RGB
    image_rgb = cv2.cvtColor(i, cv2.COLOR_BGR2RGB)
    # Convert the image from RGB to LAB
    image_lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    # Split the LAB image into L, A and B channels
    l_channel, a_channel, b_channel = cv2.split(image_lab)
    # Reshape the A channel
    reshaped_a = a_channel.reshape(a_channel.shape[0]*a_channel.shape[1],1)
    # Apply KMeans clustering to the reshaped A channel
    kmeans = KMeans(n_clusters=7, random_state=0).fit(reshaped_a)
    # Replace each pixel with its cluster center
    pixels_replaced_with_centers = kmeans.cluster_centers_[kmeans.labels_]
    # Reshape the image back to its original shape
    clustered_image = pixels_replaced_with_centers.reshape(a_channel.shape[0],a_channel.shape[1])
    clustered_image = clustered_image.astype(np.uint8)
    # Apply thresholding to the clustered image
    ret, thresholded_image = cv2.threshold(clustered_image,141,255 ,cv2.THRESH_BINARY)
    # Fill holes in the thresholded image
    filled_image = ndi.binary_fill_holes(thresholded_image)
    # Remove small objects from the filled image
    cleaned_image_1 = morphology.remove_small_objects(filled_image, 200)
    # Remove small holes from the cleaned image
    cleaned_image_2 = morphology.remove_small_holes(cleaned_image_1,250)
    cleaned_image_2 = cleaned_image_2.astype(np.uint8)
    # Apply the cleaned image as a mask to the original image
    segmented_image = cv2.bitwise_and(image_rgb, image_rgb, mask=cleaned_image_2)
    # Convert the segmented image from RGB to BGR for display
    segmented_image_bgr = cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR)
    # Display the original image
    st.image(image, caption='Image téléchargée.',width=224)
    st.write("")
    # Display the 'a' channel
    st.image(a_channel, caption='Canal a')
    # Display the clustered image
    st.image(clustered_image, caption='Clustering')
    # Display the thresholded image
    st.image(thresholded_image, caption='Seuillage')
    # Display the final segmented image
    st.image(segmented_image_bgr, caption='Image segmentée')
    # Prétraitement de l'image pour le modèle
    image_features = []
    process_images([segmented_image_bgr],["image"],image_features,True)
    image_features_df = prepare_for_prediction(image_features)
    print(image_features_df)
    num_cells = image_features_df['avg_cells'].shape[0]
    images_scaled_features_df = scale_data(image_features_df)

    # Prédiction
    predictions = model.predict(images_scaled_features_df)
    print(predictions)
    counter = Counter(predictions)

    # Retourne l'élément le plus fréquent
    most_common_element = counter.most_common(1)[0][0]

    st.markdown(f'<div style="color: white; background-color: blue; padding:10px; border-radius:5px;">Nombre de cellules segmentées est: {num_cells}</div>', unsafe_allow_html=True)
    st.write("")
    st.markdown(f'<div style="color: white; background-color: green; padding:10px; border-radius:5px;">La classe prédite est: {most_common_element}</div>', unsafe_allow_html=True)