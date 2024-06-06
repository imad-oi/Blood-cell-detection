import streamlit as st
import cv2
import numpy as np
import pickle
from PIL import Image
from collections import Counter
# from features import process_images, prepare_for_prediction, scale_data
from features import process_images
import numpy as np
from scipy import ndimage as ndi
from skimage import morphology
from sklearn.cluster import KMeans
import os
import random
import matplotlib.pyplot as plt
import pandas as pd
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

# Charger le modèle pré-entraîné
with open('../models/svm_model2.pkl', 'rb') as f:
    model_svm = pickle.load(f)
with open('../models/dt_model2.pkl', 'rb') as f:
    model_dt = pickle.load(f)
with open('../models/rf_model2.pkl', 'rb') as f:
    model_rf = pickle.load(f)
with open('../models/knn_model2.pkl', 'rb') as f:
    model_knn = pickle.load(f)

# Set page layout to wide
st.set_page_config(layout="wide", page_title="Cellules Cancéreuses", page_icon=":microscope:")

logo = f"""<img src="https://images.unsplash.com/photo-1557683304-673a23048d34?w=500&auto=format&fit=crop&q=60&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8MTJ8fGJhY2tncm91bmR8ZW58MHx8MHx8fDA%3D" 
style="float: left; margin: 10px; height: 200px;width:100%;">"""

max_width = 800  # Set the maximum width to 800 pixels
padding_top = 1  # Set the top padding to 1 rem
padding_right = 1  # Set the right padding to 1 rem
padding_left = 1  # Set the left padding to 1 rem
padding_bottom = 1  # Set the bottom padding to 1 rem
primaryColor = "#f63366"  # Set the primary color to pink

st.markdown(
    f"""
    <style>
        .reportview-container .main .block-container{{
            max-width: {max_width}px;
            padding-top: {padding_top}rem;
            padding-right: {padding_right}rem;
            padding-left: {padding_left}rem;
            padding-bottom: {padding_bottom}rem;
        }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div>
    {logo}
    <h1 style="color: {primaryColor};text-align:center;">
    {'Détection Automatisée des Cellules Cancéreuses dans le Sang'} 
    </h1>
    </div>
    """,
    unsafe_allow_html=True,
)
# st.title('Détection Automatisée des Cellules Cancéreuses dans le Sang')     
uploaded_files = st.file_uploader("Choisissez des images...", type=["jpg","png"], accept_multiple_files=True)
if len(uploaded_files):
    cols = st.columns(len(uploaded_files))
else:
    st.markdown(f'<div style="color: white; background-color: {primaryColor}; padding:10px; border-radius:5px;">Aucune image télechargée</div>', unsafe_allow_html=True)
    st.write("")

images_path = []

for index, uploaded_file in enumerate(uploaded_files):
    img = Image.open(uploaded_file)
    img = img.resize((224, 224))  
    img.save(uploaded_file.name, 'JPEG')
    images_path.append(uploaded_file.name)
    
    # Display the uploaded image in a column with the image name in the caption
    cols[index].image(img, caption=f'Uploaded Image: {uploaded_file.name}', use_column_width=True)
    
    # # Image processing
    image = np.array(img)
    i = cv2.resize(image,(224,224))
    image_rgb = cv2.cvtColor(i, cv2.COLOR_BGR2RGB)
    image_lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(image_lab)
    reshaped_a = a_channel.reshape(a_channel.shape[0]*a_channel.shape[1],1)
    kmeans = KMeans(n_clusters=7, random_state=0).fit(reshaped_a)
    pixels_replaced_with_centers = kmeans.cluster_centers_[kmeans.labels_]
    clustered_image = pixels_replaced_with_centers.reshape(a_channel.shape[0],a_channel.shape[1])
    clustered_image = clustered_image.astype(np.uint8)
    ret, thresholded_image = cv2.threshold(clustered_image,141,255 ,cv2.THRESH_BINARY)
    filled_image = ndi.binary_fill_holes(thresholded_image)
    cleaned_image_1 = morphology.remove_small_objects(filled_image, 200)
    cleaned_image_2 = morphology.remove_small_holes(cleaned_image_1,250)
    cleaned_image_2 = cleaned_image_2.astype(np.uint8)
    segmented_image = cv2.bitwise_and(image_rgb, image_rgb, mask=cleaned_image_2)
    segmented_image_bgr = cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR)
    
     # Watershed segmentation
    distance = ndi.distance_transform_edt(cleaned_image_2)
    coords = peak_local_max(distance, min_distance=10, labels=cleaned_image_2)
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndi.label(mask)
    labels = watershed(-distance, markers, mask=cleaned_image_2)
    # Convert labels to RGB
    from skimage.color import label2rgb
    labels_rgb = label2rgb(labels)
    # 
    
    # # Display the processed images
    cols[index].image(a_channel, caption=f'Canal a : {uploaded_file.name}', use_column_width=True)
    # cols[index].image(clustered_image, caption=f'Clustering : {uploaded_file.name}', use_column_width=True)
    cols[index].image(thresholded_image, caption=f'Seuillage : {uploaded_file.name}', use_column_width=True)
    cols[index].image(segmented_image_bgr, caption=f'Image segmentée : {uploaded_file.name}', use_column_width=True)
    cols[index].image(labels_rgb, caption=f'Watershed Segmentation : {uploaded_file.name}', use_column_width=True)
    
labeled_features_testing = []
process_images([cv2.imread(img) for img in images_path] ,images_path , labeled_features_testing,False)

df_labeled_features_testing = pd.DataFrame(labeled_features_testing)

if uploaded_files:
    df_labeled_features_testing['R'] = df_labeled_features_testing['color'].apply(lambda x: x[0])
    df_labeled_features_testing['G'] = df_labeled_features_testing['color'].apply(lambda x: x[1])
    df_labeled_features_testing['B'] = df_labeled_features_testing['color'].apply(lambda x: x[2])
    df_labeled_features_testing = df_labeled_features_testing.drop('color', axis=1)

    df_labeled_features_testing['centroid_x'] = df_labeled_features_testing['centroid'].apply(lambda x: x[1])
    df_labeled_features_testing['centroid_y'] = df_labeled_features_testing['centroid'].apply(lambda x: x[0])
    df_labeled_features_testing = df_labeled_features_testing.drop('centroid', axis=1)

    labeled_features_testing_df = df_labeled_features_testing.groupby('image_index').agg({
    'solidity': ['mean', 'std'],
    'eccentricity': ['mean', 'std'],
    'orientation': ['mean', 'std'],
    'equivalent_diameter': ['mean', 'std'],
    'contrast': ['mean', 'std'],
    'mean_intensity': ['mean', 'std'],
    'median_intensity': ['mean', 'std'],
    'std_intensity': ['mean', 'std'],
    'correlation': ['mean', 'std'],
    'energy': ['mean', 'std'],
    'convex_area': ['mean', 'std'],
    'bbox_area': ['mean', 'std'],
    'extent': ['mean', 'std'],
    'circulatie': ['mean', 'std'],
    'cells_in_image': 'first',
    'area': ['mean', 'std'],
    'perimeter':  ['mean', 'std'],
    'aspect_ratio':  ['mean', 'std'],
    'R':  ['mean', 'std'],
    'G':  ['mean', 'std'],
    'B':  ['mean', 'std'],
    'centroid_x':  ['mean', 'std'],
    'centroid_y':  ['mean', 'std'],
    'label': 'first',
    })

    labeled_features_testing_df =  labeled_features_testing_df.drop(columns=['label'])
    print("============================================")
    print("labeled_features_testing_df\n")
    print("============================================")
    print(labeled_features_testing_df)
    print("============================================")

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()

    column_names = labeled_features_testing_df.columns
    scaler.fit(labeled_features_testing_df)

    scaled_features = scaler.transform(labeled_features_testing_df)
    print(column_names)

    print("============ svm ===================")
    svm_predictions = model_svm.predict(scaled_features)
    print(svm_predictions)
    print("============ RF ===================")
    rf_predictions = model_rf.predict(scaled_features)
    print(rf_predictions)
    print("============ DT ===================")
    dt_predictions = model_dt.predict(scaled_features)
    print(dt_predictions)
    print("============ KNN ===================")
    knn_predictions = model_knn.predict(scaled_features)
    print(knn_predictions)

    # Afficher les prédictions pour chaque image
    for index, uploaded_file in enumerate(uploaded_files):
        cols[index].write(f'Classe prédite SVM: {svm_predictions[index]}')
        cols[index].write(f'Classe prédite RF: {rf_predictions[index]}')
        cols[index].write(f'Classe prédite DT: {dt_predictions[index]}')
        cols[index].write(f'Classe prédite KNN: {knn_predictions[index]}')

    