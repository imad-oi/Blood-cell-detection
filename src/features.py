import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import cv2
from scipy import ndimage as ndi
from skimage import morphology
from skimage import io, color, filters, segmentation
from skimage.feature import graycomatrix, graycoprops, peak_local_max
from skimage.color import rgb2gray
from skimage.util import img_as_ubyte
from skimage.measure import label as sk_label 
from skimage.measure import regionprops
from skimage import measure
from skimage.segmentation import watershed
from imutils import paths
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.metrics import mean_absolute_error,mean_squared_error
import time
import os
import random
import warnings
warnings.filterwarnings('ignore')


# =========================================
# function pour extraire les caracteristiques utilisant k-means

def kmeans_extract_cell_properties(image_rgb, k=7, threshold_value=142):
    image_lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    l,a,b = cv2.split(image_lab)                         
    
    a_reshaped = a.reshape(-1, 1)
    kmeans = KMeans(n_clusters=k, random_state=0).fit(a_reshaped)
    clustered_a = kmeans.cluster_centers_[kmeans.labels_].reshape(a.shape[0], a.shape[1])
    
    ic = clustered_a.astype(np.uint8)
    _, thresholded_a = cv2.threshold(ic, threshold_value, 255, cv2.THRESH_BINARY)
    
    filled_image = ndi.binary_fill_holes(thresholded_a)
    
    cleaned_image = morphology.remove_small_objects(filled_image, min_size=200)
    cleaned_image = morphology.remove_small_holes(cleaned_image, area_threshold=250)
    cleaned_image = cleaned_image.astype(np.uint8)
    
    distance = ndi.distance_transform_edt(cleaned_image)
    coords = peak_local_max(distance, min_distance=10, labels=cleaned_image)
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndi.label(mask)
    labels = watershed(-distance, markers, mask=cleaned_image)
    
    regions = measure.regionprops(labels)
    return regions, labels

# =========================================
# function pour extraire les caracteristiques des régions

def extract_features(region,image_rgb):
    
    solidity = region.solidity
    eccentricity = region.eccentricity
    orientation = region.orientation
    equivalent_diameter = region.equivalent_diameter
    
    minr, minc, maxr, maxc = region.bbox
    cell = image_rgb[minr:maxr, minc:maxc]
    color = np.mean(cell, axis=(0, 1))
    
    cell_gray = rgb2gray(cell)
    cell_gray_ubyte = img_as_ubyte(cell_gray)
    glcm = graycomatrix(cell_gray_ubyte, [5], [0], 256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    
    mean_intensity = np.mean(cell)
    median_intensity = np.median(cell)
    std_intensity = np.std(cell)
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    
    centroid = region.centroid
    convex_area = region.convex_area
    bbox_area = region.bbox_area
    bbox = region.bbox
    extent = region.extent
    circulatie = region.perimeter ** 2 / (4 * np.pi * region.area)
    
    return {
        "solidity": solidity,
        "eccentricity": eccentricity,
        "orientation": orientation,
        "equivalent_diameter": equivalent_diameter,
        "color": color,
        "contrast": contrast,
        "mean_intensity": mean_intensity,
        "median_intensity": median_intensity,
        "std_intensity": std_intensity,
        "correlation": correlation,
        "energy": energy,
        "centroid": centroid,
        "convex_area": convex_area,
        "bbox_area": bbox_area,
        "extent": extent,
        "circulatie": circulatie,
    }
    
# =========================================
# function pour faire la segmentation et l'extraction

def process_images(rgb_images, images_label, labeled_features,display_images=False):
    for image_rgb, image_label in zip(rgb_images, images_label):
        regions,output_image= kmeans_extract_cell_properties(image_rgb)
        count_cells_in_image = len(regions)
        
        features = []
        
        for region in regions:
            region_features = extract_features(region, image_rgb)
            region_features['cells_in_image'] = count_cells_in_image
            features.append(region_features)
        
        if display_images:
            plt.figure(figsize=(12, 12))
            plt.imshow(output_image)
            plt.title(image_label)
            plt.axis('off')
        
        for region, region_features in zip(regions, features):
            minr, minc, maxr, maxc = region.bbox
            rect = Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor='red', linewidth=2)
            plt.gca().add_patch(rect)
            floored_color = tuple(map(int, region_features["color"]))
            # plt.text(minc, minr - 10, f'Area: {region.area:.2f}\nPerimeter: {region.perimeter:.2f}\nAspect Ratio: {region.major_axis_length / region.minor_axis_length:.2f}\nColor: {floored_color}\nTexture: {region_features["contrast"]:.2f}', color='red')
            region_features["area"] = region.area
            region_features["perimeter"] = region.perimeter
            region_features["aspect_ratio"] = region.major_axis_length / (region.minor_axis_length + 1e-10)
            region_features["label"] = image_label
            labeled_features.append(region_features)
            
        if display_images:
            plt.axis('off')
            plt.show()

# ========================================
# fonction pour preparer les colonnes à l'entrainement et la prédiction

def prepare_for_prediction(features):
    features_df = pd.DataFrame(features)
    features_df['R'] = features_df['color'].apply(lambda x: x[0])
    features_df['G'] = features_df['color'].apply(lambda x: x[1])
    features_df['B'] = features_df['color'].apply(lambda x: x[2])
    features_df = features_df.drop('color', axis=1)

    features_df['centroid_x'] = features_df['centroid'].apply(lambda x: x[1])
    features_df['centroid_y'] = features_df['centroid'].apply(lambda x: x[0])
    features_df = features_df.drop('centroid', axis=1)
    
    features_df.groupby('label').mean()['cells_in_image']

    features_df['avg_cells'] = features_df.groupby('label')['cells_in_image'].transform('mean')
    features_df = features_df.drop('cells_in_image', axis=1)
    selected_columns = ['equivalent_diameter', 'area', 'R', 'B', 'avg_cells']
    drop_columns_mask_test = ~features_df.columns.isin(selected_columns)
    drop_columns_test = features_df.columns[drop_columns_mask_test]
    features_df = features_df.drop(columns=drop_columns_test)
    
    return features_df
    

# ========================================== 
# fonction pour la mise en echelle des données

def scale_data(data):
    scaler = StandardScaler()

    column_names = data.columns
    scaler.fit(data)

    scaled_features = scaler.transform(data)
    scale_data_df = pd.DataFrame(scaled_features, columns=column_names)
    return scale_data_df