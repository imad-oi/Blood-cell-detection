from flask import Flask, render_template, request, redirect, url_for
import cv2
import numpy as np
import pandas as pd
import pickle
from PIL import Image
from collections import Counter
from features import process_images
# import numpy as np
from scipy import ndimage as ndi
from skimage import morphology
from sklearn.cluster import KMeans
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
import os

from flask import jsonify


app = Flask(__name__)

# Load pre-trained models
with open('../models/svm_model2.pkl', 'rb') as f:
    model_svm = pickle.load(f)
with open('../models/dt_model2.pkl', 'rb') as f:
    model_dt = pickle.load(f)
with open('../models/rf_model2.pkl', 'rb') as f:
    model_rf = pickle.load(f)
with open('../models/knn_model2.pkl', 'rb') as f:
    model_knn = pickle.load(f)

import base64
from PIL import Image
import io

uploaded_images = None

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        images_path = []
        processed_images = []
        predictions = {
            "svm": [],
            "rf": [],
            "dt": [],
            "knn": []
        }
        files = request.files.getlist('image')
        for file in files:
            img = Image.open(file)
            img = img.resize((224, 224))  
            img.save(file.filename, 'JPEG') # save the image in the server
            images_path.append(file.filename)
            image = np.array(img)
            # Process the image
            processed_image, labels_rgb = process_image(image)
            processed_images.append({
                "original": img,
                "a_channel": processed_image['a_channel'],
                "thresholded": processed_image['thresholded'],
                "segmented": processed_image['segmented'],
                "watershed": labels_rgb
            })

            # save the processed image in server 
            cv2.imwrite(f"../flask/static/images/original_{file.filename}", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            cv2.imwrite(f"../flask/static/images/a_channel_{file.filename}", cv2.cvtColor(processed_image['a_channel'], cv2.COLOR_BGR2RGB))
            cv2.imwrite(f"../flask/static/images/thresholded_{file.filename}", cv2.cvtColor(processed_image['thresholded'], cv2.COLOR_BGR2RGB))
            cv2.imwrite(f"../flask/static/images/segmented_{file.filename}", cv2.cvtColor(processed_image['segmented'], cv2.COLOR_BGR2RGB))
            labels_rgb_normalized = ((labels_rgb - labels_rgb.min()) * (255 / (labels_rgb.max() - labels_rgb.min()))).astype(np.uint8)
            cv2.imwrite(f"../flask/static/images/watershed_{file.filename}", cv2.cvtColor(labels_rgb_normalized, cv2.COLOR_RGB2BGR))
        # Feature extraction and prediction
        labeled_features_testing = []
        process_images([cv2.imread(img) for img in images_path], images_path, labeled_features_testing, False)

        df_labeled_features_testing = pd.DataFrame(labeled_features_testing)

        if not df_labeled_features_testing.empty:
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

            labeled_features_testing_df = labeled_features_testing_df.drop(columns=['label'])

            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()

            # column_names = labeled_features_testing_df.columns
            scaler.fit(labeled_features_testing_df)

            scaled_features = scaler.transform(labeled_features_testing_df)

            predictions["svm"] = model_svm.predict(scaled_features)
            predictions["rf"] = model_rf.predict(scaled_features)
            predictions["dt"] = model_dt.predict(scaled_features)
            predictions["knn"] = model_knn.predict(scaled_features)

            # make the predictions serializable
            predictions = {k: v.tolist() for k, v in predictions.items()}

            # return the json response
            return jsonify({"predictions": predictions})

        # return render_template('index.html', processed_images=processed_images, predictions=predictions)
        # return render_template('index.html', message = "uploaded succ")

    return render_template('index.html')

def process_image(image):
    i = cv2.resize(image, (224, 224))
    image_rgb = cv2.cvtColor(i, cv2.COLOR_BGR2RGB)
    image_lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(image_lab)
    reshaped_a = a_channel.reshape(a_channel.shape[0] * a_channel.shape[1], 1)
    kmeans = KMeans(n_clusters=7, random_state=0).fit(reshaped_a)
    pixels_replaced_with_centers = kmeans.cluster_centers_[kmeans.labels_]
    clustered_image = pixels_replaced_with_centers.reshape(a_channel.shape[0], a_channel.shape[1])
    clustered_image = clustered_image.astype(np.uint8)
    ret, thresholded_image = cv2.threshold(clustered_image, 141, 255, cv2.THRESH_BINARY)
    filled_image = ndi.binary_fill_holes(thresholded_image)
    cleaned_image_1 = morphology.remove_small_objects(filled_image, 200)
    cleaned_image_2 = morphology.remove_small_holes(cleaned_image_1, 250)
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

    return {
        "a_channel": a_channel,
        "thresholded": thresholded_image,
        "segmented": segmented_image_bgr
    }, labels_rgb

if __name__ == '__main__':
    app.run(debug=True)
