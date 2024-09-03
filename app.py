import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from flask import Flask, request, render_template, redirect, url_for
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['HEATMAP_FOLDER'] = 'static/heatmaps/'

# Load pre-trained models
cnn_model = load_model('retrace_cnn_model.h5')
feature_extractor = cnn_model.get_layer('dense')
svm_pipeline = make_pipeline(StandardScaler(), SVC(kernel='linear', probability=True))

# Utility functions
def preprocess_image(image_path, img_size=(128, 128), segment_size=(32, 32)):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, img_size)
    normalized_image = image / 255.0

    segments = []
    for i in range(0, img_size[0], segment_size[0]):
        for j in range(0, img_size[1], segment_size[1]):
            segment = normalized_image[i:i+segment_size[0], j:j+segment_size[1]]
            segments.append(segment)
    return normalized_image, segments

def generate_heatmap(image_path, svm_pipeline, feature_extractor, img_size=(128, 128), segment_size=(32, 32)):
    resized_image, segments = preprocess_image(image_path, img_size, segment_size)
    segments = np.array(segments).reshape(-1, img_size[0], img_size[1], 1)
    segment_features = feature_extractor.predict(segments)
    probabilities = svm_pipeline.predict_proba(segment_features)[:, 1]
    num_segments = img_size[0] // segment_size[0]
    heatmap = probabilities.reshape((num_segments, num_segments))

    plt.figure(figsize=(6, 6))
    plt.imshow(resized_image, cmap='gray')
    plt.imshow(heatmap, cmap='jet', alpha=0.5)
    plt.colorbar(label='Manipulation Probability')
    plt.axis('off')

    heatmap_filename = os.path.basename(image_path).replace('.jpg', '_heatmap.png')
    heatmap_path = os.path.join(app.config['HEATMAP_FOLDER'], heatmap_filename)
    plt.savefig(heatmap_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    return heatmap_path

def classify_image(image_path, svm_pipeline, feature_extractor, img_size=(128, 128), segment_size=(32, 32)):
    _, segments = preprocess_image(image_path, img_size, segment_size)
    segments = np.array(segments).reshape(-1, img_size[0], img_size[1], 1)
    segment_features = feature_extractor.predict(segments)
    probabilities = svm_pipeline.predict_proba(segment_features)[:, 1]
    avg_prob = probabilities.mean()
    classification = "Manipulated" if avg_prob > 0.5 else "Authentic"
    return classification

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'image' not in request.files:
        return redirect(request.url)
    file = request.files['image']
    if file.filename == '':
        return redirect(request.url)
    if file:
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        classification = classify_image(filepath, svm_pipeline, feature_extractor)
        heatmap_path = generate_heatmap(filepath, svm_pipeline, feature_extractor)

        heatmap_url = url_for('static', filename='heatmaps/' + os.path.basename(heatmap_path))
        return render_template('result.html', classification=classification, heatmap_url=heatmap_url)

if __name__ == '__main__':
    app.run(debug=True)
