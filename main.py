from flask import Flask, request, jsonify, send_file
import rasterio
import numpy as np
import os

app = Flask(_name_)

# Dictionary to store country names and their corresponding image file names
deforestation_data = {
    "China": ("china-before.png", "china-after.png"),
    "India": ("india-before.png", "india-after.png"),
    "Indonesia": ("indonesia-before.png", "indonesia-after.png")
}

# Function to calculate NDVI
def calculate_ndvi(image_path):
    with rasterio.open(image_path) as src:
        red = src.read(3).astype(float)  # Red band
        nir = src.read(4).astype(float)  # Near-infrared band
        ndvi = (nir - red) / (nir + red + 1e-5)  # NDVI formula
        return ndvi

# Function to compare before & after images and detect deforestation
def compare_deforestation(before_image, after_image, threshold=0.3):
    ndvi_before = calculate_ndvi(before_image)
    ndvi_after = calculate_ndvi(after_image)
    
    min_shape = (min(ndvi_before.shape[0], ndvi_after.shape[0]),
                 min(ndvi_before.shape[1], ndvi_after.shape[1]))
    ndvi_before = ndvi_before[:min_shape[0], :min_shape[1]]
    ndvi_after = ndvi_after[:min_shape[0], :min_shape[1]]
    
    veg_before = ndvi_before > threshold
    veg_after = ndvi_after > threshold
    
    vegetation_lost = np.sum(veg_before & ~veg_after)
    total_vegetation_before = np.sum(veg_before)
    
    deforestation_percentage = (vegetation_lost / total_vegetation_before) * 100 if total_vegetation_before > 0 else 0
    
    return deforestation_percentage

@app.route('/process', methods=['POST'])
def process_deforestation():
    data = request.json
    country_name = data.get("country")
    
    if country_name not in deforestation_data:
        return jsonify({"error": "Country data not found"}), 404
    
    before_image, after_image = deforestation_data[country_name]
    
    deforestation_percentage = compare_deforestation(before_image, after_image)
    
    return jsonify({
        "deforestation_percentage": f"{deforestation_percentage:.2f}%",
        "before_image": before_image,
        "after_image": after_image
    })

@app.route('/get_image/<image_name>')
def get_image(image_name):
    image_path = os.path.join(os.getcwd(), image_name)
    if os.path.exists(image_path):
        return send_file(image_path, mimetype='image/png')
    return jsonify({"error": "Image not found"}), 404

if _name_ == '_main_':
    app.run(debug=True)
