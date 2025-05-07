from flask import Flask, request, jsonify, render_template, send_from_directory, url_for
from groq import Groq
import numpy as np
import cv2
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from datetime import datetime

# Initialize Flask app
app = Flask(__name__)
app.config['STATIC_IMAGES'] = 'static/images'
os.makedirs(app.config['STATIC_IMAGES'], exist_ok=True)


# Configuration
GROQ_API_KEY = "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"

# Then later in your code
client = Groq(api_key=GROQ_API_KEY)

# Country image mappings (PNG format)
COUNTRY_IMAGES = {
    "India": {
        "before": "india-before.png",
        "after": "india-after.png"
    },
    "China": {
        "before": "china-before.png",
        "after": "china-after.png"
    },
    "Indonesia": {
        "before": "indonesia-before.png",
        "after": "indonesia-after.png"
    },
    "Japan": {
        "before": "japan-before.png",
        "after": "japan-after.png"
    },
    "Nepal":{
        "before": "nepal-before.png",
        "after": "nepal-after.png"
    }
}

def validate_image(image_path):
    """Validate that an image exists and is readable"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    if len(img.shape) != 3 or img.shape[2] != 4:
        raise ValueError("Image must be in RGBA format (4 channels)")
    return img

def calculate_ndvi(image_path, target_size=None):
    """
    Calculate NDVI from PNG image with optional resizing
    Assumes RGBA format where:
    - R (channel 0) = Red band
    - A (channel 3) = NIR band
    """
    try:
        img = validate_image(image_path)
        
        # Extract bands
        red = img[:, :, 0].astype(float)   # Red band
        nir = img[:, :, 3].astype(float)   # NIR band in alpha channel
        
        # Resize if target_size is provided (width, height)
        if target_size:
            red = cv2.resize(red, target_size, interpolation=cv2.INTER_CUBIC)
            nir = cv2.resize(nir, target_size, interpolation=cv2.INTER_CUBIC)
            print(f"Resized image to: {red.shape[::-1]}")  # Print as (width, height)
        
        # Calculate NDVI with numerical stability
        denominator = (nir + red + 1e-10)  # Small constant to avoid division by zero
        ndvi = (nir - red) / denominator
        
        print(f"\nNDVI Calculation for {os.path.basename(image_path)}:")
        print(f"Image dimensions: {ndvi.shape}")
        print(f"Red band range: {np.min(red):.2f} to {np.max(red):.2f}")
        print(f"NIR band range: {np.min(nir):.2f} to {np.max(nir):.2f}")
        print(f"NDVI range: {np.min(ndvi):.2f} to {np.max(ndvi):.2f}")
        
        return ndvi
    except Exception as e:
        print(f"Error in calculate_ndvi: {str(e)}")
        raise

def generate_ndvi_plot(ndvi, title):
    """Generate visualization of NDVI data as base64 encoded image"""
    plt.figure(figsize=(8, 6))
    img = plt.imshow(ndvi, cmap="RdYlGn", vmin=-1, vmax=1)
    plt.title(title, fontsize=12)
    plt.axis("off")
    cbar = plt.colorbar(img, fraction=0.046, pad=0.04)
    cbar.set_label('NDVI Value', rotation=270, labelpad=15)
    
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('ascii')

def analyze_deforestation(ndvi_before, ndvi_after):
    """Calculate deforestation metrics from NDVI arrays"""
    VEGETATION_THRESHOLD = 0.4
    
    # Create vegetation masks
    veg_before = ndvi_before > VEGETATION_THRESHOLD
    veg_after = ndvi_after > VEGETATION_THRESHOLD
    
    # Calculate metrics - convert to native Python types immediately
    total_veg_before = int(np.sum(veg_before))  # Convert to Python int
    total_veg_after = int(np.sum(veg_after))    # Convert to Python int
    vegetation_lost = int(np.sum(veg_before & ~veg_after))  # Convert to Python int
    vegetation_gained = int(np.sum(~veg_before & veg_after))  # Convert to Python int
    
    # Calculate net deforestation percentage
    if total_veg_before > 0:
        net_deforestation = float(((vegetation_lost - vegetation_gained) / total_veg_before) * 100)  # Convert to Python float
    else:
        net_deforestation = 0.0  # As float
    
    # Print detailed results
    print("\nDeforestation Analysis Results:")
    print(f"Total Vegetation Before: {total_veg_before} pixels")
    print(f"Total Vegetation After: {total_veg_after} pixels")
    print(f"Vegetation Lost: {vegetation_lost} pixels")
    print(f"Vegetation Gained: {vegetation_gained} pixels")
    print(f"Net Deforestation Percentage: {net_deforestation:.2f}%")
    
    return {
        'total_veg_before': total_veg_before,
        'total_veg_after': total_veg_after,
        'vegetation_lost': vegetation_lost,
        'vegetation_gained': vegetation_gained,
        'deforestation_percentage': net_deforestation
    }
@app.route('/')
def index():
    """Render the main index page"""
    return render_template('index.html')

@app.route('/analyze', methods=['GET', 'POST'])
def analyze():
    """Render the analysis page for selected country"""
    country = request.form.get('country', 'India') if request.method == 'POST' else request.args.get('country', 'India')
    
    country_data = COUNTRY_IMAGES.get(country, COUNTRY_IMAGES["India"])
    return render_template('analyze.html', 
                         country=country,
                         before_image=url_for('static', filename=f'images/{country_data["before"]}'),
                         after_image=url_for('static', filename=f'images/{country_data["after"]}'))

@app.route('/process', methods=['POST'])
def process():
    """Process the images with automatic resizing"""
    try:
        data = request.json
        country = data.get('country', 'India')
        
        print(f"\nStarting analysis for {country}...")
        country_data = COUNTRY_IMAGES.get(country, COUNTRY_IMAGES["India"])
        
        # Get image paths
        before_path = os.path.join(app.config['STATIC_IMAGES'], country_data["before"])
        after_path = os.path.join(app.config['STATIC_IMAGES'], country_data["after"])

        # Get dimensions of before image
        before_img = cv2.imread(before_path, cv2.IMREAD_UNCHANGED)
        target_size = (before_img.shape[1], before_img.shape[0])  # (width, height)
        
        # Calculate NDVI with consistent sizing
        ndvi_before = calculate_ndvi(before_path)
        ndvi_after = calculate_ndvi(after_path, target_size=target_size)

        # Analyze deforestation
        results = analyze_deforestation(ndvi_before, ndvi_after)
        
        # Generate visualizations
        before_plot = generate_ndvi_plot(ndvi_before, f"NDVI Before ({country})")
        after_plot = generate_ndvi_plot(ndvi_after, f"NDVI After ({country})")

        return jsonify({
            'result': {
                **results,
                'before_plot': before_plot,
                'after_plot': after_plot,
                'before_image': url_for('static', filename=f'images/{country_data["before"]}'),
                'after_image': url_for('static', filename=f'images/{country_data["after"]}')
            }
        })

    except Exception as e:
        error_msg = f"Analysis failed: {str(e)}"
        print(error_msg)
        return jsonify({'error': error_msg}), 400

@app.route('/chat', methods=['POST'])    
def chat():
    user_message = request.json.get('message')
    
    try:
        # Get chatbot response from Groq API
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are ForestGuard AI, an expert assistant on deforestation patterns, environmental impact, and conservation strategies. Provide concise, factual information about forest cover changes, biodiversity impacts, and sustainable solutions.Make the answer as concise as possible with an average of 100-300 tokens in the response"
                },
                {
                    "role": "user",
                    "content": user_message
                }
            ],
            model="llama3-70b-8192",  # or any other model you prefer
            temperature=0.3,
            max_tokens=1024
        )
        
        bot_response = chat_completion.choices[0].message.content
        return jsonify({'response': bot_response})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/static/<path:filename>')
def serve_static(filename):
    """Serve static files"""
    return send_from_directory('static', filename)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
