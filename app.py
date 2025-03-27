from flask import Flask, render_template, request, jsonify
import os
from groq import Groq
from werkzeug.utils import secure_filename

app = Flask(_name_)

# Configuration
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize Groq client
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    country = request.form.get('country')
    
    # In a real app, you would process the country data here
    # For now, we'll just return the template with the country name
    return render_template('analysis.html', country=country)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    image_type = request.form.get('type')  # 'bfr' or 'aft'
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(f"{image_type}_{file.filename}")
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        return jsonify({'filename': filename})
    
    return jsonify({'error': 'File type not allowed'}), 400

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message')
    
    try:
        # Get chatbot response from Groq API
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are ForestGuard AI, an expert assistant on deforestation patterns, environmental impact, and conservation strategies. Provide concise, factual information about forest cover changes, biodiversity impacts, and sustainable solutions."
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

if _name_ == '_main_':
    app.run(debug=True)
