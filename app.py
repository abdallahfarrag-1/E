from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import pipeline
from PIL import Image
import io

app = Flask(__name__)
CORS(app)  # للسماح لصفحة GitHub Pages بالاتصال بالخادم

# تحميل نموذج رؤية مفتوح المصدر لوصف الصور
captioner = pipeline("image-to-text", model="Salesforce/blip-image-captioning-large")

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'لم يتم إرسال صورة'}), 400
    
    file = request.files['image']
    img_bytes = file.read()
    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    
    # تحليل الصورة
    result = captioner(image)
    description = result[0]['generated_text']
    
    return jsonify({'result': description})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
