"""
ArchVision AI Web Application
Простий веб-сервер для аналізу архітектурних зображень
"""

from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import os
import base64
from ml_engine import ArchVisionAnalyzer
import logging

app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size
analyzer = ArchVisionAnalyzer()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="uk">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ArchVision AI - Архітектурний Аналізатор</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        .container {
            background: white;
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }
        h1 {
            text-align: center;
            color: #333;
            margin-bottom: 30px;
            font-size: 2.5em;
        }
        .upload-area {
            border: 3px dashed #667eea;
            border-radius: 10px;
            padding: 40px;
            text-align: center;
            margin: 20px 0;
            background: #f8f9ff;
            transition: all 0.3s ease;
        }
        .upload-area:hover {
            border-color: #764ba2;
            background: #f0f2ff;
        }
        .upload-area.dragover {
            border-color: #28a745;
            background: #e8f5e8;
        }
        input[type="file"] {
            display: none;
        }
        .upload-btn {
            background: linear-gradient(45deg, #667eea, #764ba2);
            color: white;
            padding: 12px 30px;
            border: none;
            border-radius: 25px;
            cursor: pointer;
            font-size: 16px;
            transition: transform 0.2s;
        }
        .upload-btn:hover {
            transform: translateY(-2px);
        }
        .analyze-btn {
            background: linear-gradient(45deg, #28a745, #20c997);
            color: white;
            padding: 15px 40px;
            border: none;
            border-radius: 25px;
            cursor: pointer;
            font-size: 18px;
            margin: 20px 0;
            width: 100%;
            transition: transform 0.2s;
        }
        .analyze-btn:hover {
            transform: translateY(-2px);
        }
        .analyze-btn:disabled {
            background: #ccc;
            cursor: not-allowed;
            transform: none;
        }
        .preview {
            max-width: 100%;
            max-height: 400px;
            border-radius: 10px;
            margin: 20px 0;
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }
        .results {
            background: #f8f9fa;
            border-radius: 10px;
            padding: 20px;
            margin: 20px 0;
            border-left: 5px solid #667eea;
        }
        .loading {
            text-align: center;
            padding: 20px;
            color: #667eea;
            font-size: 18px;
        }
        .style-result {
            background: linear-gradient(45deg, #667eea, #764ba2);
            color: white;
            padding: 15px;
            border-radius: 10px;
            margin: 10px 0;
            font-size: 18px;
            font-weight: bold;
        }
        .confidence {
            background: #28a745;
            color: white;
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 14px;
            margin-left: 10px;
        }
        .analysis-section {
            margin: 15px 0;
            padding: 15px;
            background: white;
            border-radius: 8px;
            border: 1px solid #e0e0e0;
        }
        .section-title {
            font-weight: bold;
            color: #333;
            margin-bottom: 10px;
            font-size: 16px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🏛️ ArchVision AI</h1>
        <p style="text-align: center; color: #666; font-size: 18px;">
            Универсальный архитектурный анализатор с ИИ
        </p>
        
        <div class="upload-area" id="uploadArea">
            <p style="font-size: 18px; color: #666;">
                Перетягніть зображення сюди або клацніть для вибору файлу
            </p>
            <input type="file" id="imageInput" accept="image/*">
            <button class="upload-btn" onclick="document.getElementById('imageInput').click()">
                Вибрати зображення
            </button>
        </div>
        
        <div id="imagePreview"></div>
        
        <button class="analyze-btn" id="analyzeBtn" onclick="analyzeImage()" disabled>
            Аналізувати архітектуру
        </button>
        
        <div id="results"></div>
    </div>

    <script>
        let selectedImage = null;
        
        const uploadArea = document.getElementById('uploadArea');
        const imageInput = document.getElementById('imageInput');
        const imagePreview = document.getElementById('imagePreview');
        const analyzeBtn = document.getElementById('analyzeBtn');
        const results = document.getElementById('results');
        
        uploadArea.addEventListener('click', () => imageInput.click());
        
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });
        
        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('dragover');
        });
        
        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                handleImageUpload(files[0]);
            }
        });
        
        imageInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                handleImageUpload(e.target.files[0]);
            }
        });
        
        function handleImageUpload(file) {
            if (!file.type.startsWith('image/')) {
                alert('Будь ласка, виберіть файл зображення');
                return;
            }
            
            const reader = new FileReader();
            reader.onload = (e) => {
                selectedImage = e.target.result;
                imagePreview.innerHTML = `<img src="${selectedImage}" class="preview" alt="Вибране зображення">`;
                analyzeBtn.disabled = false;
            };
            reader.readAsDataURL(file);
        }
        
        async function analyzeImage() {
            if (!selectedImage) return;
            
            analyzeBtn.disabled = true;
            results.innerHTML = '<div class="loading">🔍 Аналізуємо архітектуру...</div>';
            
            try {
                const response = await fetch('/analyze', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        image: selectedImage.split(',')[1]
                    })
                });
                
                const data = await response.json();
                displayResults(data);
            } catch (error) {
                results.innerHTML = '<div class="results">❌ Помилка аналізу: ' + error.message + '</div>';
            } finally {
                analyzeBtn.disabled = false;
            }
        }
        
        function displayResults(data) {
            let html = '<div class="results">';
            
            if (data.architectural_style) {
                html += `
                    <div class="style-result">
                        🏛️ ${data.architectural_style}
                        <span class="confidence">${data.confidence}%</span>
                    </div>
                `;
            }
            
            if (data.ai_analysis) {
                html += `
                    <div class="analysis-section">
                        <div class="section-title">🤖 ІІ Аналіз:</div>
                        <div>${data.ai_analysis}</div>
                    </div>
                `;
            }
            
            if (data.geographical_context) {
                html += `
                    <div class="analysis-section">
                        <div class="section-title">🌍 Географічний контекст:</div>
                        <div>${data.geographical_context}</div>
                    </div>
                `;
            }
            
            if (data.historical_period) {
                html += `
                    <div class="analysis-section">
                        <div class="section-title">📅 Історичний період:</div>
                        <div>${data.historical_period}</div>
                    </div>
                `;
            }
            
            if (data.cultural_significance) {
                html += `
                    <div class="analysis-section">
                        <div class="section-title">🎭 Культурне значення:</div>
                        <div>${data.cultural_significance}</div>
                    </div>
                `;
            }
            
            html += '</div>';
            results.innerHTML = html;
        }
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    with open('index.html', 'r', encoding='utf-8') as f:
        return f.read()

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'Файл занадто великий'}), 413

@app.route('/analyze/full', methods=['POST'])
def analyze():
    try:
        
        # Обробка FormData
        if 'file' in request.files:
            file = request.files['file']
            image_bytes = file.read()
        elif 'image' in request.files:
            file = request.files['image']
            image_bytes = file.read()
        else:
            # Обробка JSON
            try:
                data = request.get_json()
                if data:
                    image_data = data.get('image')
                    if image_data:
                        # Декодуємо base64 зображення
                        image_bytes = base64.b64decode(image_data)
                    else:
                        return jsonify({'error': 'Не надано зображення'}), 400
                else:
                    return jsonify({'error': 'Не надано зображення'}), 400
            except:
                return jsonify({'error': 'Неправильний формат запиту'}), 400
        
        # Конвертуємо байти в PIL Image
        from PIL import Image
        import io
        image = Image.open(io.BytesIO(image_bytes))
        
        # Аналізуємо зображення (async метод)
        import asyncio
        result = asyncio.run(analyzer.analyze_full(image))
        
        logger.info(f"Результат аналізу: {result}")
        
        # Проверяем географические данные
        if 'architectural_style' in result and 'geographical_data' in result['architectural_style']:
            geo_data = result['architectural_style']['geographical_data']
            logger.info(f"Географические данные: regions={len(geo_data.get('regions', []))}, buildings={len(geo_data.get('famous_buildings', []))}")
        else:
            logger.warning("Географические данные не найдены в результате")
        
        # Оборачиваем в нужную структуру для фронта
        response = {
            'status': 'success',
            'data': result
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Помилка аналізу: {str(e)}")
        return jsonify({'error': f'Помилка аналізу: {str(e)}'}), 500

if __name__ == '__main__':
    print("Запуск ArchVision AI...")
    print("Відкрийте http://localhost:8002 у браузері")
    app.run(debug=True, host='0.0.0.0', port=8002)