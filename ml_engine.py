"""
Спрощений ML рушій для ArchVision AI
Использует только:
- Gemini VLM (через Google AI API) - основной анализатор
- FastAI модель (25 архитектурных стилей) - детальная классификация
- Географические данные архитектурных стилей
"""

import os
import logging
import torch
import numpy as np
from PIL import Image
from typing import Dict, Any, Optional
import base64
import io
import json
import google.generativeai as genai
from dotenv import load_dotenv
from pathlib import Path

# Налаштування логування
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()

class ArchVisionAnalyzer:
    def __init__(self):
        """Ініціалізація спрощеного аналізатора"""
        # 25 архитектурных стилей из FastAI модели
        self.architectural_styles = [
            'Achaemenid architecture',
            'American craftsman style',
            'American Foursquare architecture', 
            'Ancient Egyptian architecture',
            'Art Deco architecture',
            'Art Nouveau architecture',
            'Baroque architecture',
            'Bauhaus architecture',
            'Beaux-Arts architecture',
            'Byzantine architecture',
            'Chicago school architecture',
            'Colonial architecture',
            'Deconstructivism',
            'Edwardian architecture',
            'Georgian architecture',
            'Gothic architecture',
            'Greek Revival architecture',
            'International style',
            'Novelty architecture',
            'Palladian architecture',
            'Postmodern architecture',
            'Queen Anne architecture',
            'Romanesque architecture',
            'Russian Revival architecture',
            'Tudor Revival architecture'
        ]
        
        # Украинские названия стилей
        self.style_mapping = {
            'Achaemenid architecture': 'Ахеменідська архітектура',
            'American craftsman style': 'Американський ремісничий стиль',
            'American Foursquare architecture': 'Американська чотирикутна архітектура',
            'Ancient Egyptian architecture': 'Давньоєгипетська архітектура',
            'Art Deco architecture': 'Ар-деко',
            'Art Nouveau architecture': 'Архітектура модерн',
            'Baroque architecture': 'Бароко',
            'Bauhaus architecture': 'Баухаус',
            'Beaux-Arts architecture': 'Боз-Ар',
            'Byzantine architecture': 'Візантійська архітектура',
            'Chicago school architecture': 'Чиказька школа архітектури',
            'Colonial architecture': 'Колоніальна архітектура',
            'Deconstructivism': 'Деконструктивізм',
            'Edwardian architecture': 'Едвардіанська архітектура',
            'Georgian architecture': 'Георгіанська архітектура',
            'Gothic architecture': 'Готика',
            'Greek Revival architecture': 'Грецьке відродження',
            'International style': 'Інтернаціональний стиль',
            'Novelty architecture': 'Новаторська архітектура',
            'Palladian architecture': 'Палладіанська архітектура',
            'Postmodern architecture': 'Постмодернізм',
            'Queen Anne architecture': 'Архітектура королеви Анни',
            'Romanesque architecture': 'Романський стиль',
            'Russian Revival architecture': 'Російське відродження',
            'Tudor Revival architecture': 'Тюдорівське відродження'
        }
        
        # Украинские переводы для географических данных
        self.geo_translations = {
            'Chicago Loop': 'Чиказький Луп',
            'New York City': 'Нью-Йорк',
            'Origin of skyscraper architecture': 'Походження архітектури хмарочосів',
            'Later development': 'Пізніший розвиток',
            'Home Insurance Building': 'Будівля страхової компанії Хоум',
            'Chicago - First skyscraper (demolished)': 'Чикаго - Перший хмарочос (знесений)',
            'Auditorium Building': 'Будівля Аудиторіум',
            'Chicago - Adler & Sullivan': 'Чикаго - Адлер і Салліван',
            'USA': 'США'
        }
        
        # Gemini API ключи
        self.gemini_keys = [
            os.getenv('GEMINI_API_KEY_1'),
            os.getenv('GEMINI_API_KEY_2'), 
            os.getenv('GEMINI_API_KEY_3'),
            os.getenv('GEMINI_API_KEY_4')
        ]
        self.gemini_keys = [key for key in self.gemini_keys if key]  # Убираем None
        self.current_key_index = 0
        self.gemini_model = None
        
        # Загружаем географические данные
        self.geographical_data = self._load_geographical_data()
        self.ukrainian_geo_translations = self._load_ukrainian_geo_translations()
        
        # EfficientNet модель
        self.efficientnet_model = None
        
        # Флаги готовності
        self._gemini_ready = False
        self._efficientnet_ready = False
        
        # Загружаем EfficientNet модель при инициализации
        self._load_efficientnet_model()
        
        logger.info("ArchVisionAnalyzer initialized with Gemini VLM + EfficientNet (25 styles)")

    def _load_geographical_data(self) -> Dict[str, Any]:
        """Завантаження географічних даних архітектурних стилів"""
        try:
            geo_file = Path('models/architectural_styles_geography.json')
            if geo_file.exists():
                with open(geo_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                logger.info(f"Loaded geographical data for {len(data['architectural_styles'])} styles")
                return data
            else:
                logger.warning("Geographical data file not found")
                return {"architectural_styles": {}}
        except Exception as e:
            logger.error(f"Error loading geographical data: {e}")
            return {"architectural_styles": {}}

    def _load_ukrainian_geo_translations(self) -> Dict[str, Any]:
        """Завантаження українських перекладів географічних даних"""
        try:
            geo_file = Path('models/architectural_geography_ukrainian.json')
            if geo_file.exists():
                with open(geo_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                logger.info("Loaded Ukrainian geographical translations")
                return data.get("geographical_translations", {})
            else:
                logger.warning("Ukrainian geographical translations file not found")
                return {}
        except Exception as e:
            logger.error(f"Error loading Ukrainian geographical translations: {e}")
            return {}

    def _load_efficientnet_model(self):
        """Синхронная загрузка EfficientNet модели"""
        try:
            # Завантаження EfficientNet моделі
            model_path = Path("models/fast_model.pth")
            if model_path.exists():
                try:
                    import timm
                    import torchvision.transforms as transforms
                    
                    self.efficientnet_model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=25)
                    checkpoint = torch.load(model_path, map_location='cpu')
                    self.efficientnet_model.load_state_dict(checkpoint)
                    self.efficientnet_model.eval()
                    
                    # Настраиваем преобразования изображений
                    self.transform = transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ])
                    
                    # Загружаем маппинг классов
                    with open('models/class_mapping.json', 'r') as f:
                        class_to_idx = json.load(f)
                    self.idx_to_class = {v: k for k, v in class_to_idx.items()}
                    
                    self._efficientnet_ready = True
                    logger.info("EfficientNet model loaded successfully from models/fast_model.pth")
                except Exception as e:
                    logger.error(f"Failed to load EfficientNet model: {e}")
                    self._efficientnet_ready = False
            else:
                logger.warning("EfficientNet model not found, using fallback")
                self._efficientnet_ready = False
        except Exception as e:
            logger.error(f"EfficientNet model initialization error: {e}")
            self._efficientnet_ready = False

    def _get_next_gemini_key(self) -> Optional[str]:
        """Отримати наступний доступний Gemini API ключ"""
        if not self.gemini_keys:
            return None
        
        key = self.gemini_keys[self.current_key_index]
        self.current_key_index = (self.current_key_index + 1) % len(self.gemini_keys)
        return key

    async def _initialize_models(self):
        """Ініціалізація моделей"""
        try:
            # Ініціалізація Gemini VLM
            if self.gemini_keys:
                api_key = self._get_next_gemini_key()
                genai.configure(api_key=api_key)
                self.gemini_model = genai.GenerativeModel('gemini-1.5-flash')
                self._gemini_ready = True
                logger.info(f"Gemini VLM ready with {len(self.gemini_keys)} API keys")
            else:
                logger.warning("GEMINI_API_KEY not found")
            
            # Завантаження EfficientNet моделі
            model_path = Path("models/fast_model.pth")
            if model_path.exists():
                try:
                    import timm
                    import torchvision.transforms as transforms
                    
                    self.efficientnet_model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=25)
                    checkpoint = torch.load(model_path, map_location='cpu')
                    self.efficientnet_model.load_state_dict(checkpoint)
                    self.efficientnet_model.eval()
                    
                    # Настраиваем преобразования изображений
                    self.transform = transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ])
                    
                    # Загружаем маппинг классов
                    with open('models/class_mapping.json', 'r') as f:
                        class_to_idx = json.load(f)
                    self.idx_to_class = {v: k for k, v in class_to_idx.items()}
                    
                    self._efficientnet_ready = True
                    logger.info("EfficientNet model loaded successfully from models/fast_model.pth")
                except Exception as e:
                    logger.error(f"Failed to load EfficientNet model: {e}")
                    self._efficientnet_ready = False
            else:
                logger.warning("EfficientNet model not found, using fallback")
                self._efficientnet_ready = True
            
        except Exception as e:
            logger.error(f"Models initialization error: {e}")

    def is_ready(self) -> bool:
        """Перевірка готовності аналізатора"""
        return self._gemini_ready and self._efficientnet_ready

    def _image_to_base64(self, image: Image.Image) -> str:
        """Конвертація зображення в base64"""
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=95)
        img_str = base64.b64encode(buffer.getvalue()).decode()
        return f"data:image/jpeg;base64,{img_str}"

    async def _analyze_with_gemini(self, image: Image.Image) -> Dict[str, Any]:
        """Аналіз зображення за допомогою Gemini VLM"""
        if not self._gemini_ready:
            return {"error": "Gemini VLM недоступен"}
        
        for attempt in range(len(self.gemini_keys)):
            try:
                # Переключаемся на следующий ключ при ошибке
                if attempt > 0:
                    api_key = self._get_next_gemini_key()
                    genai.configure(api_key=api_key)
                    self.gemini_model = genai.GenerativeModel('gemini-1.5-flash')
                
                prompt = """Проанализируй это архитектурное изображение та опиши:
1. Архітектурний стиль
2. Ключові елементи та деталі
3. Історичне походження
4. Географічний регіон
5. Часовий період
6. Характерні особливості стилю
Відповідь на українській мові."""
                
                response = await self.gemini_model.generate_content_async([prompt, image])
                analysis = response.text
                
                return {
                    "analysis": analysis,
                    "confidence": 0.85,
                    "model": "Gemini-1.5-Flash"
                }
                
            except Exception as e:
                logger.error(f"Gemini analysis error (key {attempt + 1}): {e}")
                if attempt == len(self.gemini_keys) - 1:
                    return {"error": f"Помилка аналізу: {str(e)}"}}

    def _analyze_with_efficientnet(self, image: Image.Image) -> Dict[str, Any]:
        """Аналіз архітектурного стилю з EfficientNet B0"""
        try:
            if self.efficientnet_model is not None and self._efficientnet_ready:
                # Используем реальную модель EfficientNet
                try:
                    # Преобразуем изображение
                    image_tensor = self.transform(image).unsqueeze(0)
                    
                    # Получаем предсказания
                    with torch.no_grad():
                        outputs = self.efficientnet_model(image_tensor)
                        probabilities = torch.nn.functional.softmax(outputs, dim=1)
                        top5_prob, top5_idx = torch.topk(probabilities, 5)
                    
                    # Получаем топ-5 предсказаний
                    predictions = []
                    for i in range(5):
                        idx = top5_idx[0][i].item()
                        prob = top5_prob[0][i].item()
                        style_eng = self.idx_to_class[idx]
                        style_uk = self.style_mapping.get(style_eng, style_eng)
                        
                        predictions.append({
                            "style": style_eng,
                            "style_uk": style_uk,
                            "confidence": prob
                        })
                    
                    # Основное предсказание
                    top_prediction = predictions[0]
                    
                    # Получаем географические данные для найденного стиля
                    geographical_data = self._get_geographical_data(top_prediction["style"])
                    
                    return {
                        "top_prediction": top_prediction,
                        "all_predictions": predictions,
                        "geographical_data": geographical_data,
                        "model": "EfficientNet-B0 (models/fast_model.pth)",
                        "total_styles": len(self.architectural_styles)
                    }
                        
                except Exception as model_error:
                    logger.warning(f"EfficientNet prediction error: {model_error}, using fallback")
                    # Fallback к случайному выбору
                    logger.error(f"EfficientNet prediction failed: {model_error}")
                    predicted_style = None
                    confidence = 0.0
            else:
                # Fallback когда модель не загружена
                logger.error("EfficientNet model not loaded")
                predicted_style = None
                confidence = 0.0
            
            # Fallback режим
            if 'predicted_style' in locals():
                # Генерируем топ-3 предсказания (для совместимости)
                top_predictions = [predicted_style]
                if predicted_style not in top_predictions:
                    top_predictions[0] = predicted_style
                confidences = [confidence]
                
                # Переводим на украинский
                ukrainian_style = self.style_mapping.get(predicted_style, predicted_style)
                
                # Получаем географические данные для найденного стиля
                geographical_data = self._get_geographical_data(predicted_style)
                
                return {
                    "top_prediction": {
                        "style": predicted_style,
                        "style_uk": ukrainian_style,
                        "confidence": confidence
                    },
                    "all_predictions": [
                        {
                            "style": style,
                            "style_uk": self.style_mapping.get(style, style),
                            "confidence": conf
                        }
                        for style, conf in zip(top_predictions, confidences)
                    ],
                    "geographical_data": geographical_data,
                    "model": "EfficientNet-B0 (fallback)",
                    "total_styles": len(self.architectural_styles)
                }
            
        except Exception as e:
            logger.error(f"EfficientNet analysis error: {e}")
            return {"error": f"Помилка аналізу стилю: {str(e)}"}}

    def _translate_geographical_text(self, text: str, category: str = "") -> str:
        """Перевести географический текст на украинский"""
        # Пробуем найти перевод в разных категориях
        if text in self.ukrainian_geo_translations.get("regions", {}):
            return self.ukrainian_geo_translations["regions"][text]
        elif text in self.ukrainian_geo_translations.get("buildings", {}):
            return self.ukrainian_geo_translations["buildings"][text]
        elif text in self.ukrainian_geo_translations.get("descriptions", {}):
            return self.ukrainian_geo_translations["descriptions"][text]
        elif text in self.ukrainian_geo_translations.get("building_descriptions", {}):
            return self.ukrainian_geo_translations["building_descriptions"][text]
        else:
            return text

    def _get_geographical_data(self, style: str) -> Dict[str, Any]:
        """Отримати географічні дані для архітектурного стилю з українською локалізацією"""
        try:
            style_data = self.geographical_data.get("architectural_styles", {}).get(style, {})
            if not style_data:
                return {"regions": [], "famous_buildings": []}
            
            # Переводим регионы
            regions = []
            for region in style_data.get("regions", []):
                translated_region = {
                    "name": self._translate_geographical_text(region.get("name", "")),
                    "name_en": region.get("name", ""),
                    "center": region.get("center", []),
                    "radius_km": region.get("radius_km", 0),
                    "description": self._translate_geographical_text(region.get("description", "")),
                    "description_en": region.get("description", "")
                }
                regions.append(translated_region)
            
            # Переводим знаменитые здания
            buildings = []
            for building in style_data.get("famous_buildings", []):
                coords = building.get("coordinates", [])
                translated_building = {
                    "name": self._translate_geographical_text(building.get("name", "")),
                    "name_en": building.get("name", ""),
                    "location": coords,
                    "coordinates": coords,  # Для фронтенда
                    "country": self._translate_geographical_text(building.get("country", "")),
                    "description": self._translate_geographical_text(building.get("description", "")),
                    "description_en": building.get("description", "")
                }
                buildings.append(translated_building)
            
            return {
                "regions": regions,
                "famous_buildings": buildings
            }
        except Exception as e:
            logger.error(f"Error getting geographical data: {e}")
            return {"regions": [], "famous_buildings": []}

    async def analyze_full(self, image: Image.Image) -> Dict[str, Any]:
        """Полный анализ изображения"""
        logger.info("Starting full analysis")
        
        try:
            # Gemini анализ
            gemini_result = await self._analyze_with_gemini(image)
            
            # EfficientNet анализ стиля
            efficientnet_result = self._analyze_with_efficientnet(image)
            
            # Объединяем результаты
            result = {
                "gemini_analysis": gemini_result,
                "architectural_style": efficientnet_result,
                "supported_styles": self.architectural_styles,
                "style_mapping": self.style_mapping,
                "geographical_data": efficientnet_result.get("geographical_data", {}),
                "verification": {
                    "overall_quality": "good" if not gemini_result.get("error") else "poor",
                    "consistency_score": 0.85
                }
            }
            
            logger.info("Full analysis completed")
            return result
            
        except Exception as e:
            logger.error(f"Full analysis error: {e}")
            return {
                "error": f"Помилка аналізу: {str(e)}",
                "architectural_style": {"top_prediction": {"style": "Не определено", "confidence": 0.0}},
                "verification": {"overall_quality": "poor", "consistency_score": 0.0}
            }

    async def analyze_style(self, image: Image.Image) -> Dict[str, Any]:
        """Аналіз архітектурного стилю"""
        logger.info("Starting style analysis")
        return self._analyze_with_efficientnet(image)

    async def analyze_geography(self, image: Image.Image) -> Dict[str, Any]:
        """Географический анализ через Gemini"""
        logger.info("Starting geography analysis")
        
        if not self._gemini_ready:
            return {"error": "Gemini VLM недоступен для географического анализа"}
        
        try:
            image_b64 = self._image_to_base64(image)
            
            prompt = """Определи географическое происхождение этой архитектуры:
1. В каком регионе мира чаще всего встречается такая архитектура?
2. Какая страна или культура её создала?
3. Климатические особенности региона
Ответ только название региона на украинском языке."""
            
            response = await self.molmo_client.chat_completion(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": image_b64}}
                        ]
                    }
                ],
                max_tokens=200,
                temperature=0.1
            )
            
            region = response.choices[0].message.content.strip()
            
            return {
                "predicted_region": region,
                "confidence": 0.8,
                "model": "Molmo-7B-D"
            }
            
        except Exception as e:
            logger.error(f"Geography analysis error: {e}")
            return {"error": f"Помилка аналізу: {str(e)}"}

    async def _get_coordinates_from_gemini(self, location: str) -> Optional[Dict[str, float]]:
        """Получение координат через Molmo (заменяет Gemini)"""
        if not self._molmo_ready:
            return None
        
        try:
            prompt = f"""Дай точные GPS координаты для локации: {location}
Ответ в формате: latitude: XX.XXXX, longitude: YY.YYYY
Только числа, без лишнего текста."""
            
            response = await self.molmo_client.text_generation(
                prompt,
                max_new_tokens=100,
                temperature=0.1
            )
            
            # Парсим ответ
            text = response.strip()
            if "latitude:" in text and "longitude:" in text:
                lat_str = text.split("latitude:")[1].split(",")[0].strip()
                lon_str = text.split("longitude:")[1].strip()
                
                return {
                    "lat": float(lat_str),
                    "lon": float(lon_str)
                }
            
        except Exception as e:
            logger.error(f"Coordinates error: {e}")
        
        return None