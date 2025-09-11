"""
Тестирование архитектурного классификатора
"""

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import json
import timm
import os

def load_model():
    """Загрузка модели"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Попробуем загрузить EfficientNet модель
    if os.path.exists('models/fast_model.pth'):
        print("Загружается EfficientNet модель...")
        model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=25)
        checkpoint = torch.load('models/fast_model.pth', map_location=device)
        model.load_state_dict(checkpoint)
        model_name = "EfficientNet-B0"
    elif os.path.exists('architecture_model.pth'):
        print("Загружается ResNet50 модель...")
        model = models.resnet50(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, 25)
        model.load_state_dict(torch.load('architecture_model.pth', map_location=device))
        model_name = "ResNet50"
    else:
        raise FileNotFoundError("Модель не найдена!")
    
    model = model.to(device)
    model.eval()
    
    # Загрузка маппинга классов
    try:
        if os.path.exists('models/class_mapping.json'):
            with open('models/class_mapping.json', 'r') as f:
                class_mapping = json.load(f)
        else:
            with open('class_mapping.json', 'r') as f:
                class_mapping = json.load(f)
    except:
        # Стандартные классы
        class_mapping = {
            "0": "Achaemenid architecture",
            "1": "American craftsman style", 
            "2": "Art Deco architecture",
            "3": "Gothic architecture",
            "4": "Modern architecture"
        }
    
    return model, class_mapping, device, model_name

def predict_image(image_path):
    """Предсказание для изображения"""
    model, class_mapping, device, model_name = load_model()
    
    # Трансформации
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Загрузка и обработка изображения
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Предсказание
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence, predicted = torch.max(probabilities, 0)
    
    predicted_class = class_mapping.get(str(predicted.item()), "Unknown")
    confidence_percent = confidence.item() * 100
    
    print(f"Модель: {model_name}")
    print(f"Предсказанный стиль: {predicted_class}")
    print(f"Уверенность: {confidence_percent:.1f}%")
    
    return predicted_class, confidence_percent

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        predict_image(image_path)
    else:
        print("Использование: python test_model.py <путь_к_изображению>")