"""
Тренировка архитектурного классификатора
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import argparse
import os
from tqdm import tqdm
import json

def train_model():
    # Настройка устройства
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Используется устройство: {device}")
    
    # Трансформации данных
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    transform_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Загрузка данных
    train_dataset = datasets.ImageFolder('dataset', transform=transform_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # Модель
    model = models.resnet50(pretrained=True)
    num_classes = len(train_dataset.classes)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)
    
    # Оптимизатор и функция потерь
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Сохранение маппинга классов
    class_mapping = {i: class_name for i, class_name in enumerate(train_dataset.classes)}
    with open('class_mapping.json', 'w') as f:
        json.dump(class_mapping, f, indent=2)
    
    # Тренировка
    model.train()
    for epoch in range(5):
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/5')
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            accuracy = 100 * correct / total
            pbar.set_postfix({'Loss': f'{running_loss/total:.4f}', 'Acc': f'{accuracy:.2f}%'})
        
        print(f'Epoch {epoch+1} - Accuracy: {accuracy:.2f}%')
    
    # Сохранение модели
    torch.save(model.state_dict(), 'architecture_model.pth')
    print("Модель сохранена как architecture_model.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='dataset', help='Directory with training data')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    
    args = parser.parse_args()
    train_model()