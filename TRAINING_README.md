# Architectural Style Classification Training

Этот проект обучает нейронную сеть для классификации архитектурных стилей на основе датасета с Kaggle.

## Обзор

- **Датасет**: [dumitrux/architectural-styles-dataset](https://www.kaggle.com/datasets/dumitrux/architectural-styles-dataset)
- **Модель**: ResNet-50 (предобученная на ImageNet)
- **Классы**: 25 архитектурных стилей
- **Фреймворк**: PyTorch + torchvision

## Архитектурные стили

Модель обучается распознавать следующие 25 архитектурных стилей:

1. American craftsman style - Американський ремісничий стиль
2. American foursquare architecture - Американська чотирикутна архітектура
3. Ancient Egyptian architecture - Давньоєгипетська архітектура
4. Art nouveau architecture - Архітектура модерн
5. Baroque architecture - Бароко
6. Bauhaus architecture - Баухаус
7. Beaux-Arts architecture - Боз-Ар
8. Byzantine architecture - Візантійська архітектура
9. Chicago school architecture - Чиказька школа архітектури
10. Colonial architecture - Колоніальна архітектура
11. Deconstructivism - Деконструктивізм
12. Edwardian architecture - Едвардіанська архітектура
13. Georgian architecture - Георгіанська архітектура
14. Gothic architecture - Готика
15. Greek Revival architecture - Грецьке відродження
16. International style - Інтернаціональний стиль
17. Novelty architecture - Новаторська архітектура
18. Palladian architecture - Палладіанська архітектура
19. Postmodern architecture - Постмодернізм
20. Prairie School architecture - Школа прерій
21. Queen Anne architecture - Архітектура королеви Анни
22. Romanesque architecture - Романський стиль
23. Russian Revival architecture - Російське відродження
24. Tudor Revival architecture - Тюдорівське відродження
25. Victorian architecture - Вікторіанська архітектура

## Структура файлов

```
.
├── train_architecture_classifier.py  # Основной скрипт обучения
├── prepare_dataset.py                # Подготовка и анализ датасета
├── evaluate_model.py                 # Оценка обученной модели
├── setup_training.py                 # Автоматическая настройка окружения
├── training_requirements.txt         # Зависимости для обучения
├── kaggle.json                       # API ключи Kaggle
├── dataset/                          # Распакованный датасет
├── models/                           # Сохраненные модели
├── logs/                            # Логи обучения
├── checkpoints/                     # Промежуточные checkpoint'ы
└── results/                         # Результаты оценки
```

## Быстрый старт

### 1. Автоматическая настройка

```bash
# Установка окружения, загрузка датасета и анализ
python setup_training.py
```

### 2. Ручная настройка

```bash
# Установка зависимостей
pip install -r training_requirements.txt

# Загрузка датасета (требуется kaggle.json)
kaggle datasets download -d dumitrux/architectural-styles-dataset
unzip architectural-styles-dataset.zip -d dataset

# Анализ датасета
python prepare_dataset.py --analyze --dataset_path dataset
```

### 3. Обучение модели

```bash
# Базовое обучение
python train_architecture_classifier.py

# С настройками
python train_architecture_classifier.py \
    --epochs 100 \
    --batch_size 64 \
    --learning_rate 0.001 \
    --data_dir dataset
```

### 4. Оценка модели

```bash
# Оценка на тестовом наборе
python evaluate_model.py --model_path best_model.pth --dataset_path dataset

# Предсказание для одного изображения
python evaluate_model.py --model_path best_model.pth --image path/to/image.jpg
```

## Параметры обучения

### train_architecture_classifier.py

- `--data_dir`: Путь к датасету (по умолчанию: '.')
- `--epochs`: Количество эпох (по умолчанию: 50)
- `--batch_size`: Размер батча (по умолчанию: 32)
- `--learning_rate`: Скорость обучения (по умолчанию: 0.001)
- `--resume`: Путь к checkpoint для продолжения обучения

### prepare_dataset.py

- `--dataset_path`: Путь к датасету (по умолчанию: 'dataset')
- `--analyze`: Анализировать структуру датасета
- `--clean`: Очистить датасет от поврежденных файлов
- `--output_dir`: Папка для результатов анализа

### evaluate_model.py

- `--model_path`: Путь к обученной модели (обязательно)
- `--dataset_path`: Путь к датасету (по умолчанию: 'dataset')
- `--split`: Набор для оценки ['train', 'val', 'test'] (по умолчанию: 'test')
- `--class_mapping`: Путь к файлу маппинга классов
- `--output_dir`: Папка для результатов оценки
- `--image`: Путь к одному изображению для предсказания

## Аугментация данных

Обучение использует следующие техники аугментации:

- **Изменение размера**: 256x256 -> Random crop 224x224
- **Горизонтальный переворот**: p=0.5
- **Поворот**: ±10 градусов
- **Изменение цвета**: яркость, контрастность, насыщенность, оттенок
- **Нормализация**: ImageNet mean/std

Для валидации используется только изменение размера и нормализация.

## Архитектура модели

- **Backbone**: ResNet-50 (предобученная на ImageNet)
- **Classifier**: FC слой с Dropout (0.5) → 25 классов
- **Loss function**: CrossEntropyLoss
- **Optimizer**: Adam (lr=0.001, weight_decay=1e-4)
- **Scheduler**: StepLR (step_size=15, gamma=0.1)

## Мониторинг обучения

Обучение создает следующие файлы:

- `training.log`: Детальные логи
- `best_model.pth`: Лучшая модель по validation accuracy
- `final_model.pth`: Финальная модель
- `checkpoint_epoch_X.pth`: Checkpoint'ы каждые 10 эпох
- `training_history.png`: График loss и accuracy
- `class_mapping.json`: Маппинг классов

## Оценка модели

Скрипт оценки генерирует:

- **Метрики**: Accuracy, Top-3/Top-5 accuracy, Precision, Recall, F1
- **Визуализации**: Confusion matrix, Per-class F1 scores, Метрики сравнения
- **Отчеты**: JSON с метриками, текстовый отчет, classification report

## Требования к системе

### Минимальные требования

- **CPU**: 4+ ядер
- **RAM**: 16+ ГБ
- **Диск**: 10+ ГБ свободного места
- **Python**: 3.8+

### Рекомендуемые требования

- **GPU**: CUDA-совместимая (8+ ГБ VRAM)
- **CPU**: 8+ ядер
- **RAM**: 32+ ГБ
- **Диск**: SSD, 20+ ГБ

## Устранение проблем

### Проблема: Недостаточно памяти

```bash
# Уменьшить batch_size
python train_architecture_classifier.py --batch_size 16

# Уменьшить количество workers
# В коде: num_workers=2 вместо 4
```

### Проблема: Медленное обучение

```bash
# Проверить использование GPU
python -c "import torch; print(torch.cuda.is_available())"

# Увеличить batch_size если есть память
python train_architecture_classifier.py --batch_size 64
```

### Проблема: Переобучение

```bash
# Использовать data augmentation (уже включена)
# Добавить больше Dropout или Weight Decay
# Остановить обучение раньше при падении val accuracy
```

### Проблема: Низкая точность

1. Увеличить количество эпох
2. Попробовать другую learning rate
3. Очистить датасет от плохих изображений
4. Использовать более сильную аугментацию

## Интеграция с ArchVision AI

После обучения модель может быть интегрирована в основной проект:

1. **Замена эмуляции**: Замените эмуляцию в `ml_engine.py` на загрузку реальной модели
2. **Обновление стилей**: 25 стилей уже добавлены в ML engine
3. **Обновление frontend**: Список стилей уже добавлен в `index.html`

```python
# В ml_engine.py замените эмуляцию на:
def _analyze_with_fastai(self, image: Image.Image) -> Dict[str, Any]:
    # Загрузка реальной модели
    if not hasattr(self, 'fastai_model'):
        self.fastai_model = ArchitecturalStyleClassifier(num_classes=25)
        checkpoint = torch.load('best_model.pth')
        self.fastai_model.load_state_dict(checkpoint['model_state_dict'])
        self.fastai_model.eval()
    
    # Предсказание
    # ... код предсказания
```

## Лицензия

Датасет использует лицензию CC0-1.0. Код проекта может использоваться в соответствии с лицензией основного проекта.