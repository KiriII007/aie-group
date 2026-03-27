# HW10-11 – компьютерное зрение в PyTorch: CNN, transfer learning, segmentation

## 1. Кратко: что сделано

- **Часть A:** Flowers102 — 102 класса цветов; сравнивались SimpleCNN без/с аугментациями
  и ResNet18 head-only / partial fine-tune (layer4+fc).
- **Часть B:** OxfordIIITPet — бинарная сегментация «питомец/фон�� с pretrained DeepLabV3-ResNet50.
  Два режима постобработки: V1 (argmax), V2 (порог softmax > 0.5).

## 2. Среда и воспроизводимость

- Python: 3.10+
- torch / torchvision: 2.x / 0.17+
- Устройство: GPU (CUDA) при наличии, иначе CPU
- Seed: 42
- Как запустить: открыть `HW10-11.ipynb` → Run All.

## 3. Данные

### 3.1. Часть A: классификация

- **Датасет:** Flowers102 (`torchvision.datasets.Flowers102`)
- **Разделение:** official split — train (1 020), val (1 020), test (6 149)
- **Базовые transforms (CNN):** `Resize(224)` → `ToTensor` → `Normalize(ImageNet)`
- **Augmentation transforms:** `Resize(256)` → `RandomCrop(224)` → `RandomHorizontalFlip` →
  `RandomRotation(15)` → `ColorJitter` → `ToTensor` → `Normalize(ImageNet)`
- **ResNet transforms:** `Resize(256)` → `CenterCrop(224)` → `ToTensor` → `Normalize(ImageNet)`
- **Комментарий:** Flowers102 содержит 102 класса, но лишь по 10 изображений на класс в train,
  что делает задачу крайне сложной для обучения с нуля и идеально подходит для демонстрации
  transfer learning. Изображения разного размера и масштаба.

### 3.2. Часть B: structured vision

- **Датасет:** OxfordIIITPet (`torchvision.datasets.OxfordIIITPet`)
- **Трек:** segmentation
- **Ground truth:** trimap; foreground = значение 1 (тело питомца), background = 2 и 3
- **Предсказания:** DeepLabV3-ResNet50, pretrained COCO; foreground = cat (8) | dog (12)
- **Комментарий:** OxfordIIITPet содержит фото кошек и собак крупным планом; DeepLabV3,
  обученная на COCO, имеет классы cat и dog, поэтому бинарная сегментация pet/background разумна.

## 4. Часть A: модели и обучение (C1-C4)

| ID | Модель | Augmentation | Что обучается | LR |
|----|--------|-------------|----------------|------|
| C1 | SimpleCNN (4 conv + BN) | нет | всё | 1e-3 |
| C2 | SimpleCNN (4 conv + BN) | да | всё | 1e-3 |
| C3 | ResNet18 pretrained | нет | fc only | 1e-3 |
| C4 | ResNet18 pretrained | нет | layer4 + fc | 1e-4 |

- **Loss:** CrossEntropyLoss
- **Optimizer:** Adam
- **Batch size:** 32
- **Epochs:** 30
- **Критерий выбора лучшей модели:** best val accuracy

## 5. Часть B: постановка задачи и режимы оценки (V1-V2)

### Segmentation track

- **Модель:** DeepLabV3_ResNet50 (pretrained COCO / VOC labels)
- **Foreground:** пиксели, предсказанные как cat (8) или dog (12)
- **V1 (basic):** argmax по 21 классу → бинарная маска cat|dog
- **V2 (alternative):** softmax P(cat)+P(dog) > 0.5 — при пороге бинаризации отсекаются
  неуверенные предсказания, повышая precision за счёт recall
- **mean IoU:** IoU по бинарной маске (pet vs background), усреднённый по изображениям
- **Дополнительные метрики:** pixel_precision, pixel_recall

## 6. Результаты

- Таблица результатов: `./artifacts/runs.csv`
- Лучшая модель: `./artifacts/best_classifier.pt`
- Конфиг лучшей модели: `./artifacts/best_classifier_config.json`
- Кривые лучшего прогона: `./artifacts/figures/classification_curves_best.png`
- Сравнение C1-C4: `./artifacts/figures/classification_compare.png`
- Визуализация аугментаций: `./artifacts/figures/augmentations_preview.png`
- Примеры сегментации: `./artifacts/figures/segmentation_examples.png`
- Метрики сегментации: `./artifacts/figures/segmentation_metrics.png`

**Сводка:**

- Лучший эксперимент части A: **C4 (ResNet18 finetune layer4+fc)**
- Лучшая val_accuracy: **0.9069**
- Test accuracy: **0.8788**
- C2 vs C1 (аугментации): val_accuracy выросла с 0.2520 до 0.2912 (+3.9 п.п.); аугментации помогают, но не компенсируют нехватку данных при 10 изображениях на класс.
- C3/C4 vs C1/C2 (transfer learning): pretrained ResNet18 кратно превосходит SimpleCNN (0.87–0.91 vs 0.25–0.29).
- C4 vs C3 (fine-tune vs head-only): partial fine-tune даёт +4.0 п.п. (0.9069 vs 0.8667), адаптируя layer4 к домену цветов.
- V1 (argmax): mean_IoU = 0.7398, precision = 