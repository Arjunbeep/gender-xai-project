# 🧠 Gender Classification with EfficientNet & XAI

This project focuses on building an explainable gender classification model using an EfficientNet-B0 backbone with an integrated attention mechanism. It also incorporates multiple Explainable AI (XAI) techniques—**Grad-CAM**, **Saliency Maps**, and **LIME**—to visualize and understand the model's predictions.

---

## 📁 Dataset

- **Name**: Gender-XAI (based on Microsoft COCO)
- **Size**: 5,000 images (2,500 male / 2,500 female)
- **Annotations**: 
  - Class labels: `male`, `female`
  - Pixel-wise attention masks: factual and counterfactual (for a subset)

---

## 🧰 Model

- **Backbone**: `EfficientNet-B0` (via `timm`)
- **Custom Additions**:
  - Attention map input (7x7)
  - Fusion of visual + attention features before final classification
- **Loss Function**: Weighted CrossEntropyLoss to handle class imbalance
- **Evaluation**:
  - Overall Accuracy: 82.74%
  - Male Accuracy: 80.40%
  - Female Accuracy: 85.08%

---

## 🔍 XAI Techniques Used

| Method     | Purpose                                | Output Type              |
|------------|----------------------------------------|---------------------------|
| Grad-CAM   | Localizes important regions for decision | Heatmap over image       |
| Saliency   | Highlights pixel-level influence        | Gradient-based overlay    |
| LIME       | Superpixel explanation of predictions  | Interpretable segmentation|

---

## 📊 Results

- High performance on both classes with balanced precision and recall
- XAI visualizations confirmed the model focuses on facial and contextual cues
- LIME and Grad-CAM provided human-interpretable justifications for predictions

---

## 🚀 Usage

### 🖼️ Predict from Image

```bash
python predict.py --image_path path/to/image.jpg
