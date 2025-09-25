# MobileNetV2 Dierenherkenningsmodel – Augustus 2025

Dit model is getraind met MobileNetV2 op een dataset van **20 Europese diersoorten**.  
Het is bedoeld voor classificatie van wildlife-afbeeldingen en kan worden ingezet in monitoring, natuurbeheer of embedded systemen.

## 📁 Modelinformatie

- **Modeltype:** MobileNetV2 (fine-tuned op ImageNet-gewichten)  
- **Aantal klassen:** 20  
- **Trainingsdatum:** 25 augustus 2025  
- **Bestand:** `custom_mobilenet_model.h5`  
- **Afbeeldingsformaat:** 224×224 RGB  
- **Output:** Waarschijnlijkheidsverdeling over 20 klassen  

## 🧠 Herkende dierklassen

badger, beaver, brown_bear, european_polecat, fallow_deer, fox, hare, lynx, mallard, mouflon, pheasant, pine_marten, rabbit, raccoon, raccoon_dog, red_deer, roe_deer, stone_marten, wild_boar, wolf

## 📊 Prestaties

- **Train accuracy:** ~90%  
- **Val accuracy:** ~76%  
- **Test accuracy:** ~79%  
- **Modelgrootte:** ~20 MB  

### 🔎 Classification Report (samenvatting)

- **Precision:** 0.79  
- **Recall:** 0.79  
- **F1-score:** 0.78  
- **Accuracy:** 0.79  

### 📌 Observaties

- **Sterk:** brown_bear, pheasant, mallard, fox  
- **Moeilijker:** badger, hare, stone_marten, wild_boar  

### 📉 Confusion Matrix

De confusion matrix laat zien waar de meeste verwarringen optreden, onder andere tussen **hare ↔ rabbit**, **stone_marten ↔ pine_marten**, en **wild_boar ↔ herten**.

| Class            | Precision | Recall | F1-score | Support |
|------------------|-----------|--------|----------|---------|
| badger           | 0.56      | 0.32   | 0.41     | 44      |
| beaver           | 0.73      | 0.82   | 0.77     | 33      |
| brown_bear       | 0.91      | 0.96   | 0.93     | 45      |
| european_polecat | 0.68      | 0.72   | 0.70     | 50      |
| fallow_deer      | 0.80      | 0.74   | 0.77     | 38      |
| fox              | 1.00      | 0.86   | 0.93     | 36      |
| hare             | 0.58      | 0.72   | 0.64     | 36      |
| lynx             | 0.80      | 0.94   | 0.86     | 50      |
| mallard          | 0.95      | 0.95   | 0.95     | 41      |
| mouflon          | 0.88      | 0.80   | 0.84     | 45      |
| pheasant         | 0.91      | 0.98   | 0.94     | 41      |
| pine_marten      | 0.76      | 0.76   | 0.76     | 45      |
| rabbit           | 0.73      | 0.71   | 0.72     | 49      |
| raccoon          | 0.85      | 0.74   | 0.79     | 38      |
| raccoon_dog      | 0.78      | 0.69   | 0.74     | 36      |
| red_deer         | 0.72      | 0.87   | 0.79     | 39      |
| roe_deer         | 0.91      | 0.81   | 0.85     | 36      |
| stone_marten     | 0.75      | 0.56   | 0.64     | 32      |
| wild_boar        | 0.57      | 0.92   | 0.71     | 25      |
| wolf             | 0.89      | 0.89   | 0.89     | 47      |

**Accuracy:** 0.79 (806 samples)  
**Macro avg:** Precision 0.79 – Recall 0.79 – F1 0.78  
**Weighted avg:** Precision 0.79 – Recall 0.79 – F1 0.78
