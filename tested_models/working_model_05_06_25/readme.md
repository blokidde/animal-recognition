# MobileNetV2 Dierenherkenningsmodel – Nieuwe versie (juni 2025)

Dit model is getraind met MobileNetV2 op een uitgebreidere dataset van 14 Europese diersoorten. Het model is bedoeld voor classificatie van wildlife-afbeeldingen met een hoge nauwkeurigheid, geschikt voor toepassingen in monitoring, natuurbeheer of embedded systemen.

## 📁 Modelinformatie

- **Modeltype:** MobileNetV2
- **Aantal klassen:** 14
- **Trainingsdatum:** juni 2025
- **Bestand:** `custom_mobilenet_model.h5`
- **Afbeeldingsformaat:** 224×224 RGB
- **Activatiefunctie laatste laag:** Softmax
- **Output:** Waarschijnlijkheidsverdeling over 14 klassen

## 🧠 Herkende dierklassen

- badger  
- beaver  
- fallow_deer  
- fox  
- hare  
- lynx  
- mouflon  
- pheasant  
- rabbit  
- raccoon  
- red_deer  
- roe_deer  
- wild_boar  
- wolf

## 📊 Prestaties

- **Val accuracy:** ±93%
- **Modelgrootte:** ~20 MB
- **Trained using:** `train_animal_classifier.py` met `ImageDataGenerator`
- **Input normalisatie:** pixelwaarden geschaald tussen 0–1 (`rescale=1./255`)

## 📦 Gebruik

Het model accepteert een enkele afbeelding van formaat (224, 224, 3) en retourneert een vector met 14 waarschijnlijkheden. De hoogste waarde bepaalt de voorspelde klasse.