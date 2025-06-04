# MobileNetV2 Dierenherkenningsmodel – 16 mei 2025

Dit model is getraind op 9 dierklassen met behulp van MobileNetV2 als basisarchitectuur. Het model is bedoeld voor eenvoudige en snelle inferentie op wildlife-afbeeldingen, bijvoorbeeld voor gebruik op embedded devices of in mobiele toepassingen.

## 📁 Modelinformatie

- **Modeltype:** MobileNetV2
- **Aantal klassen:** 9
- **Bestand:** `custom_mobilenet_mode_workingl.h5`
- **Trainingsdatum:** 16-05-2025
- **Afbeeldingsformaat:** 224x224 RGB
- **Output:** Klassevoorspelling met waarschijnlijkheden

## 🧠 Herkende dierklassen

De klassen zijn automatisch afgeleid uit de mapstructuur van de trainingsdataset. Dit waren de 9 dierklassen op het moment van training (bijvoorbeeld):

- badger
- mouflon
- fallow_deer
- fox
- hare
- wild_boar
- rabbit
- red_deer
- roe_deer

Dit model is getraind op deze klassen en moet dus ook met deze hoeveelheid klassen in de dataset gebruikt worden