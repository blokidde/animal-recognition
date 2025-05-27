# Dierenherkenning met Machine Learning

Dit project stelt je in staat om automatisch dierafbeeldingen te downloaden, duplicaten te verwijderen, de datasetbalans te controleren en een neuraal netwerk te trainen met MobileNetV2 of EfficientNetB0.

## Afbeeldingen downloaden
Gebruik drd.py om afbeeldingen van dieren te downloaden

```bash
python3 drd.py --num 100 --searchterm "red fox" --directory ./animal_photos/simple_images/red_fox
```

Zorg dat je een aparte submap gebruikt voor elke diersoort.

## Duplicaten verwijderen
Gebruik duplicates_check.py om vergelijkbare afbeeldingen te detecteren en te verplaatsen:

```bash
python duplicates_check.py
Gebruikt perceptuele hashing
```

Duplicaten worden verplaatst naar de submap duplicates.

## Datasetbalans controleren
Gebruik balancing_check.py om te zien hoeveel afbeeldingen er per klasse zijn:

```bash
python balancing_check.py
```

Dit helpt je om onevenwichtige datasets te herkennen voordat je gaat trainen.

## Bestanden hernoemen per categorie
Gebruik renameimg.py om alle afbeeldingsbestanden in een specifieke map automatisch te hernoemen naar een consistent formaat: <categorie>_image_<nummer>.<extensie>. Dit helpt om je dataset overzichtelijk en bruikbaar te maken voor training.

```bash
python3 renameimg.py --folder "/pad/naar/foto's/wild_boar" --category "wild_boar"
```

Bestanden in de opgegeven map worden bijvoorbeeld als volgt hernoemd:

```
IMG_1234.jpg     -> wild_boar_image_1.jpg
DSC_5678.jpeg    -> wild_boar_image_2.jpeg
```

## Model trainen
Gebruik train_animal_classifier.py om een model te trainen:

```bash
python train_animal_classifier.py \
  --data_dir ./animal_photos/simple_images \
  --model efficientnet \
  --img_size 224 \
  --batch_size 16 \
  --epochs 30
```
--model kan mobilenet of efficientnet zijn.

Het getrainde model wordt opgeslagen als custom_<model>_model.h5

## Output
Trainingslogs worden opgeslagen in de logs/ map (geschikt voor TensorBoard)

Het uiteindelijke .h5 modelbestand staat in de root van het project

## Opmerkingen
Zorg voor een goede balans in de dataset (gelijke verdeling van klassen)

Verwijder ongeldige of irrelevante afbeeldingen handmatig voor betere resultaten