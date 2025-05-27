import os

def rename_files_by_category(folder_path, category_name):
    """
    Verandert de namen van alle bestanden in een specifieke map naar een gewenst patroon,
    bijvoorbeeld 'fox_image_1', 'fox_image_2', enz.

    Parameters:
    - folder_path: Het pad naar de map met de bestanden.
    - category_name: De naam van de categorie (bijvoorbeeld "fox").
    """
    if not os.path.exists(folder_path):
        print(f"De map '{folder_path}' bestaat niet.")
        return

    # Haal alle bestanden in de map op
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

    for index, file in enumerate(files, start=1):
        # Bepaal de extensie van het bestand
        file_extension = os.path.splitext(file)[1]  # Geeft de extensie zoals ".jpg"
        
        # Nieuwe bestandsnaam met index
        new_filename = f"{category_name}_image_{index}{file_extension}"
        
        # Oude en nieuwe volledige paden
        old_file_path = os.path.join(folder_path, file)
        new_file_path = os.path.join(folder_path, new_filename)
        
        # Hernoem het bestand
        os.rename(old_file_path, new_file_path)
        print(f"'{file}' hernoemd naar '{new_filename}'")
    
    print(f"Alle bestanden in '{folder_path}' zijn hernoemd.")

# Gebruik de functie
folder_path = "/home/jurriaan/animalrec/animal_photos/simple_images/wild boar"  # Vervang dit door het pad naar de map met foto's van de vos
category_name = "wild_boar"  # De gewenste categorie (bijvoorbeeld "fox")
rename_files_by_category(folder_path, category_name)
