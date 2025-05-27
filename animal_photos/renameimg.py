import os
import argparse

def rename_files_by_category(folder_path, category_name):
    
    # Controleer of de opgegeven map bestaat
    if not os.path.exists(folder_path):
        print(f"De map '{folder_path}' bestaat niet.")
        return

    # Haal alle bestanden op in de map (negeer submappen)
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

    # Loop door de bestanden en hernoem ze één voor één
    for index, file in enumerate(files, start=1):
        # Bepaal de bestandsextensie (zoals .jpg of .png)
        file_extension = os.path.splitext(file)[1]

        # Stel de nieuwe bestandsnaam samen
        new_filename = f"{category_name}_image_{index}{file_extension}"

        # Volledige paden voor oud en nieuw bestand
        old_file_path = os.path.join(folder_path, file)
        new_file_path = os.path.join(folder_path, new_filename)

        # Hernoem het bestand
        os.rename(old_file_path, new_file_path)
        print(f"'{file}' hernoemd naar '{new_filename}'")

    print(f"Alle bestanden in '{folder_path}' zijn hernoemd.")

def main():
    # Command-line argumenten instellen
    parser = argparse.ArgumentParser(description="Hernoem afbeeldingen in een map volgens een categorienaam.")
    parser.add_argument(
        '--folder', '-f', required=True, type=str,
        help="Pad naar de map met afbeeldingen"
    )
    parser.add_argument(
        '--category', '-c', required=True, type=str,
        help="Naam van de categorie, bijvoorbeeld 'wild_boar'"
    )

    # Parse de argumenten
    args = parser.parse_args()

    # Voer de hernoemfunctie uit
    rename_files_by_category(args.folder, args.category)

# Zorg dat het script alleen uitgevoerd wordt als hoofdprogramma
if __name__ == "__main__":
    main()
