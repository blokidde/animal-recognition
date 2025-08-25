from google_images_search import GoogleImagesSearch
from dotenv import load_dotenv
import argparse
import os

# load .env file
load_dotenv()

# assign .env aan variables
API_KEY = os.getenv('GCS_API_KEY2')
CX = os.getenv('GCS_CX2')

if not API_KEY or not CX:
    raise ValueError("API key of CX ontbreekt. Zorg dat GCS_API_KEY en GCS_CX in je .env bestand staan.")


# give API key to googleAPI
gis = GoogleImagesSearch(API_KEY, CX)

# Command-line argumenten instellen
parser = argparse.ArgumentParser(description="Download afbeeldingen van dieren.")
parser.add_argument(
    '--num', 
    type=int, 
    required=True, 
    help="Het aantal afbeeldingen dat je wilt downloaden."
)
parser.add_argument(
    '--searchterm',
    type=str,
    required=True,
    help="De zoekterm voor de afbeeldingen."
)
parser.add_argument(
    '--directory',
    type=str,
    required=True,
    help="De zoekterm voor de afbeeldingen."
)

args = parser.parse_args()

# Zoekparameters
search_params = {
    'q': args.searchterm,       # Zoekterm
    'num': args.num,             # Aantal resultaten
    'safe': 'off',         # Schakel veilige zoekinstelling uit
    'fileType': 'jpg',     # Alleen JPG-afbeeldingen
    'imgType': 'photo',    # Alleen foto's
    'imgSize': 'medium',   # Medium formaat afbeeldingen
}

# Map waar de afbeeldingen worden opgeslagen
output_folder = args.directory

# Controleer of de map bestaat, zo niet, maak deze aan
os.makedirs(output_folder, exist_ok=True)

# Voer de zoekopdracht uit en download de afbeeldingen
gis.search(search_params=search_params, path_to_dir=output_folder)
print(f"Afbeeldingen succesvol gedownload naar {output_folder}")
