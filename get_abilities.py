import requests
import json

abilities_url = 'https://api.opendota.com/api/constants/abilities'
data = requests.get(abilities_url).json()

print(f"found {len(data)} abilities")

with open('opendota_constants/abilities.json', 'w') as f:
    f.write(json.dumps(data))