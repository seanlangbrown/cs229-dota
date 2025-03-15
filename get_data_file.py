from data_loader import load_file_from_s3
import json

key = "preprocess/yes_tower_kills/1811441857/1811441857_1608151078.json"

data = json.loads(load_file_from_s3(key))

basename = key.split('/')[-1]

with open(f'./data/{basename}', 'w') as f:
    f.write(json.dumps(data))