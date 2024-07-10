import json
import os
from pprint import pprint
import matplotlib.pyplot as plt

# Load the training information
dir_loc = './'
json_files = [f for f in os.listdir(dir_loc) if f.endswith('.json') and f.startswith('cp')]

info = {}
model_series = {
    "DueHeadNet_FMCE" : [],
    "DueHeadNet_NFMCE" : [],
    "Se-ResNet": []
}

for json_file in json_files:
    with open(f'{dir_loc}/{json_file}', 'r') as f:
        cp_n = json_file.split('.')[0]
        info[cp_n] = json.load(f)

for cp_n in info.keys():
    for model in info[cp_n].keys():
        if model.startswith("DueHeadNet"):
            if "NFMCE" in model:
                temp = info[cp_n][model]
                temp["model_name"] = model
                model_series["DueHeadNet_NFMCE"].append(temp)
            else:
                temp = info[cp_n][model]
                temp["model_name"] = model
                model_series["DueHeadNet_FMCE"].append(temp)
        else:
            temp = info[cp_n][model]
            temp["model_name"] = model
            model_series["Se-ResNet"].append(temp)

# Save aggregated data
with open(f'./aggregated_data.json', 'w') as f:
    json.dump(model_series, f)
