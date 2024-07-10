import json
import os
import matplotlib.pyplot as plt

# Load the training information
dir_loc = './json_data'
json_files = os.listdir(dir_loc)

info = {}

for json_file in json_files:
    with open(f'{dir_loc}/{json_file}', 'r') as f:
        model_name = json_file.split('.')[0]
        info[model_name] = json.load(f)

# Extracting model names, accuracy, and parameters
models = list(info.keys())
best_acc = [info[model]["best_acc"] for model in models]
parameters = [info[model]["parameters"] for model in models]

# Plotting best accuracy
plt.figure(figsize=(10, 5))
plt.plot(models, best_acc, marker='o', label='Best Accuracy')
plt.title('Best Accuracy of Models')
plt.xlabel('Models')
plt.ylabel('Accuracy (%)')
plt.grid(True)
plt.xticks(rotation=45)
plt.legend()
plt.savefig('./image/best_acc.png')

# Plotting parameters
plt.figure(figsize=(10, 5))
plt.plot(models, parameters, marker='o', color='red', label='Parameters')
plt.title('Parameters of Models')
plt.xlabel('Models')
plt.ylabel('Number of Parameters')
plt.grid(True)
plt.xticks(rotation=45)
plt.legend()
plt.savefig('./image/parameters.png')