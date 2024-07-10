import json
import pandas as pd

# Load data from the provided JSON file
file_path = './aggregated_data.json'
with open(file_path, 'r') as file:
    data = json.load(file)

# Convert JSON data into a DataFrame
models = []
for key, value in data.items():
    for model_info in value:
        model_info['Series'] = key
        models.append(model_info)

df = pd.DataFrame(models)

import matplotlib.pyplot as plt
import seaborn as sns

# Plotting accuracy vs parameters for different series of models
plt.figure(figsize=(12, 8))
sns.scatterplot(data=df, x='parms', y='best_acc', hue='Series', style='model_name', s=250)
plt.title('Comparison of Model Accuracy vs Parameters(CIFAR-10)')
plt.xlabel('Number of Parameters')
plt.ylabel('Best Accuracy (%)')
plt.legend(title='Model Series', loc='lower right')  # Changed legend location to lower right
plt.grid(True)
plt.savefig('model_comparison.png')


plt.figure(figsize=(12, 8))

# Plot each series with lines connecting the points and larger markers
for series, group_data in df.groupby('Series'):
    sorted_data = group_data.sort_values('parms')
    plt.plot(sorted_data['parms'], sorted_data['best_acc'], label=series, marker='o', markersize=12)  # Increased marker size
    for i, txt in enumerate(sorted_data['model_name']):
        # Adjust text position to avoid overlap
        plt.annotate(txt, (sorted_data['parms'].iloc[i], sorted_data['best_acc'].iloc[i] + 0.15))

plt.title('Comparison of Model Accuracy vs Parameters with Series Connection')
plt.xlabel('Number of Parameters')
plt.ylabel('Best Accuracy (%)')
plt.legend(title='Model Series', loc='lower right')
plt.grid(True)
plt.savefig('model_comparison_series.png')