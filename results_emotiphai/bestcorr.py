
import pandas as pd

# Substitua 'seu_arquivo.xlsx' pelo caminho do seu arquivo Excel
data = pd.read_excel('PIC.xlsx', engine='openpyxl')
correlation_matrix = data.corr()

target_correlations = correlation_matrix['ann_record'].abs()
best_features = target_correlations.sort_values(ascending=False).index
print(best_features)
import seaborn as sns
import matplotlib.pyplot as plt

sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.show()
plt.savefig('correlations.png')
