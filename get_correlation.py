import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def plot_filtered_correlation(df, method='both', p_value_threshold=0.001, figsize=(14, 6)):
    valid_methods = ['pearson', 'spearman', 'both']
    if method not in valid_methods:
        raise ValueError(f"Method must be one of {valid_methods}, got {method} instead.")
    
    methods_to_use = ['pearson', 'spearman'] if method == 'both' else [method]
    
    if method == 'both':
        fig, axes = plt.subplots(1, 2, figsize=figsize)
    else:
        fig, axes = plt.subplots(1, 1, figsize=(figsize[0]//2, figsize[1]))
        axes = [axes]
    
    for i, corr_method in enumerate(methods_to_use):
        correlation = df.corr(method=corr_method)
        
        p_values = pd.DataFrame(np.zeros_like(correlation), 
                              index=correlation.index, 
                              columns=correlation.columns)
        for i_idx in range(len(correlation.columns)):
            for j_idx in range(len(correlation.columns)):
                if i_idx != j_idx:
                    x = df.iloc[:, i_idx].dropna()
                    y = df.iloc[:, j_idx].dropna()
                    
                    if corr_method == 'pearson':
                        corr_coef, p_value = stats.pearsonr(x, y)
                    else: 
                        corr_coef, p_value = stats.spearmanr(x, y)
                        
                    p_values.iloc[i_idx, j_idx] = p_value
        mask_p_value = p_values > p_value_threshold
        mask_upper = np.triu(np.ones_like(correlation, dtype=bool))
        combined_mask = np.logical_or(mask_upper, mask_p_value)
        sns.heatmap(correlation, annot=True, cmap='coolwarm', vmin=-1, vmax=1, 
                    mask=combined_mask, ax=axes[i], fmt='.2f', 
                    cbar_kws={'label': 'Correlation Coefficient'},
                    annot_kws={'size': 8})
        
        axes[i].set_title(f'{corr_method.capitalize()} Correlation (p < {p_value_threshold})')
    
    plt.tight_layout()
    plt.savefig(f"figures/{method}.pdf")
  
df = pd.read_csv('institution_df.csv').drop(columns=['institution_id', 'institution_acr'])
df["num_professors"] = df["in_strength"]
# df = pd.read_csv('institution_normalized_df.csv')
df.info()

plot_filtered_correlation(df, method='pearson')

plot_filtered_correlation(df, method='spearman')

plot_filtered_correlation(df, method='both')