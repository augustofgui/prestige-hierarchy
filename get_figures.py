import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)
institution_data = pd.read_csv("institution_df.csv")

def gini_coefficient(data):
    data = np.sort(data) 
    n = len(data)
    cumulative = np.cumsum(data)
    relative_cumulative = cumulative / cumulative[-1]  
    lorenz_curve_area = np.sum(relative_cumulative) / n 
    return 1 - 2 * lorenz_curve_area 

gini_produced = gini_coefficient(institution_data['out_strength'])
gini_hired = gini_coefficient(institution_data['in_strength'])

print(f"Gini Coefficient for Faculty Produced: {gini_produced:.3f}")
print(f"Gini Coefficient for Faculty Hired: {gini_hired:.3f}")

def plot_lorenz_curve(data, title):
    data = np.sort(data)
    n = len(data)
    cumulative = np.cumsum(data) / np.sum(data) 
    x = np.linspace(0, 1, n)
    plt.figure(figsize=(6, 6))
    plt.plot(x, cumulative, label="Lorenz Curve", color="blue")
    plt.plot([0, 1], [0, 1], linestyle="--", color="red", label="Perfect Equality")
    plt.fill_between(x, x, cumulative, color="lightblue", alpha=0.5)
    plt.xlabel("Fraction of Institutions")
    plt.ylabel("Fraction of Faculty Produced/Hired")
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.show()
    plt.savefig(f"{title}.pdf")

plot_lorenz_curve(institution_data['out_strength'], "produced")
plot_lorenz_curve(institution_data['in_strength'], "hired")