import pandas as pd
import matplotlib.pyplot as plt
plt.style.use(['science', 'ieee', 'high-vis'])

df = pd.read_csv('different_net_sizes_exp.csv')
# Generate plot
groups = df.groupby('Model')
# Plot
fig, ax = plt.subplots(figsize=(4, 3))
# ax.margins(0.05)  # Optional, just adds 5% padding to the autoscaling
for name, group in groups:
    ax.scatter(group['Num. Lanes'], group['Time'], marker='o', label=name)
ax.set_xlabel('Num. Lanes in Network')
ax.set_ylabel('Computational Time (ms)')
ax.set_yscale('log')
ax.set_xscale('log')
ax.grid()
ax.legend(frameon=1, facecolor='white', framealpha=0.75, loc=2)
plt.tight_layout()
plt.savefig('problem_size.pdf')
