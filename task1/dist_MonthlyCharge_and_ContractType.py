import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Load data
df_raw = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
df = df_raw.copy()

print("=== Essential Contract-Charges Correlation Analysis ===")

# Numerical conversion
contract_mapping = {'Month-to-month': 0, 'One year': 1, 'Two year': 2}
df['Contract_Numeric'] = df['Contract'].map(contract_mapping)

# Basic correlation
correlation = df['Contract_Numeric'].corr(df['MonthlyCharges'])
print(f"Correlation coefficient: {correlation:.4f}")

# Statistical significance
stat, p_value = stats.spearmanr(df['Contract_Numeric'], df['MonthlyCharges'])
print(f"Spearman correlation: {stat:.4f} (p-value: {p_value:.6f})")

# Essential visualizations
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Essential Contract-Charges Correlation Analysis', fontsize=16, fontweight='bold')

# 1. Scatter plot with regression line
ax1 = axes[0]
colors = {'Month-to-month': 'red', 'One year': 'orange', 'Two year': 'green'}
for contract in df['Contract'].unique():
    subset = df[df['Contract'] == contract]
    ax1.scatter(subset['Contract_Numeric'], subset['MonthlyCharges'], 
               alpha=0.6, label=contract, color=colors[contract], s=20)

# Regression line
z = np.polyfit(df['Contract_Numeric'], df['MonthlyCharges'], 1)
p = np.poly1d(z)
ax1.plot(df['Contract_Numeric'].unique(), p(df['Contract_Numeric'].unique()), 
         "r--", alpha=0.8, linewidth=2)

ax1.set_title(f'1. Scatter Plot with Regression\n(Correlation: {correlation:.3f})', fontweight='bold')
ax1.set_xlabel('Contract Type (0:Month-to-month, 1:One year, 2:Two year)')
ax1.set_ylabel('Monthly Charges ($)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Box plot
ax2 = axes[1]
df.boxplot(column='MonthlyCharges', by='Contract', ax=ax2)
ax2.set_title('2. Monthly Charges Distribution\nby Contract Type', fontweight='bold')
ax2.set_xlabel('Contract Type')
ax2.set_ylabel('Monthly Charges ($)')
ax2.get_figure().suptitle('')  # Remove default boxplot title

# 3. Average charges bar chart
ax3 = axes[2]
mean_charges = df.groupby('Contract')['MonthlyCharges'].mean()
bars = ax3.bar(mean_charges.index, mean_charges.values, 
               color=['red', 'orange', 'green'], alpha=0.7)
ax3.set_title('3. Average Monthly Charges\nby Contract Type', fontweight='bold')
ax3.set_ylabel('Average Monthly Charges ($)')
ax3.set_xlabel('Contract Type')

# Display values on bars
for bar, value in zip(bars, mean_charges.values):
    ax3.text(bar.get_x() + bar.get_width()/2., value + 1,
             f'${value:.2f}', ha='center', va='bottom', fontweight='bold')

plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)

plt.tight_layout()
plt.show()

# Key insights
print("\n=== Key Insights ===")

# Correlation strength interpretation
if abs(correlation) < 0.1:
    strength = "very weak"
elif abs(correlation) < 0.3:
    strength = "weak"
elif abs(correlation) < 0.5:
    strength = "moderate"
elif abs(correlation) < 0.7:
    strength = "strong"
else:
    strength = "very strong"

print(f"1. Correlation strength: {strength} ({correlation:.3f})")
print(f"2. Statistical significance: p-value = {p_value:.6f}")

# Charge differences
contracts = ['Month-to-month', 'One year', 'Two year']
charges_by_contract = {}
for contract in contracts:
    charges_by_contract[contract] = df[df['Contract'] == contract]['MonthlyCharges'].mean()

print(f"3. Average charges:")
for contract in contracts:
    print(f"   {contract}: ${charges_by_contract[contract]:.2f}")

max_diff = max(charges_by_contract.values()) - min(charges_by_contract.values())
print(f"4. Maximum difference: ${max_diff:.2f}")
print(f"5. Pattern: Longer contracts tend to have lower average charges")