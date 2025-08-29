import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Load data
df_raw = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
df = df_raw.copy()

# Data preparation
df['Churn_Binary'] = (df['Churn'] == 'Yes').astype(int)
df['Long_Contract'] = (df['Contract'] != 'Month-to-month').astype(int)

# Standardization
scaler = StandardScaler()
df['MonthlyCharges_Std'] = scaler.fit_transform(df[['MonthlyCharges']])

# Create binary variables
for col in ['Partner', 'Dependents', 'PhoneService']:
    df[col + '_Binary'] = (df[col] == 'Yes').astype(int)

df['TotalCharges_Numeric'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges_Numeric'] = df['TotalCharges_Numeric'].fillna(df['TotalCharges_Numeric'].median())

print("=== Essential Causal Inference Analysis ===")

# Confounding factors
confounders = ['SeniorCitizen', 'Partner_Binary', 'Dependents_Binary', 
               'tenure', 'PhoneService_Binary', 'TotalCharges_Numeric']

# Calculate effects
# Unadjusted effects
model_contract_unadj = LogisticRegression(random_state=42)
model_contract_unadj.fit(df[['Long_Contract']], df['Churn_Binary'])
contract_effect_unadj = model_contract_unadj.coef_[0][0]

model_charges_unadj = LogisticRegression(random_state=42)
model_charges_unadj.fit(df[['MonthlyCharges_Std']], df['Churn_Binary'])
charges_effect_unadj = model_charges_unadj.coef_[0][0]

# Adjusted effects
X_adjusted = df[['Long_Contract', 'MonthlyCharges_Std'] + confounders]
model_adjusted = LogisticRegression(random_state=42)
model_adjusted.fit(X_adjusted, df['Churn_Binary'])
contract_effect_adj = model_adjusted.coef_[0][0]
charges_effect_adj = model_adjusted.coef_[0][1]

print(f"Contract type effect - Unadjusted: {contract_effect_unadj:.3f}, Adjusted: {contract_effect_adj:.3f}")
print(f"Monthly charges effect - Unadjusted: {charges_effect_unadj:.3f}, Adjusted: {charges_effect_adj:.3f}")

# Essential visualizations
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Essential Causal Inference Analysis', fontsize=16, fontweight='bold')

# 1. Unadjusted vs adjusted effects comparison
ax1 = axes[0, 0]
effects = ['Contract\n(Unadj)', 'Contract\n(Adj)', 'Charges\n(Unadj)', 'Charges\n(Adj)']
values = [contract_effect_unadj, contract_effect_adj, charges_effect_unadj, charges_effect_adj]
colors = ['lightcoral', 'red', 'lightblue', 'blue']

bars = ax1.bar(effects, values, color=colors, alpha=0.7)
ax1.set_title('1. Unadjusted vs Adjusted Effects\n(Shows Bias)', fontweight='bold')
ax1.set_ylabel('Logistic Regression Coefficient')
ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
ax1.grid(True, alpha=0.3)

for bar, value in zip(bars, values):
    ax1.text(bar.get_x() + bar.get_width()/2., value + 0.05 if value > 0 else value - 0.1,
             f'{value:.3f}', ha='center', va='bottom' if value > 0 else 'top', fontweight='bold')

# 2. Average charges by contract type (source of spurious correlation)
ax2 = axes[0, 1]
contract_charges = df.groupby('Contract')['MonthlyCharges'].mean()
bars = ax2.bar(range(len(contract_charges)), contract_charges.values, 
               color=['red', 'orange', 'green'], alpha=0.7)
ax2.set_title('2. Average Monthly Charges by Contract\n(Source of Spurious Correlation)', fontweight='bold')
ax2.set_ylabel('Average Monthly Charges ($)')
ax2.set_xticks(range(len(contract_charges)))
ax2.set_xticklabels(contract_charges.index, rotation=45)

for bar, value in zip(bars, contract_charges.values):
    ax2.text(bar.get_x() + bar.get_width()/2., value + 1,
             f'${value:.1f}', ha='center', va='bottom', fontweight='bold')

# 3. Correlation vs causal effects comparison
ax3 = axes[1, 0]
correlations = [df['Long_Contract'].corr(df['Churn_Binary']), 
                df['MonthlyCharges'].corr(df['Churn_Binary'])]
causal_effects = [contract_effect_adj, charges_effect_adj]

x_pos = np.arange(2)
width = 0.35

ax3.bar(x_pos - width/2, np.abs(correlations), width, label='Correlation', alpha=0.7, color='orange')
ax3.bar(x_pos + width/2, np.abs(causal_effects), width, label='Causal Effect', alpha=0.7, color='blue')

ax3.set_title('3. Correlation vs Causal Effects\n(Key Insight)', fontweight='bold')
ax3.set_ylabel('Effect Size (Absolute Value)')
ax3.set_xticks(x_pos)
ax3.set_xticklabels(['Contract Type', 'Monthly Charges'])
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. Spurious correlation structure diagram
ax4 = axes[1, 1]
ax4.text(0.1, 0.8, 'Contract Type', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"), 
         ha='center', va='center', fontsize=12, fontweight='bold')
ax4.text(0.5, 0.5, 'Confounders\n(tenure, services)', 
         bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"), 
         ha='center', va='center', fontsize=10)
ax4.text(0.9, 0.8, 'Monthly Charges', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"), 
         ha='center', va='center', fontsize=12, fontweight='bold')
ax4.text(0.5, 0.2, 'Churn', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral"), 
         ha='center', va='center', fontsize=12, fontweight='bold')

# Arrows
ax4.arrow(0.2, 0.8, 0.6, 0, head_width=0.03, head_length=0.03, fc='blue', ec='blue')
ax4.arrow(0.1, 0.7, 0.3, -0.4, head_width=0.03, head_length=0.03, fc='blue', ec='blue')
ax4.arrow(0.9, 0.7, -0.3, -0.4, head_width=0.03, head_length=0.03, fc='green', ec='green')
ax4.arrow(0.4, 0.4, 0, -0.1, head_width=0.03, head_length=0.02, fc='gray', ec='gray')

ax4.set_xlim(0, 1)
ax4.set_ylim(0, 1)
ax4.axis('off')
ax4.set_title('4. Spurious Correlation Structure\n(Explains the Mechanism)', fontweight='bold')

plt.tight_layout()
plt.show()

# Key insights
print("\n=== Key Insights ===")
print(f"1. Contract type bias: {contract_effect_unadj - contract_effect_adj:.3f} (overestimated)")
print(f"2. Monthly charges bias: {charges_effect_unadj - charges_effect_adj:.3f} (underestimated)")
print(f"3. True effect is {charges_effect_adj/charges_effect_unadj:.1f}x larger for charges")

if abs(contract_effect_adj) > abs(charges_effect_adj):
    print("4. Contract type has stronger causal effect")
else:
    print("4. Monthly charges have stronger causal effect")
print("5. Simple correlation analysis is misleading due to spurious correlations")