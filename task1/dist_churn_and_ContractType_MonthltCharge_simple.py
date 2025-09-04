import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df_raw = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
df = df_raw.copy()
df['Churn_Binary'] = (df['Churn'] == 'Yes').astype(int)

# Extract churned customers only
churned_customers = df[df['Churn_Binary'] == 1]

print(f"=== Essential Churn Distribution Analysis ===")
print(f"Total churned customers: {len(churned_customers):,} customers")

# Create charge levels
df['Charge_Level'] = pd.cut(df['MonthlyCharges'],
                           bins=[0, 30, 50, 70, 90, 120],
                           labels=['Very Low\n(<$30)', 'Low\n($30-50)', 'Medium\n($50-70)',
                                  'High\n($70-90)', 'Very High\n(>$90)'])

churned_customers['Charge_Level'] = pd.cut(churned_customers['MonthlyCharges'],
                                          bins=[0, 30, 50, 70, 90, 120],
                                          labels=['Very Low\n(<$30)', 'Low\n($30-50)', 'Medium\n($50-70)',
                                                 'High\n($70-90)', 'Very High\n(>$90)'])

# Essential visualizations
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Essential Churn Distribution Analysis', fontsize=16, fontweight='bold')

# 1. Contract type vs churn count
ax1 = axes[0]
contract_counts = churned_customers['Contract'].value_counts()
bars1 = ax1.bar(contract_counts.index, contract_counts.values, color=['red', 'orange', 'lightcoral'])
ax1.set_title('Contract Type vs Churned Customers', fontsize=14, fontweight='bold')
ax1.set_ylabel('Number of Churned Customers')
ax1.set_xlabel('Contract Type')

for bar in bars1:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 10,
             f'{int(height):,}', ha='center', va='bottom', fontweight='bold')

plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

# 2. Charge level vs churn count
ax2 = axes[1]
charge_counts = churned_customers['Charge_Level'].value_counts().reindex(['Very Low\n(<$30)', 'Low\n($30-50)', 'Medium\n($50-70)', 
                                                                         'High\n($70-90)', 'Very High\n(>$90)'])
bars2 = ax2.bar(range(len(charge_counts)), charge_counts.values,
                color=['lightblue', 'skyblue', 'gold', 'orange', 'red'])
ax2.set_title('Monthly Charge Level vs Churned Customers', fontsize=14, fontweight='bold')
ax2.set_ylabel('Number of Churned Customers')
ax2.set_xlabel('Monthly Charge Level')
ax2.set_xticks(range(len(charge_counts)))
ax2.set_xticklabels(charge_counts.index, rotation=45)

for i, bar in enumerate(bars2):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 5,
             f'{int(height):,}', ha='center', va='bottom', fontweight='bold')

# 3. Contract × charge level heatmap
ax3 = axes[2]
pivot_churned = churned_customers.pivot_table(values='Churn_Binary', 
                                            index='Contract', 
                                            columns='Charge_Level', 
                                            aggfunc='count', 
                                            fill_value=0)
sns.heatmap(pivot_churned, annot=True, fmt='d', cmap='Reds', ax=ax3, 
            cbar_kws={'label': 'Number of Churned Customers'})
ax3.set_title('Contract Type × Charge Level\nChurned Customers Heatmap', fontsize=14, fontweight='bold')
ax3.set_ylabel('Contract Type')
ax3.set_xlabel('Monthly Charge Level')

plt.tight_layout()
plt.show()

# Key insights summary
print("\n=== Key Insights ===")
print(f"1. Contract type with most churned customers: {contract_counts.idxmax()} ({contract_counts.max():,} customers)")
print(f"2. Charge level with most churned customers: {charge_counts.idxmax().replace(chr(10), ' ')} ({charge_counts.max():,} customers)")

# Most problematic segment
max_segment = pivot_churned.stack().idxmax()
max_count = pivot_churned.stack().max()
print(f"3. Highest-risk segment: {max_segment[0]} × {max_segment[1].replace(chr(10), ' ')} ({max_count:,} customers)")