import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df_raw = pd.read_csv('../WA_Fn-UseC_-Telco-Customer-Churn.csv')
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

# Graph 1: Contract type vs churn count
plt.figure(figsize=(8, 6))
contract_counts = churned_customers['Contract'].value_counts()
bars1 = plt.bar(contract_counts.index, contract_counts.values, color=['red', 'orange', 'lightcoral'])
plt.title('Contract Type vs Churned Customers', fontsize=15, fontweight='bold', pad=20)
plt.ylabel('Number of Churned Customers')
plt.xlabel('Contract Type')

for bar in bars1:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 10,
             f'{int(height):,}', ha='center', va='bottom', fontweight='bold')

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Graph 2: Charge level vs churn count
plt.figure(figsize=(10, 6))
charge_counts = churned_customers['Charge_Level'].value_counts().reindex(['Very Low\n(<$30)', 'Low\n($30-50)', 'Medium\n($50-70)', 
                                                                         'High\n($70-90)', 'Very High\n(>$90)'])
bars2 = plt.bar(range(len(charge_counts)), charge_counts.values,
                color=['lightblue', 'skyblue', 'gold', 'orange', 'red'])
plt.title('Monthly Charge Level vs Churned Customers', fontsize=15, fontweight='bold', pad=20)
plt.ylabel('Number of Churned Customers')
plt.xlabel('Monthly Charge Level')
plt.xticks(range(len(charge_counts)), charge_counts.index, rotation=45)

for i, bar in enumerate(bars2):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 5,
             f'{int(height):,}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.show()

# Graph 3: Contract × charge level heatmap
plt.figure(figsize=(12, 6))
pivot_churned = churned_customers.pivot_table(values='Churn_Binary', 
                                            index='Contract', 
                                            columns='Charge_Level', 
                                            aggfunc='count', 
                                            fill_value=0)
sns.heatmap(pivot_churned, annot=True, fmt='d', cmap='Reds', 
            cbar_kws={'label': 'Number of Churned Customers'})
plt.title('Contract Type × Charge Level\nChurned Customers Heatmap', fontsize=15, fontweight='bold', pad=20)
plt.ylabel('Contract Type')
plt.xlabel('Monthly Charge Level')

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