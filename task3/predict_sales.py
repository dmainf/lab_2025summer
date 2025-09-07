import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
import lightgbm as lgb
import optuna
import warnings
warnings.filterwarnings('ignore')

from lib.prepro import clean_data, one_hot_encoding, merge_df
from lib.feature_eng import feature_engineering, feature_group_diff

df_raw = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
df_after = pd.read_csv('after_data.csv')

df_raw['TotalCharges'] = pd.to_numeric(df_raw['TotalCharges'], errors='coerce')
df_after['TotalCharges'] = pd.to_numeric(df_after['TotalCharges'], errors='coerce')

print("=== Data Summary ===")
print(f"Total customers: {len(df_raw)}")
print(f"before: Total charges sum: ${df_raw['TotalCharges'].sum():,.2f}")
print(f"after:  Total charges sum: ${df_after['TotalCharges'].sum():,.2f}")

before_total = df_raw['TotalCharges'].sum()
after_total = df_after['TotalCharges'].sum()
difference = after_total - before_total
percent_increase = (difference / before_total) * 100

plt.figure(figsize=(10, 6))

categories = ['Before', 'After']
values = [before_total, after_total]
x_pos = [0.3, 1.3]
bars = plt.bar(x_pos, values, color=['#d62728', '#2ca02c'], width=0.6)
plt.xticks(x_pos, categories)

# Y軸の最小値を調整して差を強調
y_min = min(values) * 0.98
y_max = max(values) * 1.02
plt.ylim(y_min, y_max)

plt.title(f'Total Charges: Before vs After\n(Increase: ${difference:,.0f} | +{percent_increase:.1f}%)', 
          fontsize=14, fontweight='bold')
plt.ylabel('Total Charges ($)', fontsize=12)

for i, bar in enumerate(bars):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + (y_max-y_min)*0.01,
             f'${height:,.0f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.ticklabel_format(style='plain', axis='y')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()


"""
    @sales.py 現段階では全体で26.5%の解約率がある *Churn=Yes の Contact=Month-to-month または MonthlyCharges>=65 
  のデータを一定数 Churn=NO にし,TotalCharges を一定数増やす *Contact=Month-to-month のデータを一定数 MonthlyCh
  arges=One year にし TotalChargesを一定数増やす *Churn=Yes の Contract=Two year のデータを一定数
  Churn=No にし TotalCharges を一定数増やす 以上のことをして全体の解約率が20%になるようにうまく調整するように
  @sales.csv を変更して，変更後のデータで TotalCharges の合計出力して
"""