import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, roc_auc_score, classification_report
from imblearn.over_sampling import SMOTE
from sklearn.utils import shuffle
import lightgbm as lgb
import optuna
import warnings
warnings.filterwarnings('ignore')

from lib.prepro import clean_data, one_hot_encoding, merge_df
from lib.feature_eng import feature_engineering, feature_group_diff

df_raw = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

print("=== 高度な特徴量エンジニアリング開始 ===")
clean_df = shuffle(clean_data(df_raw.copy()), random_state=42).reset_index()
clean_df = feature_engineering(clean_df)
clean_df = feature_group_diff(clean_df)
clean_df = clean_df.drop(columns=['CustomerID'])

feature_df = clean_df.drop(columns=['Churn'])
target_df = clean_df['Churn']

X_train, X_test, y_train, y_test = train_test_split(feature_df, target_df, test_size=0.2, random_state=42, stratify=target_df)
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
print("特徴量数:", feature_df.shape[1])
print("訓練データサイズ (SMOTE後):", X_train_smote.shape)

# Optunaを使ったLightGBMハイパーパラメータ最適化
def objective(trial):
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': trial.suggest_int('num_leaves', 10, 100),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'verbose': -1
    }
    train_data = lgb.Dataset(X_train_smote, label=y_train_smote)
    cv_result = lgb.cv(
        params,
        train_data,
        num_boost_round=1000,
        nfold=3,
        shuffle=True,
        stratified=True,
        callbacks=[lgb.early_stopping(10, verbose=False)],
        return_cvbooster=True
    )
    best_score = max(cv_result['valid auc-mean'])
    best_round = len(cv_result['valid auc-mean'])
    trial.set_user_attr('best_round', best_round)
    return best_score

print("\n=== Optunaによるハイパーパラメータ最適化 ===")
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=20)
print(f"最適なAUC: {study.best_value:.4f}")
print(f"最適なパラメータ: {study.best_params}")
best_num_boost_round = study.best_trial.user_attrs['best_round']
print(f"最適なラウンド数: {best_num_boost_round}")

# 最適化されたLightGBMモデル
best_params = study.best_params
best_params.update({
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'verbose': -1
})

def train_lgb_with_seed(seed):
    params = best_params.copy()
    params['random_state'] = seed
    train_data = lgb.Dataset(X_train_smote, label=y_train_smote)
    model = lgb.train(params, train_data, num_boost_round=best_num_boost_round)
    return model
print("\n=== Seed Averaging実行 ===")
seeds = [42, 123, 456, 789, 999]
lgb_models = []
for seed in seeds:
    model = train_lgb_with_seed(seed)
    lgb_models.append(model)
def predict_with_seed_averaging(models, X):
    predictions = []
    for model in models:
        pred = model.predict(X, num_iteration=model.best_iteration)
        predictions.append(pred)
    return np.mean(predictions, axis=0)
lgb_seed_avg_pred = predict_with_seed_averaging(lgb_models, X_test)

# 最適化RandomForest
rf_optimized = RandomForestClassifier(
    n_estimators=200,
    max_depth=7,
    max_features=0.7,
    min_samples_leaf=3,
    min_samples_split=5,
    class_weight='balanced',
    oob_score=True,
    random_state=42
)
rf_optimized.fit(X_train_smote, y_train_smote)

# SMOTEなしのモデル
def train_lgb_no_smote(seed):
    params = best_params.copy()
    params['random_state'] = seed
    train_data = lgb.Dataset(X_train, label=y_train)
    model = lgb.train(params, train_data, num_boost_round=best_num_boost_round)
    return model
lgb_no_smote_models = []
for seed in seeds:
    model = train_lgb_no_smote(seed)
    lgb_no_smote_models.append(model)
lgb_no_smote_pred = predict_with_seed_averaging(lgb_no_smote_models, X_test)

# RandomForest（SMOTEなし）
rf_no_smote = RandomForestClassifier(
    n_estimators=200,
    max_depth=7,
    max_features=0.7,
    min_samples_leaf=3,
    min_samples_split=5,
    class_weight='balanced',
    oob_score=True,
    random_state=42
)
rf_no_smote.fit(X_train, y_train)

# 結果をまとめて比較
models_comparison = {}
# LightGBM比較
models_comparison['LightGBM_with_SMOTE'] = lgb_seed_avg_pred
models_comparison['LightGBM_without_SMOTE'] = lgb_no_smote_pred
# RandomForest比較
models_comparison['RandomForest_with_SMOTE'] = rf_optimized.predict_proba(X_test)[:, 1]
models_comparison['RandomForest_without_SMOTE'] = rf_no_smote.predict_proba(X_test)[:, 1]

# ROC曲線の描画
plt.figure(figsize=(12, 8))
colors = ['red', 'blue', 'green', 'orange']
linestyles = ['-', '--', '-.', ':']
print("\n=== 最終結果比較 ===")
for i, (name, y_prob) in enumerate(models_comparison.items()):
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_score = auc(fpr, tpr)
    plt.plot(fpr, tpr,
            color=colors[i % len(colors)],
            linestyle=linestyles[i % len(linestyles)],
            linewidth=2,
            label=f'{name} (AUC = {auc_score:.4f})')
    print(f"{name:25s}: AUC = {auc_score:.4f}")
plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random (AUC = 0.5000)')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('SMOTE Comparison: With vs Without SMOTE', fontsize=14, fontweight='bold')
plt.legend(loc="lower right", fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
#plt.savefig('/home/dmainf/advanced_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

best_model_name = max(models_comparison.items(), key=lambda x: roc_auc_score(y_test, x[1]))[0]
best_auc = max([roc_auc_score(y_test, pred) for pred in models_comparison.values()])
print(f"\n=== 最高性能モデル ===")
print(f"モデル: {best_model_name}")
print(f"AUC: {best_auc:.4f}")

print(f"\n=== SMOTE効果比較 ===")
lgb_smote_auc = roc_auc_score(y_test, models_comparison['LightGBM_with_SMOTE'])
lgb_no_smote_auc = roc_auc_score(y_test, models_comparison['LightGBM_without_SMOTE'])
rf_smote_auc = roc_auc_score(y_test, models_comparison['RandomForest_with_SMOTE'])
rf_no_smote_auc = roc_auc_score(y_test, models_comparison['RandomForest_without_SMOTE'])

print(f"LightGBM:")
print(f"  SMOTE使用:   {lgb_smote_auc:.4f}")
print(f"  SMOTE未使用: {lgb_no_smote_auc:.4f}")
print(f"  改善度:      {lgb_smote_auc - lgb_no_smote_auc:+.4f}")
print(f"RandomForest:")
print(f"  SMOTE使用:   {rf_smote_auc:.4f}")
print(f"  SMOTE未使用: {rf_no_smote_auc:.4f}")
print(f"  改善度:      {rf_smote_auc - rf_no_smote_auc:+.4f}")

# 特徴量重要度（LightGBMの場合）
if best_model_name.startswith('LightGBM'):
    # 最適性能モデルを選択
    if 'without_SMOTE' in best_model_name:
        best_model = lgb_no_smote_models[0]  # 最初のシードのモデルを使用
    else:
        best_model = lgb_models[0]  # 最初のシードのモデルを使用
    feature_importance = best_model.feature_importance(importance_type='gain')
    feature_names = feature_df.columns
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    print("\n=== 上位10特徴量重要度 ===")
    print(importance_df.head(10))