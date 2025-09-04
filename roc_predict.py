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

print("=== 高度な特徴量エンジニアリング開始 ===")
clean_df = shuffle(clean_data(df_raw.copy()), random_state=42).reset_index()
clean_df = feature_engineering(clean_df)
clean_df = feature_group_diff(clean_df)
clean_df = clean_df.drop(columns=['CustomerID'])

feature_df = clean_df.drop(columns=['Churn'])
target_df = clean_df['Churn']

X_train, X_test, y_train, y_test = train_test_split(feature_df, target_df, test_size=0.2, random_state=42, stratify=target_df)

# ロジスティック回帰用に標準化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("特徴量数:", feature_df.shape[1])
print("訓練データサイズ:", X_train.shape)

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
    train_data = lgb.Dataset(X_train, label=y_train)
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
    train_data = lgb.Dataset(X_train, label=y_train)
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
rf_optimized.fit(X_train, y_train)

# ロジスティック回帰
lr_model = LogisticRegression(
    C=1.0,
    penalty='l2',
    class_weight='balanced',
    max_iter=1000,
    random_state=42
)
lr_model.fit(X_train_scaled, y_train)

# 結果をまとめて比較
models_comparison = {}
models_comparison['LightGBM'] = lgb_seed_avg_pred
models_comparison['RandomForest'] = rf_optimized.predict_proba(X_test)[:, 1]
models_comparison['LogisticRegression'] = lr_model.predict_proba(X_test_scaled)[:, 1]

# ROC曲線の描画
plt.figure(figsize=(15, 10))
colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
linestyles = ['-', '--', '-.', ':', '-', '--']
print("\n=== モデル別性能 ===")
auc_scores = {}
for i, (name, y_prob) in enumerate(models_comparison.items()):
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_score = auc(fpr, tpr)
    auc_scores[name] = auc_score
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
plt.title('Model Comparison', fontsize=14, fontweight='bold')
plt.legend(loc="lower right", fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
#plt.savefig('/home/dmainf/advanced_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

best_model_name = max(auc_scores.items(), key=lambda x: x[1])[0]
best_auc = max(auc_scores.values())
print(f"\n=== 最高性能モデル ===")
print(f"モデル: {best_model_name}")
print(f"AUC: {best_auc:.4f}")

# 特徴量重要度
print("\n=== 特徴量重要度 (最高性能モデル) ===")
if best_model_name.startswith('LightGBM'):
    best_model = lgb_models[0]  # 最初のシードのモデルを使用
    feature_importance = best_model.feature_importance(importance_type='gain')
    feature_names = feature_df.columns
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    print("上位10特徴量重要度 (LightGBM):")
    print(importance_df.head(10))
elif best_model_name.startswith('RandomForest'):
    best_model = rf_optimized
    feature_importance = best_model.feature_importances_
    feature_names = feature_df.columns
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    print("上位10特徴量重要度 (RandomForest):")
    print(importance_df.head(10))
elif best_model_name.startswith('LogisticRegression'):
    best_model = lr_model
    # ロジスティック回帰の係数を特徴量重要度として使用
    feature_importance = np.abs(best_model.coef_[0])
    feature_names = feature_df.columns
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    print("上位10特徴量重要度 (LogisticRegression - 係数の絶対値):")
    print(importance_df.head(10))