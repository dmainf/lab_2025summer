import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def feature_engineering(df):
    # 欠損値を事前に処理
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

    # 1. より詳細なドメイン知識に基づく特徴量
    # 顧客の価値レベル（より細かく分割）
    df['CustomerValueLevel'] = pd.cut(df['TotalCharges'],
                                     bins=[0, 300, 800, 1500, 3000, 6000, np.inf],
                                     labels=[0, 1, 2, 3, 4, 5]).astype(int)

    # 契約期間による詳細セグメント
    df['TenureSegment'] = pd.cut(df['Tenure'],
                                bins=[0, 6, 12, 24, 36, 60, np.inf],
                                labels=[0, 1, 2, 3, 4, 5])
    df['TenureSegment'] = df['TenureSegment'].fillna(0).astype(int)

    # 月額料金レベル（より細分化）
    df['MonthlyChargesLevel'] = pd.cut(df['MonthlyCharges'],
                                      bins=[0, 25, 40, 55, 70, 85, 100, np.inf],
                                      labels=[0, 1, 2, 3, 4, 5, 6])
    df['MonthlyChargesLevel'] = df['MonthlyChargesLevel'].fillna(0).astype(int)

    # 新規顧客リスク（最初の6ヶ月）
    df['IsNewCustomer'] = (df['Tenure'] <= 6).astype(int)
    # 中期顧客（6-24ヶ月）
    df['IsMidTermCustomer'] = ((df['Tenure'] > 6) & (df['Tenure'] <= 24)).astype(int)
    # 長期顧客（24ヶ月以上）
    df['IsLongTermCustomer'] = (df['Tenure'] > 24).astype(int)

    # 高額顧客フラグ
    df['IsHighValueCustomer'] = (df['TotalCharges'] > df['TotalCharges'].quantile(0.8)).astype(int)
    df['IsLowValueCustomer'] = (df['TotalCharges'] < df['TotalCharges'].quantile(0.2)).astype(int)

    # 料金増加率（月額料金が平均より高い場合の期間比較）
    df['MonthlyChargesAboveAvg'] = (df['MonthlyCharges'] > df['MonthlyCharges'].mean()).astype(int)
    df['ExpensiveForTenure'] = (df['MonthlyCharges'] > (df['TotalCharges'] / (df['Tenure'] + 1) * 1.2)).astype(int)

    # 2. 大幅に拡張した比率特徴量（LightGBMで重要な斜めの線）
    df['AvgMonthlyCharges'] = df['TotalCharges'] / (df['Tenure'] + 1)
    
    # サービス数を先に計算
    services_count = (df['PhoneService'] + df['InternetService'] +
                     df['OnlineSecurity'] + df['OnlineBackup'] +
                     df['DeviceProtection'] + df['TechSupport'] +
                     df['StreamingTV'] + df['StreamingMovies'])
    
    df['ChargesPerService'] = df['TotalCharges'] / (services_count + 1)
    df['MonthlyChargesPerService'] = df['MonthlyCharges'] / (services_count + 1)
    
    # より詳細な比率特徴量
    df['ChargesPerTenureMonth'] = df['TotalCharges'] / (df['Tenure'] + 1)
    df['MonthlyToTotalRatio'] = df['MonthlyCharges'] / (df['TotalCharges'] + 1)
    df['TotalChargesPerMonthlyCharges'] = df['TotalCharges'] / (df['MonthlyCharges'] + 1)
    
    # サービス別比率
    streaming_services = df['StreamingTV'] + df['StreamingMovies']
    security_services = df['OnlineSecurity'] + df['OnlineBackup'] + df['DeviceProtection'] + df['TechSupport']
    
    df['StreamingServicesRatio'] = streaming_services / (services_count + 1)
    df['SecurityServicesRatio'] = security_services / (services_count + 1)
    df['BasicServicesRatio'] = (df['PhoneService'] + df['InternetService']) / (services_count + 1)
    
    # 高度な比率特徴量
    df['ChargesEfficiency'] = df['TotalCharges'] / ((df['Tenure'] + 1) * (services_count + 1))
    df['ServiceDensity'] = services_count / (df['Tenure'] + 1)
    df['MonthlyChargesGrowthRate'] = (df['MonthlyCharges'] - df['AvgMonthlyCharges']) / (df['AvgMonthlyCharges'] + 1)
    
    # パーセンタイル比較比率
    df['MonthlyChargesPercentile'] = df['MonthlyCharges'].rank(pct=True)
    df['TotalChargesPercentile'] = df['TotalCharges'].rank(pct=True)
    df['TenurePercentile'] = df['Tenure'].rank(pct=True)
    
    # 複合比率
    df['ValuePerTenurePercentile'] = df['TotalChargesPercentile'] / (df['TenurePercentile'] + 0.01)
    df['MonthlyVsTenureBalance'] = df['MonthlyChargesPercentile'] / (df['TenurePercentile'] + 0.01)

    # 3. 詳細なサービス利用パターン
    df['ServicesCount'] = services_count
    df['ServiceUtilizationRate'] = df['ServicesCount'] / 8  # 最大8サービス（正確に計算）
    
    # サービス組み合わせパターン
    df['HasPhone'] = df['PhoneService'].astype(int)
    df['HasInternet'] = df['InternetService'].astype(int)
    df['HasMultipleLines'] = df['MultipleLines'].astype(int)
    
    # 高付加価値サービス
    df['PremiumServicesCount'] = df['StreamingTV'] + df['StreamingMovies'] + df['OnlineSecurity'] + df['TechSupport']
    df['BasicServicesCount'] = df['PhoneService'] + df['InternetService'] + df['OnlineBackup'] + df['DeviceProtection']
    
    # サービス組み合わせ効果
    df['HasBothStreaming'] = ((df['StreamingTV'] == 1) & (df['StreamingMovies'] == 1)).astype(int)
    df['HasFullSecurity'] = ((df['OnlineSecurity'] == 1) & (df['OnlineBackup'] == 1) & 
                            (df['DeviceProtection'] == 1) & (df['TechSupport'] == 1)).astype(int)
    df['HasPhoneAndInternet'] = ((df['PhoneService'] == 1) & (df['InternetService'] == 1)).astype(int)
    
    # サービスの不均衡
    df['ServiceImbalance'] = abs(df['PremiumServicesCount'] - df['BasicServicesCount'])
    df['PremiumToBasisRatio'] = df['PremiumServicesCount'] / (df['BasicServicesCount'] + 1)

    # 4. 強化された顧客ロイヤルティ指標
    df['LoyaltyScore'] = (df['Tenure'] * 0.3 + df['IsContracted'] * 0.3 + 
                         df['HasFullSecurity'] * 0.2 + df['PremiumServicesCount'] * 0.1 + 
                         (df['TotalCharges'] / (df['Tenure'] + 1) > df['MonthlyCharges'].median()) * 0.1)
    
    # より詳細なロイヤルティ指標
    df['CustomerStability'] = df['Tenure'] * df['IsContracted'] * (1 + df['ServiceUtilizationRate'])
    df['PaymentReliability'] = ((df['PaymentMethod'] == 'Bank transfer (automatic)') | 
                               (df['PaymentMethod'] == 'Credit card (automatic)')).astype(int)
    df['DigitalEngagement'] = (df['PaperlessBilling'] + df['HasBothStreaming'] + 
                              (df['InternetServiceType'] == 'Fiber optic').astype(int)) / 3

    # 5. 多面的なリスクファクター
    df['ChurnRiskFactor'] = (df['PaperlessBilling'] * 0.15 +
                            (1 - df['IsContracted']) * 0.25 +
                            (df['MonthlyCharges'] > df['MonthlyCharges'].quantile(0.8)) * 0.2 +
                            (df['Tenure'] < 12) * 0.2 +
                            (df['ServicesCount'] < 3) * 0.1 +
                            (df['PaymentMethod'] == 'Electronic check').astype(int) * 0.1)
    
    # 追加リスク要因
    df['HighChargesLowServices'] = ((df['MonthlyCharges'] > df['MonthlyCharges'].quantile(0.7)) & 
                                   (df['ServicesCount'] <= 3)).astype(int)
    df['NewCustomerHighCharges'] = ((df['Tenure'] < 6) & 
                                   (df['MonthlyCharges'] > df['MonthlyCharges'].median())).astype(int)
    df['UnstablePayment'] = (df['PaymentMethod'] == 'Electronic check').astype(int)
    
    # リスクスコア合成
    df['CompositeRiskScore'] = (df['ChurnRiskFactor'] + 
                               df['HighChargesLowServices'] * 0.3 + 
                               df['NewCustomerHighCharges'] * 0.2 + 
                               df['UnstablePayment'] * 0.1)

    # 6. 大幅に拡張した相互作用特徴量
    df['TenureMonthlyChargesInteraction'] = df['Tenure'] * df['MonthlyCharges']
    df['ContractMonthlyChargesInteraction'] = df['IsContracted'] * df['MonthlyCharges']
    
    # より多くの相互作用
    df['TenureServicesInteraction'] = df['Tenure'] * df['ServicesCount']
    df['MonthlyChargesServicesInteraction'] = df['MonthlyCharges'] * df['ServicesCount']
    df['ContractTenureInteraction'] = df['IsContracted'] * df['Tenure']
    df['SeniorPartnerInteraction'] = df['SeniorCitizen'] * df['Partner']
    df['ContractInternetInteraction'] = df['IsContracted'] * df['InternetService']
    
    # 3次相互作用
    df['TenureContractChargesInteraction'] = df['Tenure'] * df['IsContracted'] * df['MonthlyCharges']
    df['ServicesContractTenureInteraction'] = df['ServicesCount'] * df['IsContracted'] * df['Tenure']
    
    # 顧客セグメント相互作用
    df['SeniorHighValueInteraction'] = df['SeniorCitizen'] * df['IsHighValueCustomer']
    df['PartnerDependentsInteraction'] = df['Partner'] * df['Dependents']
    df['NewCustomerHighRiskInteraction'] = df['IsNewCustomer'] * df['CompositeRiskScore']
    
    # サービス組み合わせ相互作用
    df['InternetStreamingInteraction'] = df['InternetService'] * df['HasBothStreaming']
    df['PhoneMultiLineInteraction'] = df['PhoneService'] * df['MultipleLines']
    df['SecurityTechInteraction'] = df['OnlineSecurity'] * df['TechSupport']
    
    # LightGBM特化の非線形変換
    df['LogTotalCharges'] = np.log1p(df['TotalCharges'])
    df['LogMonthlyCharges'] = np.log1p(df['MonthlyCharges'])
    df['SqrtTenure'] = np.sqrt(df['Tenure'] + 1)
    df['TenureSquared'] = df['Tenure'] ** 2
    df['MonthlyChargesSquared'] = df['MonthlyCharges'] ** 2

    return df

# 新しい統計的集約特徴量
def feature_statistical_aggregates(df):
    # 各グループでの統計量を特徴量として追加
    
    # 1. 契約タイプ別統計量
    contract_stats = df.groupby('ContractType').agg({
        'MonthlyCharges': ['mean', 'std', 'median', 'max', 'min'],
        'TotalCharges': ['mean', 'std'],
        'Tenure': ['mean', 'std']
    }).round(2)
    contract_stats.columns = ['_'.join(col) for col in contract_stats.columns]
    
    # 2. インターネットサービス別統計量
    internet_stats = df.groupby('InternetServiceType').agg({
        'MonthlyCharges': ['mean', 'std'],
        'ServicesCount': ['mean', 'std']
    }).round(2)
    internet_stats.columns = ['_'.join(col) for col in internet_stats.columns]
    
    # 統計量を元のデータフレームにマージ
    for stat_col in contract_stats.columns:
        df[f'Contract_{stat_col}'] = df['ContractType'].map(
            contract_stats[stat_col].to_dict())
    
    for stat_col in internet_stats.columns:
        df[f'Internet_{stat_col}'] = df['InternetServiceType'].map(
            internet_stats[stat_col].to_dict())
    
    return df

# 大幅に拡張されたグループ内平均との差分特徴量
def feature_group_diff(df):
    # 統計的集約特徴量を先に追加
    df = feature_statistical_aggregates(df)
    
    # 1. 基本的なグループ別差分
    # 契約タイプ別の平均との差分
    contract_groups = df.groupby('ContractType')
    df['MonthlyCharges_ContractDiff'] = df['MonthlyCharges'] - df['ContractType'].map(
        contract_groups['MonthlyCharges'].mean())
    df['Tenure_ContractDiff'] = df['Tenure'] - df['ContractType'].map(
        contract_groups['Tenure'].mean())
    df['TotalCharges_ContractDiff'] = df['TotalCharges'] - df['ContractType'].map(
        contract_groups['TotalCharges'].mean())
    df['ServicesCount_ContractDiff'] = df['ServicesCount'] - df['ContractType'].map(
        contract_groups['ServicesCount'].mean())

    # インターネットサービスタイプ別の平均との差分
    internet_groups = df.groupby('InternetServiceType')
    df['MonthlyCharges_InternetDiff'] = df['MonthlyCharges'] - df['InternetServiceType'].map(
        internet_groups['MonthlyCharges'].mean())
    df['TotalCharges_InternetDiff'] = df['TotalCharges'] - df['InternetServiceType'].map(
        internet_groups['TotalCharges'].mean())
    df['Tenure_InternetDiff'] = df['Tenure'] - df['InternetServiceType'].map(
        internet_groups['Tenure'].mean())
    df['ServicesCount_InternetDiff'] = df['ServicesCount'] - df['InternetServiceType'].map(
        internet_groups['ServicesCount'].mean())

    # 年齢層（SeniorCitizen）別の平均との差分
    senior_groups = df.groupby('SeniorCitizen')
    df['MonthlyCharges_SeniorDiff'] = df['MonthlyCharges'] - df['SeniorCitizen'].map(
        senior_groups['MonthlyCharges'].mean())
    df['Tenure_SeniorDiff'] = df['Tenure'] - df['SeniorCitizen'].map(
        senior_groups['Tenure'].mean())
    df['TotalCharges_SeniorDiff'] = df['TotalCharges'] - df['SeniorCitizen'].map(
        senior_groups['TotalCharges'].mean())

    # 2. 支払い方法別の差分特徴量
    payment_groups = df.groupby('PaymentMethod')
    df['MonthlyCharges_PaymentDiff'] = df['MonthlyCharges'] - df['PaymentMethod'].map(
        payment_groups['MonthlyCharges'].mean())
    df['Tenure_PaymentDiff'] = df['Tenure'] - df['PaymentMethod'].map(
        payment_groups['Tenure'].mean())
    df['TotalCharges_PaymentDiff'] = df['TotalCharges'] - df['PaymentMethod'].map(
        payment_groups['TotalCharges'].mean())

    # 3. 顧客属性の組み合わせ別差分
    # パートナー・扶養家族別の差分
    partner_dep_groups = df.groupby(['Partner', 'Dependents'])
    df['MonthlyCharges_PartnerDepDiff'] = df['MonthlyCharges'] - df.apply(
        lambda x: partner_dep_groups['MonthlyCharges'].mean().get((x['Partner'], x['Dependents']), 
                                                                df['MonthlyCharges'].mean()), axis=1)
    df['Tenure_PartnerDepDiff'] = df['Tenure'] - df.apply(
        lambda x: partner_dep_groups['Tenure'].mean().get((x['Partner'], x['Dependents']), 
                                                          df['Tenure'].mean()), axis=1)

    # 4. サービス利用パターン別差分
    # サービス数レベル別の差分
    service_level_groups = df.groupby(pd.cut(df['ServicesCount'], bins=[0, 2, 4, 6, 8], labels=[0, 1, 2, 3]))
    df['ServicesLevel'] = pd.cut(df['ServicesCount'], bins=[0, 2, 4, 6, 8], labels=[0, 1, 2, 3]).fillna(0).astype(int)
    df['MonthlyCharges_ServiceLevelDiff'] = df['MonthlyCharges'] - df['ServicesLevel'].map(
        service_level_groups['MonthlyCharges'].mean())
    
    # ストリーミングサービス利用別の差分
    streaming_groups = df.groupby(df['StreamingTV'] + df['StreamingMovies'])
    df['StreamingLevel'] = df['StreamingTV'] + df['StreamingMovies']
    df['MonthlyCharges_StreamingDiff'] = df['MonthlyCharges'] - df['StreamingLevel'].map(
        streaming_groups['MonthlyCharges'].mean())
    df['Tenure_StreamingDiff'] = df['Tenure'] - df['StreamingLevel'].map(
        streaming_groups['Tenure'].mean())

    # 5. 高度なグループ別差分
    # 顧客価値レベル別差分
    value_groups = df.groupby('CustomerValueLevel')
    df['MonthlyCharges_ValueDiff'] = df['MonthlyCharges'] - df['CustomerValueLevel'].map(
        value_groups['MonthlyCharges'].mean())
    df['Tenure_ValueDiff'] = df['Tenure'] - df['CustomerValueLevel'].map(
        value_groups['Tenure'].mean())
    
    # 契約期間セグメント別差分
    tenure_seg_groups = df.groupby('TenureSegment')
    df['MonthlyCharges_TenureSegDiff'] = df['MonthlyCharges'] - df['TenureSegment'].map(
        tenure_seg_groups['MonthlyCharges'].mean())
    df['TotalCharges_TenureSegDiff'] = df['TotalCharges'] - df['TenureSegment'].map(
        tenure_seg_groups['TotalCharges'].mean())

    # 6. 複合グループ別差分
    # 契約タイプ×インターネットサービス別差分
    contract_internet_groups = df.groupby(['ContractType', 'InternetServiceType'])
    df['MonthlyCharges_ContractInternetDiff'] = df['MonthlyCharges'] - df.apply(
        lambda x: contract_internet_groups['MonthlyCharges'].mean().get(
            (x['ContractType'], x['InternetServiceType']), df['MonthlyCharges'].mean()), axis=1)
    
    # シニア×パートナー別差分
    senior_partner_groups = df.groupby(['SeniorCitizen', 'Partner'])
    df['MonthlyCharges_SeniorPartnerDiff'] = df['MonthlyCharges'] - df.apply(
        lambda x: senior_partner_groups['MonthlyCharges'].mean().get(
            (x['SeniorCitizen'], x['Partner']), df['MonthlyCharges'].mean()), axis=1)

    # 7. 百分位差分特徴量
    # 各グループ内での相対的位置
    for group_col in ['ContractType', 'InternetServiceType', 'PaymentMethod']:
        group_obj = df.groupby(group_col)
        
        # MonthlyCharges の百分位
        df[f'MonthlyCharges_{group_col}Percentile'] = group_obj['MonthlyCharges'].rank(pct=True)
        df[f'MonthlyCharges_{group_col}PercentileDiff'] = df[f'MonthlyCharges_{group_col}Percentile'] - 0.5
        
        # Tenure の百分位
        df[f'Tenure_{group_col}Percentile'] = group_obj['Tenure'].rank(pct=True)
        df[f'Tenure_{group_col}PercentileDiff'] = df[f'Tenure_{group_col}Percentile'] - 0.5

    # 8. 統計的特徴量差分
    # 各グループの標準偏差からの偏差
    for group_col in ['ContractType', 'InternetServiceType']:
        group_obj = df.groupby(group_col)
        
        # 標準化差分（Z-score的特徴量）
        group_mean = group_obj['MonthlyCharges'].transform('mean')
        group_std = group_obj['MonthlyCharges'].transform('std').fillna(1)
        df[f'MonthlyCharges_{group_col}ZScore'] = (df['MonthlyCharges'] - group_mean) / group_std
        
        # 四分位差分
        group_q25 = group_obj['MonthlyCharges'].transform(lambda x: x.quantile(0.25))
        group_q75 = group_obj['MonthlyCharges'].transform(lambda x: x.quantile(0.75))
        df[f'MonthlyCharges_{group_col}IQRPosition'] = (df['MonthlyCharges'] - group_q25) / (group_q75 - group_q25 + 1)

    return df