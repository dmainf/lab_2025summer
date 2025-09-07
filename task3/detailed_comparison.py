import pandas as pd
import numpy as np

def detailed_comparison():
    # データファイルを読み込み
    original_file = "../WA_Fn-UseC_-Telco-Customer-Churn.csv"
    modified_file = "after_data.csv"
    
    print("=== 各項目ごとの詳細変化分析 ===\n")
    
    try:
        df1 = pd.read_csv(original_file)
        df2 = pd.read_csv(modified_file)
        
        print(f"分析対象データ: {len(df1)}行 × {len(df1.columns)}列\n")
        
        # 各列の変化を分析
        column_changes = {}
        
        print("=" * 80)
        print("📊 全列の変化サマリー")
        print("=" * 80)
        print(f"{'列名':<20} {'変化数':<10} {'変化率':<10} {'データ型':<15} {'固有値数(元)':<12} {'固有値数(変更後)':<12}")
        print("-" * 80)
        
        for col in df1.columns:
            if col in df2.columns:
                # 値が異なるセルの数を計算
                different_cells = ~(df1[col].astype(str) == df2[col].astype(str))
                num_changes = different_cells.sum()
                change_percentage = (num_changes / len(df1)) * 100
                
                # データ型と固有値数
                unique_original = df1[col].nunique()
                unique_modified = df2[col].nunique()
                data_type = str(df1[col].dtype)
                
                column_changes[col] = {
                    'changed_count': num_changes,
                    'change_percentage': change_percentage,
                    'unique_original': unique_original,
                    'unique_modified': unique_modified,
                    'data_type': data_type
                }
                
                print(f"{col:<20} {num_changes:<10} {change_percentage:<10.2f}% {data_type:<15} {unique_original:<12} {unique_modified:<12}")
        
        print("=" * 80)
        print()
        
        # 変化があった列の詳細分析
        changed_columns = [col for col, info in column_changes.items() if info['changed_count'] > 0]
        
        if changed_columns:
            print(f"📈 変化があった列の詳細分析 ({len(changed_columns)}列)")
            print("=" * 80)
            
            for col in changed_columns:
                info = column_changes[col]
                print(f"\n🔍 [{col}] の変化詳細:")
                print(f"   変化データ数: {info['changed_count']:,}個")
                print(f"   変化率: {info['change_percentage']:.2f}%")
                print(f"   データ型: {info['data_type']}")
                print(f"   固有値数: {info['unique_original']} → {info['unique_modified']}")
                
                # カテゴリカル変数の場合、値の変化を詳細に表示
                if df1[col].dtype == 'object' or df1[col].nunique() < 50:
                    print(f"   📋 値別の変化:")
                    
                    value_counts1 = df1[col].value_counts()
                    value_counts2 = df2[col].value_counts()
                    
                    # 全ての値を取得
                    all_values = set(list(value_counts1.index) + list(value_counts2.index))
                    
                    value_changes = []
                    for val in all_values:
                        count1 = value_counts1.get(val, 0)
                        count2 = value_counts2.get(val, 0)
                        if count1 != count2:
                            change = count2 - count1
                            if count1 > 0:
                                change_pct = (change / count1) * 100
                            else:
                                change_pct = float('inf') if count2 > 0 else 0
                            
                            value_changes.append({
                                'value': val,
                                'original': count1,
                                'modified': count2,
                                'change': change,
                                'change_pct': change_pct
                            })
                    
                    # 変化量でソート
                    value_changes.sort(key=lambda x: abs(x['change']), reverse=True)
                    
                    # 上位10位まで表示
                    for i, change_info in enumerate(value_changes[:10]):
                        val = change_info['value']
                        count1 = change_info['original']
                        count2 = change_info['modified']
                        change = change_info['change']
                        change_pct = change_info['change_pct']
                        
                        if change_pct == float('inf'):
                            pct_str = "+∞%"
                        elif change_pct == float('-inf'):
                            pct_str = "-∞%"
                        else:
                            pct_str = f"{change_pct:+.1f}%"
                        
                        print(f"      {val}: {count1:,} → {count2:,} (変化: {change:+,}, {pct_str})")
                    
                    if len(value_changes) > 10:
                        print(f"      ... (他{len(value_changes)-10}個の値に変化)")
                
                # 数値変数の場合、統計値の変化を表示
                elif df1[col].dtype in ['int64', 'float64']:
                    print(f"   📊 統計値の変化:")
                    
                    try:
                        stats1 = df1[col].describe()
                        stats2 = df2[col].describe()
                        
                        stat_names = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']
                        for stat in stat_names:
                            if stat in stats1.index and stat in stats2.index:
                                val1 = stats1[stat]
                                val2 = stats2[stat]
                                change = val2 - val1
                                if val1 != 0:
                                    change_pct = (change / val1) * 100
                                else:
                                    change_pct = 0 if change == 0 else float('inf')
                                
                                if change_pct == float('inf'):
                                    pct_str = "+∞%"
                                else:
                                    pct_str = f"{change_pct:+.2f}%"
                                
                                print(f"      {stat}: {val1:.2f} → {val2:.2f} (変化: {change:+.2f}, {pct_str})")
                    except:
                        print("      統計値の計算でエラーが発生しました")
                
                print("-" * 60)
        
        else:
            print("📌 変化があった列はありません")
        
        print("\n" + "=" * 80)
        print("📋 変化サマリー")
        print("=" * 80)
        
        total_cells = len(df1) * len(df1.columns)
        total_changes = sum(info['changed_count'] for info in column_changes.values())
        
        print(f"総データセル数: {total_cells:,}")
        print(f"変更されたセル数: {total_changes:,}")
        print(f"全体変更率: {(total_changes / total_cells) * 100:.4f}%")
        print(f"変化があった列数: {len(changed_columns)}/{len(df1.columns)}")
        
        if changed_columns:
            print(f"\n変化があった列:")
            for col in changed_columns:
                info = column_changes[col]
                print(f"  • {col}: {info['changed_count']:,}個 ({info['change_percentage']:.2f}%)")
        
        print("=" * 80)
        
    except Exception as e:
        print(f"エラーが発生しました: {e}")

if __name__ == "__main__":
    detailed_comparison()