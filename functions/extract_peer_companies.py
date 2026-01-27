import pandas as pd


def get_industry_name(code):
    """
    日経業種コードの2-3文字目を取得して業種名に変換
    """
    code_str = str(code)
    if len(code_str) >= 3:
        code_sub = code_str[1:3]  # 2文字目から3文字目まで（Pythonは0-indexedなので[1:3]）
    else:
        code_sub = code_str

    # コードと業種名のマッピング
    code_to_industry = {
        '35': '水産',
        '37': '鉱業',
        '41': '建設',
        '01': '食品',
        '03': '繊維',
        '05': 'パルプ・紙',
        '07': '化学工業',
        '09': '医薬品',
        '11': '石油',
        '13': 'ゴム',
        '15': '窯業',
        '17': '鉄鉱業',
        '19': '非金属及び金属製品',
        '21': '機械',
        '23': '電気機器',
        '25': '造船',
        '27': '自動車・自動車部品',
        '29': 'その他輸送用機器',
        '31': '精密機器',
        '33': 'その他製造業',
        '43': '商社',
        '45': '小売業',
        '47': '銀行',
        '49': '証券',
        '51': '保険',
        '52': 'その他金融業',
        '53': '不動産',
        '55': '鉄道・バス',
        '57': '陸運',
        '59': '海運',
        '61': '空輸',
        '63': '倉庫・運輸関連',
        '65': '通信',
        '67': '電力',
        '69': 'ガス',
        '71': 'サービス業'
    }

    return code_to_industry.get(code_sub, None)


def get_peer_pbr_analysis(company_name, weights_df, df_pbr, df_industry, target_date="2025/12"):
    """
    企業名を入力すると、その企業の実際のPBRと類似企業10件の情報、重みづけ平均を返す関数

    Parameters:
    -----------
    company_name : str
        分析対象の企業名
    weights_df : pd.DataFrame
        類似度の重み行列（行・列が企業名のインデックス）
    df_pbr : pd.DataFrame
        PBRデータ（Company列と日付列を持つ）
    target_date : str
        対象となる日付（デフォルト: "2025/12"）

    Returns:
    --------
    dict
        - actual_pbr: 対象企業の実際のPBR
        - peer_df: 類似企業10件のDataFrame（会社名、類似度、PBR）
        - weighted_avg_pbr: 重みづけ平均PBR
    """
    # 1. 対象企業の実際のPBRを取得
    actual_pbr = df_pbr.set_index("Company").loc[company_name][target_date]

    # 2. 類似企業10件を取得（類似度の高い順）
    similar_companies = weights_df.loc[company_name].sort_values(ascending=False).head(10).index.tolist()

    # 3. 類似企業の類似度を取得
    similarities = weights_df.loc[company_name, similar_companies]

    # 4. 類似企業のPBRを取得
    pbr_values = df_pbr.set_index("Company").loc[similar_companies][target_date]

    # 5 類似企業の業種を取得
    industry_values = df_industry.loc[similar_companies]["日経業種中分類"]
    industry_values = industry_values.apply(lambda x: get_industry_name(x))

    # 5. DataFrameを作成
    peer_df = pd.DataFrame({
        '会社名': similar_companies,
        '類似度': similarities.values,
        'PBR': pbr_values.values,
        '業種': industry_values.values
    })

    # 6. 重みづけ平均PBRを計算
    weighted_avg_pbr = (peer_df['PBR'] * peer_df['類似度']).sum() / peer_df['類似度'].sum()

    return {
        'actual_pbr': actual_pbr,
        'peer_df': peer_df,
        'weighted_avg_pbr': weighted_avg_pbr
    }