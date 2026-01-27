from datetime import datetime
import numpy as np
from functions.create_features import create_features


def create_monthly_feature_data(financial_dfs, df_pbr_sorted, df_industry):
    """
    毎月の特徴量データを作成

    ルール:
    - 各年の財務データ（financial_dfs[year]）は、year年6月末から利用可能
    - 例: financial_dfs[2020] は 2020/06 から利用可能

    リーク防止:
    - t月のデータにはt月以前に利用可能な財務データのみを使用
    """
    pbr_cols = [col for col in df_pbr_sorted.columns if col != 'Company']

    monthly_data = {}

    for month_col in pbr_cols:
        # 月を解析
        year_month = datetime.strptime(month_col, '%Y/%m')
        year = year_month.year
        month = year_month.month

        # 利用可能な財務データの年度を決定
        # 6月以降は当年の財務データ、6月未満は前年の財務データ
        if month >= 6:
            financial_year = year
        else:
            financial_year = year - 1

        # 財務データが存在しない場合はスキップ
        if financial_year not in financial_dfs:
            continue

        # 財務データを取得（インデックスはCompany）
        fin_df = financial_dfs[financial_year].copy()
        fin_df["industry"] = df_industry["日経業種中分類"].reindex(fin_df.index).astype('category')
        fin_df = create_features(fin_df)

        # 空のDataFrameの場合はスキップ
        if len(fin_df) == 0:
            continue

        # PBRデータを取得（この月のPBR）
        # Company列をインデックスに設定してマージできるようにする
        pbr_df = df_pbr_sorted[['Company', month_col]].copy()
        pbr_df = pbr_df.set_index('Company')
        pbr_series = pbr_df[month_col]

        # log_PBRを計算（0や負の値をNaNに変換）
        log_pbr = np.log(pbr_series.replace([0, np.inf, -np.inf], np.nan))

        # インデックス（Company）でマージ
        merged_df = fin_df.copy()
        merged_df['log_pbr'] = log_pbr

        monthly_data[month_col] = merged_df

    return monthly_data
