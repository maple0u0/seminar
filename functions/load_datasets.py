import pandas as pd
import numpy as np
from functools import reduce
from datetime import datetime


def extract_item(df):
    tmp = df.rename(columns={df.columns[0]: "Company"})
    # 同じ企業名のレコードがある場合、最初のレコードを利用
    tmp = tmp.groupby("Company", as_index=False).first()
    return tmp.iloc[:, [0] + list(range(4, tmp.shape[1]))]


def sort_columns_chronologically(df):
    """
    日付列を時系列順（古い→新しい）に並び替え

    Parameters:
    -----------
    df : DataFrame
        日付列を含むDataFrame（Company列を含む）

    Returns:
    --------
    DataFrame
        時系列順に並び替えられたDataFrame
    """
    # Company列以外の列（日付列）を取得
    date_cols = [col for col in df.columns if col != 'Company']

    # 日付列をdatetimeに変換してソート
    date_tuples = [(col, datetime.strptime(col, '%Y/%m')) for col in date_cols]
    sorted_date_cols = [col for col, _ in sorted(date_tuples, key=lambda x: x[1])]

    # Company列 + ソート済み日付列の順でDataFrameを再構成
    return df[['Company'] + sorted_date_cols]


def load_and_merge_from_config(
    base_path: str,
    config: dict,
    on: str = "Company",
    how: str = "left",
    preprocess: bool = True,
):
    """
    Excelファイル群を読み込み、extract_itemを適用し、
    指定カラムで順にmergeしたDataFrameを返す
    """
    dfs = [
        extract_item(
            pd.read_excel(base_path + file, sheet_name=sheet, header=1)
        )
        for _, (file, sheet) in config.items()
    ]

    merged_df = reduce(
        lambda left, right: pd.merge(left, right, on=on, how=how),
        dfs
    )

    if preprocess:
        numeric_cols_pbr = [col for col in merged_df.columns if col != 'Company']
        merged_df[numeric_cols_pbr] = merged_df[numeric_cols_pbr].apply(pd.to_numeric, errors="coerce")

        merged_df = sort_columns_chronologically(merged_df)

    return merged_df


def load_datasets(base_path: str):

    pbr_config = {
        "df_pbr_20": ("pbr_2020.xls", 1),
        "df_pbr_21": ("pbr_2021.xls", 1),
        "df_pbr_22": ("pbr_2022.xls", 1),
        "df_pbr_23": ("pbr_2023.xls", 1),
        "df_pbr_24": ("pbr_2024.xls", 1),
        "df_pbr_25": ("pbr_2025.xls", 1),
    }

    stp_config = {
        "df_stp_20": ("stp_2020.xls", 1),
        "df_stp_21": ("stp_2021.xls", 1),
        "df_stp_22": ("stp_2022.xls", 1),
        "df_stp_23": ("stp_2023.xls", 1),
        "df_stp_24": ("stp_2024.xls", 1),
        "df_stp_25": ("stp_2025.xls", 1),
    }

    mktcap_config = {
        "df_mktcap_25_20": ("pbr_mktcap_2025-2020.xls", 2),
        "df_mktcap_19_15": ("pbr_mktcap_2019-2015.xls", 2),
        "df_mktcap_14_10": ("pbr_mktcap_2014-2010.xls", 2),
        "df_mktcap_09_05": ("pbr_mktcap_2009-2005.xls", 2),
    }

    df_pbr = load_and_merge_from_config(base_path, pbr_config)
    df_stp = load_and_merge_from_config(base_path, stp_config)
    df_mktcap = load_and_merge_from_config(base_path, mktcap_config, preprocess=False)

    return df_pbr, df_stp, df_mktcap


def split_df_by_year(df, p_to_year):
    """
    DataFrameを年ごとに分割する

    Parameters:
    -----------
    df : DataFrame
        Company列を含むDataFrame
    p_to_year : dict
        P列名と年のマッピング（例: {"P": 2025, "P-1": 2024, ...}）

    Returns:
    --------
    dict
        年をキー、Seriesを値とする辞書
    """
    df = df.set_index("Company")
    df = df[~df.index.duplicated(keep="first")]
    year_dict = {}
    for p_col, year in p_to_year.items():
        if p_col not in df.columns:
            continue
        s = df[p_col]
        s = s[~s.index.duplicated(keep="first")]
        year_dict[year] = s
    return year_dict


def organize_financial_data_by_year(dfs, years=None):
    """
    複数のDataFrameを年ごとに整理して結合する

    Parameters:
    -----------
    dfs : dict
        DataFrame名をキー、DataFrameを値とする辞書
    years : list, optional
        年のリスト。デフォルトは2025から2019まで（降順）

    Returns:
    --------
    dict
        年をキー、結合されたDataFrameを値とする辞書
    """
    if years is None:
        years = list(range(2025, 2019, -1))

    p_cols = ["P"] + [f"P-{i}" for i in range(1, 6)]
    p_to_year = dict(zip(p_cols, years))
    financial_dfs = {year: [] for year in years}

    for name, df in dfs.items():
        yearly_series = split_df_by_year(df, p_to_year)

        for year, series in yearly_series.items():
            financial_dfs[year].append(
                series.rename(name.replace("df_", ""))
            )

    for year in financial_dfs:
        if financial_dfs[year]:
            financial_dfs[year] = pd.concat(financial_dfs[year], axis=1)
            financial_dfs[year] = (
                financial_dfs[year].replace("None", np.nan)
                .replace("-", np.nan)
                .apply(pd.to_numeric, errors="coerce")
            )
        else:
            financial_dfs[year] = pd.DataFrame()

    return financial_dfs
