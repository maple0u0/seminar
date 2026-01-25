def calculate_monthly_returns(df_price):
    """
    株価データから月次リターンを計算（時系列順であることを前提）

    Parameters:
    -----------
    df_price : DataFrame
        株価データ (行: 企業, 列: Company + 日付)
        ※日付列は時系列順（古い→新しい）に並んでいること

    Returns:
    --------
    DataFrame
        月次リターン (%)
    """
    # 日付列のみを抽出（Companyを除く）
    price_cols = [col for col in df_price.columns if col != 'Company']

    # リターンを格納するDataFrame
    returns = df_price[['Company']].copy()

    # 各月について、翌月のリターンを計算
    for i in range(len(price_cols) - 1):
        current_month = price_cols[i]
        next_month = price_cols[i + 1]

        returns[f'ret_{next_month}'] = (
            (df_price[next_month] / df_price[current_month] - 1)
        )

    return returns
