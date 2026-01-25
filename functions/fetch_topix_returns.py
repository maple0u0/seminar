"""
yfinanceを使用してTOPIXデータを取得し、月次リターン系列を返す関数
"""

import pandas as pd
import yfinance as yf


def fetch_topix_returns(start_date: str, end_date: str) -> pd.Series:
    """
    yfinanceからTOPIXデータを取得し、月次リターン系列を返す

    Parameters:
    -----------
    start_date : str
        開始日（YYYY-MM-DD形式）
    end_date : str
        終了日（YYYY-MM-DD形式）

    Returns:
    --------
    pd.Series
        月次リターン系列（インデックスはYYYY/MM形式）
    """
    # TOPIXのティッカーシンボル（^TPX または 1306.T など）
    ticker = "1306.T"

    # データを取得
    topix = yf.download(ticker, start=start_date, end=end_date, progress=False)

    if topix.empty:
        raise ValueError(f"TOPIXデータを取得できませんでした（{start_date} - {end_date}）")

    # 調整後終値を使用
    if 'Adj Close' in topix.columns:
        prices = topix['Adj Close']
    elif ('Adj Close', '^TPX') in topix.columns:
        prices = topix[('Adj Close', '^TPX')]
    elif 'Close' in topix.columns:
        prices = topix['Close']
    else:
        # MultiIndexの場合
        prices = topix.iloc[:, 0]

    # 月末データにリサンプリング
    monthly_prices = prices.resample('ME').last()

    # 月次リターンを計算（パーセント表示ではなく小数）
    monthly_returns = monthly_prices.pct_change().dropna()

    # インデックスをYYYY/MM形式に変換
    monthly_returns.index = monthly_returns.index.strftime('%Y/%m')
    monthly_returns.name = 'TOPIX'

    return monthly_returns


def fetch_topix_cumulative_returns(start_date: str, end_date: str) -> pd.Series:
    """
    yfinanceからTOPIXデータを取得し、累積リターン系列を返す（開始=1.0）

    Parameters:
    -----------
    start_date : str
        開始日（YYYY-MM-DD形式）
    end_date : str
        終了日（YYYY-MM-DD形式）

    Returns:
    --------
    pd.Series
        累積リターン系列（インデックスはYYYY/MM形式、開始=1.0）
    """
    monthly_returns = fetch_topix_returns(start_date, end_date)
    cumulative_returns = (1 + monthly_returns).cumprod()
    cumulative_returns.name = 'TOPIX'

    return cumulative_returns


if __name__ == "__main__":
    # テスト実行
    returns = fetch_topix_returns("2020-01-01", "2025-12-31")
    print("月次リターン:")
    print(returns.head(10))
    print(f"\n期間: {returns.index[0]} - {returns.index[-1]}")
    print(f"観測数: {len(returns)}")
