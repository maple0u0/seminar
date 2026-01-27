import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def analyze_peer_industry_match(
    weights_df: pd.DataFrame,
    df_industry: pd.DataFrame,
    top_k: int = 10,
    show_plot: bool = True,
    industry_col: str = None
) -> pd.DataFrame:
    """
    類似企業（上位K社）が同じ業種に分類されている割合を分析し、
    ヒストグラムとして描画する。

    Parameters
    ----------
    weights_df : pd.DataFrame
        calculate_peer_weights()から返される類似度重み行列
        index/columnsが企業名
    df_industry : pd.DataFrame
        日経業種分類データ
        - インデックスが企業名の場合: industry_colで業種カラムを指定
        - columns[0]が企業名の場合: columns[2]を業種として使用
    top_k : int
        上位何社を類似企業として見るか（デフォルト: 10）
    show_plot : bool
        ヒストグラムを描画するかどうか（デフォルト: True）
    industry_col : str
        業種カラム名（df_industryのインデックスが企業名の場合に指定）

    Returns
    -------
    pd.DataFrame
        各企業の同業種企業数と割合を含むDataFrame
    """
    # 業種データを辞書として準備
    # df_industryのインデックスが企業名かどうかで処理を分岐
    if industry_col is not None:
        # インデックスが企業名、指定されたカラムが業種
        industry_dict = df_industry[industry_col].to_dict()
    elif df_industry.index.name == 'Company' or df_industry.index.dtype == 'object':
        # インデックスが企業名と推測される場合
        # 最初の文字列カラムを業種として使用
        str_cols = df_industry.select_dtypes(include=['object']).columns
        if len(str_cols) > 0:
            industry_dict = df_industry[str_cols[0]].to_dict()
        else:
            industry_dict = df_industry.iloc[:, -1].to_dict()
    else:
        # 従来の形式: columns[0]が企業名、columns[2]が業種
        _industry_col = df_industry.columns[2]
        _company_col = df_industry.columns[0]
        industry_dict = {}
        for _, row in df_industry.iterrows():
            company_name = row[_company_col]
            industry_name = row[_industry_col]
            if company_name not in industry_dict and pd.notna(company_name):
                industry_dict[company_name] = industry_name

    # 各企業について、上位K社の類似企業が同業種かどうかを分析
    results = []

    for company in weights_df.index:
        # この企業の業種を取得
        if company not in industry_dict:
            continue

        company_industry = industry_dict[company]

        # 類似度の高い上位K社を取得（自分自身は除外済み）
        similarities = weights_df.loc[company].copy()
        top_peers = similarities.nlargest(top_k).index.tolist()

        # 上位K社のうち同業種の企業数をカウント
        same_industry_count = 0
        valid_peers = 0

        for peer in top_peers:
            if peer in industry_dict:
                valid_peers += 1
                if industry_dict[peer] == company_industry:
                    same_industry_count += 1

        if valid_peers > 0:
            results.append({
                'Company': company,
                'Industry': company_industry,
                'SameIndustryCount': same_industry_count,
                'ValidPeers': valid_peers,
                'SameIndustryRatio': same_industry_count / valid_peers
            })

    result_df = pd.DataFrame(results)

    if len(result_df) == 0:
        print("Warning: No valid data to analyze")
        return result_df

    # 統計情報を表示
    print("=" * 70)
    print(f"類似企業（上位{top_k}社）の業種一致度分析")
    print("=" * 70)
    print(f"分析対象企業数: {len(result_df)}")
    print("\n同業種企業数の分布:")
    print(f"  平均: {result_df['SameIndustryCount'].mean():.2f} / {top_k}")
    print(f"  中央値: {result_df['SameIndustryCount'].median():.1f} / {top_k}")
    print(f"  標準偏差: {result_df['SameIndustryCount'].std():.2f}")
    print(f"  最小: {result_df['SameIndustryCount'].min()}")
    print(f"  最大: {result_df['SameIndustryCount'].max()}")
    print(f"\n同業種割合の平均: {result_df['SameIndustryRatio'].mean()*100:.1f}%")
    print("=" * 70)

    # ヒストグラムを描画
    if show_plot:
        _, axes = plt.subplots(1, 2, figsize=(14, 5))

        # 左: 同業種企業数のヒストグラム
        ax1 = axes[0]
        bins = np.arange(-0.5, top_k + 1.5, 1)
        ax1.hist(result_df['SameIndustryCount'], bins=bins,
                 color='#4A5899', edgecolor='black', alpha=0.8)
        ax1.set_xlabel('Number of Same-Industry Peers', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax1.set_title(f'Distribution of Same-Industry Peers (Top {top_k})',
                      fontsize=14, fontweight='bold')
        ax1.set_xticks(range(0, top_k + 1))
        ax1.axvline(x=result_df['SameIndustryCount'].mean(), color='red',
                    linestyle='--', linewidth=2, label=f"Mean: {result_df['SameIndustryCount'].mean():.1f}")
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3, axis='y')

        # 右: 同業種割合のヒストグラム（割合表示）
        ax2 = axes[1]
        # weightsを使って割合（パーセント）に変換
        weights = np.ones(len(result_df)) / len(result_df) * 100
        ax2.hist(result_df['SameIndustryRatio'] * 100, bins=11,
                 weights=weights,
                 color='#8B3A3A', edgecolor='black', alpha=0.8)
        ax2.set_xlabel('Same-Industry Ratio (%)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Proportion (%)', fontsize=12, fontweight='bold')
        ax2.set_title(f'Distribution of Same-Industry Ratio (Top {top_k})',
                      fontsize=14, fontweight='bold')
        ax2.axvline(x=result_df['SameIndustryRatio'].mean() * 100, color='red',
                    linestyle='--', linewidth=2, label=f"Mean: {result_df['SameIndustryRatio'].mean()*100:.1f}%")
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.show()

    return result_df


def analyze_peer_industry_by_sector(
    weights_df: pd.DataFrame,
    df_industry: pd.DataFrame,
    top_k: int = 10,
    show_plot: bool = True
) -> pd.DataFrame:
    """
    業種別に、類似企業の同業種割合を分析する。

    Parameters
    ----------
    weights_df : pd.DataFrame
        calculate_peer_weights()から返される類似度重み行列
    df_industry : pd.DataFrame
        日経業種分類データ
    top_k : int
        上位何社を類似企業として見るか
    show_plot : bool
        棒グラフを描画するかどうか

    Returns
    -------
    pd.DataFrame
        業種別の同業種割合統計
    """
    # まず全体の分析を実行
    result_df = analyze_peer_industry_match(
        weights_df, df_industry, top_k, show_plot=False
    )

    if len(result_df) == 0:
        return result_df

    # 業種別に集計
    industry_stats = result_df.groupby('Industry').agg({
        'SameIndustryCount': ['mean', 'std', 'count'],
        'SameIndustryRatio': 'mean'
    }).round(3)

    industry_stats.columns = ['MeanCount', 'StdCount', 'NumCompanies', 'MeanRatio']
    industry_stats = industry_stats.sort_values('MeanRatio', ascending=False)

    print("\n" + "=" * 70)
    print("業種別の同業種割合")
    print("=" * 70)
    print(industry_stats.to_string())

    if show_plot:
        # 上位・下位10業種を表示
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # 上位10業種（同業種割合が高い）
        top_industries = industry_stats.head(10)
        ax1 = axes[0]
        _ = ax1.barh(range(len(top_industries)), top_industries['MeanRatio'] * 100, color='#4A5899', edgecolor='black', alpha=0.8)
        ax1.set_yticks(range(len(top_industries)))
        ax1.set_yticklabels(top_industries.index, fontsize=10)
        ax1.set_xlabel('Same-Industry Ratio (%)', fontsize=12, fontweight='bold')
        ax1.set_title('Top 10 Industries (Highest Same-Industry Ratio)',
                      fontsize=12, fontweight='bold')
        ax1.invert_yaxis()
        ax1.grid(True, alpha=0.3, axis='x')

        # 下位10業種（同業種割合が低い）
        bottom_industries = industry_stats.tail(10)
        ax2 = axes[1]
        _ = ax2.barh(range(len(bottom_industries)), bottom_industries['MeanRatio'] * 100, color='#8B3A3A', edgecolor='black', alpha=0.8)
        ax2.set_yticks(range(len(bottom_industries)))
        ax2.set_yticklabels(bottom_industries.index, fontsize=10)
        ax2.set_xlabel('Same-Industry Ratio (%)', fontsize=12, fontweight='bold')
        ax2.set_title('Bottom 10 Industries (Lowest Same-Industry Ratio)',
                      fontsize=12, fontweight='bold')
        ax2.invert_yaxis()
        ax2.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()
        plt.show()

    return industry_stats
