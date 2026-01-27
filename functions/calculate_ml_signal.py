import numpy as np
import pandas as pd
from functions.calculate_proximity import calculate_peer_weights


def calculate_ml_signal(monthly_feature_data, df_pbr_sorted, df_returns,
                        n_quintiles=5, lookback_months=12, top_k=10, save_debug_file=None):
    pbr_cols = [col for col in df_pbr_sorted.columns if col != 'Company']
    return_cols = [col for col in df_returns.columns if col.startswith('ret_')]

    portfolio_returns = {f'Q{i+1}': [] for i in range(n_quintiles)}
    portfolio_returns['Date'] = []

    importance_list = []

    debug_data = {
        'relative_pbr': {},
        'peer_avg_pbr': {},
        'weights_df_sample': {},
        'pbr_subset': {},
    }

    available_months = sorted([m for m in monthly_feature_data.keys()])

    for i in range(len(pbr_cols) - 1):
        formation_month = pbr_cols[i]
        return_month = return_cols[i]

        portfolio_returns['Date'].append(return_month.replace('ret_', ''))

        # 2020/01～2020/05 は利用不可
        if formation_month not in monthly_feature_data:
            for q in range(n_quintiles):
                portfolio_returns[f'Q{q+1}'].append(np.nan)
            continue

        current_df = monthly_feature_data[formation_month].copy()

        # PBRがある企業だけに絞り込む
        pbr_series = df_pbr_sorted.set_index('Company')[formation_month]
        valid_pbr = pbr_series.dropna()
        common_companies = current_df.index.intersection(valid_pbr.index)

        if len(common_companies) < 50:
            for q in range(n_quintiles):
                portfolio_returns[f'Q{q+1}'].append(np.nan)
            continue

        # PBRがある企業だけでフィルタリング
        current_df_filtered = current_df.loc[common_companies]

        formation_idx = available_months.index(formation_month) if formation_month in available_months else -1

        if formation_idx < 0:
            for q in range(n_quintiles):
                portfolio_returns[f'Q{q+1}'].append(np.nan)
            continue

        past_months = available_months[max(0, formation_idx - lookback_months):formation_idx]

        train_dfs = []
        for past_month in past_months:
            if past_month in monthly_feature_data:
                # 学習データもPBRがある企業だけに絞り込む
                past_df = monthly_feature_data[past_month]
                past_pbr_col = past_month
                if past_pbr_col in df_pbr_sorted.columns:
                    past_pbr = df_pbr_sorted.set_index('Company')[past_pbr_col].dropna()
                    past_valid = past_df.index.intersection(past_pbr.index)
                    if len(past_valid) > 0:
                        train_dfs.append(past_df.loc[past_valid])

        if len(train_dfs) == 0:
            for q in range(n_quintiles):
                portfolio_returns[f'Q{q+1}'].append(np.nan)
            continue

        train_df = pd.concat(train_dfs, axis=0)

        try:
            # PBRがある企業だけで類似度計算
            weights_df, importance = calculate_peer_weights(
                train_df=train_df,
                current_features_df=current_df_filtered,
                top_k=top_k,
                num_boost_round=100
            )
            importance_list.append(importance)
        except Exception as e:
            for q in range(n_quintiles):
                portfolio_returns[f'Q{q+1}'].append(np.nan)
            print(f"Error: {e}")
            continue

        if weights_df is None:
            for q in range(n_quintiles):
                portfolio_returns[f'Q{q+1}'].append(np.nan)
            print("Error: weights_df is None")
            continue

        # weights_dfは既にPBRがある企業だけなので、フィルタリング不要
        pbr_subset = valid_pbr.loc[common_companies]

        peer_avg_pbr = weights_df @ pbr_subset
        peer_avg_pbr = peer_avg_pbr.fillna(pbr_subset.mean())

        relative_pbr = pbr_subset / peer_avg_pbr

        # formation_monthの値を確認
        if "2025" in formation_month and "11" in formation_month:
            print(f"Debug: formation_month = {formation_month}")
            debug_data['relative_pbr'][formation_month] = relative_pbr.copy()
            debug_data['peer_avg_pbr'][formation_month] = peer_avg_pbr.copy()
            debug_data['pbr_subset'][formation_month] = pbr_subset.copy()
            sample_companies = list(common_companies[:10])
            debug_data['weights_df_sample'][formation_month] = weights_df.loc[sample_companies, sample_companies].copy()
            print(f"Debug: データを保存しました - {debug_data}")
        return_series = df_returns.set_index('Company')[return_month]

        temp_df = pd.DataFrame({
            'Company': relative_pbr.index,
            'Relative_PBR': relative_pbr.values,
            'Return': return_series.reindex(relative_pbr.index).values
        })

        temp_df = temp_df.dropna()

        if len(temp_df) < n_quintiles:
            for q in range(n_quintiles):
                portfolio_returns[f'Q{q+1}'].append(np.nan)
            continue

        try:
            temp_df['Quintile'] = pd.qcut(temp_df['Relative_PBR'], q=n_quintiles, labels=False, duplicates='drop')
        except Exception as e:
            for q in range(n_quintiles):
                portfolio_returns[f'Q{q+1}'].append(np.nan)
            print(f"Error: {e}")
            continue

        for q in range(n_quintiles):
            quintile_returns = temp_df[temp_df['Quintile'] == q]['Return']
            avg_return = quintile_returns.mean() if len(quintile_returns) > 0 else np.nan
            portfolio_returns[f'Q{q+1}'].append(avg_return)

    result_df = pd.DataFrame(portfolio_returns)
    result_df = result_df.set_index('Date')

    if save_debug_file:
        import pickle
        with open(save_debug_file, 'wb') as f:
            pickle.dump(debug_data, f)
        print(f"デバッグデータを {save_debug_file} に保存しました")

    return result_df, importance_list
