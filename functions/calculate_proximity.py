import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error


def calculate_peer_weights(
    train_df,
    current_features_df,
    top_k=10,
    num_boost_round=150
):
    # =============================
    # 1. 学習データ準備
    # =============================
    features = train_df.columns.drop(["log_pbr"])
    target = "log_pbr"

    X_train = train_df[features]
    y_train = train_df[target]

    if len(X_train) < 50:
        return None, None

    # =============================
    # 2. LightGBM（近傍探索向け設定）
    # =============================
    params = {
        "objective": "regression",
        "verbose": -1,
        "random_state": 42,

        # ★ 重要：leaf を細かくする
        "num_leaves": 64,
        "max_depth": 8,
        "min_data_in_leaf": 5,

        # ★ 局所構造を重視
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1
    }

    train_set = lgb.Dataset(X_train, y_train)
    model = lgb.train(
        params,
        train_set,
        num_boost_round=num_boost_round
    )

    # =============================
    # 3. 予測性能（診断用）
    # =============================
    X_current = current_features_df[features]
    y_true = current_features_df[target].values
    y_pred = model.predict(X_current)

    mask = (~np.isnan(y_true)) & (~np.isnan(y_pred))
    if mask.sum() >= 2:
        corr, _ = pearsonr(y_true[mask], y_pred[mask])
        rmse = np.sqrt(mean_squared_error(y_true[mask], y_pred[mask]))
    else:
        corr, rmse = np.nan, np.nan

    print(f"corr: {corr:.4f}, rmse: {rmse:.4f}")

    # =============================
    # 4. Proximity（近傍探索の本体）
    # =============================
    # (N, T)
    leaf = model.predict(X_current, pred_leaf=True)

    # (N, N, T)
    matches = leaf[:, None, :] == leaf[None, :, :]

    # ★ 一致「回数」を使う（重要）
    proximity = matches.sum(axis=2).astype(float)

    # 自分自身を除外
    np.fill_diagonal(proximity, 0.0)

    # =============================
    # 5. 上位K社だけ残す（スパース化）
    # =============================
    N = proximity.shape[0]
    sparse_prox = np.zeros_like(proximity)

    for i in range(N):
        idx = np.argsort(proximity[i])[-top_k:]
        sparse_prox[i, idx] = proximity[i, idx]

    # =============================
    # 6. 行方向で正規化
    # =============================
    row_sums = sparse_prox.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    weights = sparse_prox / row_sums

    weights_df = pd.DataFrame(
        weights,
        index=current_features_df.index,
        columns=current_features_df.index
    )

    # 特徴量重要度を取得
    importance = pd.Series(
        model.feature_importance(importance_type="gain"),
        index=features
    )

    return weights_df, importance


def plot_average_importance(importance_list, figsize=(10, 6)):
    """
    複数回の特徴量重要度を平均してヒストグラムを描画する

    Parameters
    ----------
    importance_list : list of pd.Series
        calculate_peer_weights から返された importance のリスト
    figsize : tuple
        図のサイズ

    Returns
    -------
    pd.Series
        平均特徴量重要度
    """
    # None を除外
    valid_importance = [imp for imp in importance_list if imp is not None]

    if len(valid_importance) == 0:
        print("有効な特徴量重要度がありません")
        return None

    # DataFrameに変換して平均を計算
    importance_df = pd.DataFrame(valid_importance)
    avg_importance = importance_df.mean().sort_values(ascending=True)

    # ヒストグラム（横棒グラフ）を描画
    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(range(len(avg_importance)), avg_importance.values)
    ax.set_yticks(range(len(avg_importance)))
    ax.set_yticklabels(avg_importance.index)
    ax.set_xlabel("Average Feature Importance (gain)")
    ax.set_title(f"Feature Importance (averaged over {len(valid_importance)} periods)")
    plt.tight_layout()
    plt.show()

    return avg_importance
