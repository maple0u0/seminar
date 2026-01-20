import lightgbm as lgb
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
        return None

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

    return weights_df
