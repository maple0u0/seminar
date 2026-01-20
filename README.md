# MLを利用したバリュー投資戦略分析

日本株市場におけるバリュー投資戦略の効果を検証・分析するプロジェクトです。PBR（株価純資産倍率）を用いた様々な投資戦略を実装し、そのパフォーマンスを評価します。

## 目次

- [プロジェクト概要](#プロジェクト概要)
- [環境構築](#環境構築)
- [ディレクトリ構成](#ディレクトリ構成)
- [データの説明](#データの説明)
- [実験ノートブックの説明](#実験ノートブックの説明)
- [ユーティリティ関数](#ユーティリティ関数)
- [使い方](#使い方)

## プロジェクト概要

このプロジェクトでは、以下の投資戦略を分析・比較します：

1. **シンプルなバリュー戦略** - 低PBR銘柄への投資戦略
2. **業種中立バリュー戦略** - 業種内での相対的なバリュエーションに基づく戦略
3. **機械学習ベースのバリュー戦略** - LightGBMを用いたProximity（近傍）ベースの戦略

## 環境構築

### 必要要件

- Python 3.8以上
- pip（パッケージ管理）

### セットアップ手順

1. **リポジトリのクローン/ダウンロード**

```bash
cd seminar
```

2. **仮想環境の作成（推奨）**

```bash
python -m venv venv

source venv/bin/activate  # macOS/Linux
venv\Scripts\activate  # Windows
```

3. **依存パッケージのインストール**

```bash
pip install -r requirements.txt
```

## ディレクトリ構成

```
seminar/
│
├── data/                       # データファイル（別途Google Drive参照）
│
├── functions/                  # ユーティリティ関数
│   ├── __init__.py
│   ├── load_datasets.py       # データ読み込み・前処理
│   ├── calculate_monthly_returns.py  # 月次リターン計算
│   └── calculate_proximity.py # LightGBMベースのProximity計算
│
├── experiments/                # 分析用Jupyterノートブック
│   ├── value_effect.ipynb     # バリュー効果の基本検証
│   ├── simple_value_strategy.ipynb  # シンプルなバリュー戦略
│   ├── industry_neutral_value_strategy.ipynb  # 業種中立戦略
│   └── ml_value_strategy.ipynb  # 機械学習ベース戦略
│
├── requirements.txt            # Pythonパッケージの依存関係
└── README.md                   # このファイル
```
