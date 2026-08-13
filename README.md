# Telco Customer Churn Prediction

Kaggle Playground Series S6E3 の解法。

## 環境

```
Python 3.10+
xgboost
catboost
scikit-learn
pandas
numpy
matplotlib
seaborn
GPU: CUDA対応
```

## データ

以下の3ファイルをダウンロードして同じディレクトリに置く。

| ファイル | 内容 |
|---------|------|
| `train.csv` | 学習データ |
| `test.csv` | テストデータ |
| `WA_Fn-UseC_-Telco-Customer-Churn.csv` | IBM公式オリジナルデータ |

`CFG` クラスの `TRAIN_PATH` / `TEST_PATH` / `ORIGINAL_PATH` を環境に合わせて変更する。

## 実行

`churn-local-gpu-v2.ipynb` をセルの上から順に実行する。

## 特徴量

- Frequency Encoding
- 算術特徴量（charges_deviation, monthly_to_total_ratio, avg_monthly_charges）
- サービス数カウント
- オリジナルデータからの離脱率マッピング（ORIG_proba）
- 分布特徴量（pctrank / zscore 系）
- 分位距離特徴量
- 数値のカテゴリ化
- Digit特徴量
- N-gram特徴量（Bigram + Trigram）
- EDA由来特徴量（is_vulnerable, contract_risk_score など）

## モデル

- XGBoost（CUDA）
- CatBoost（GPU）
- アンサンブル: LRスタッキング → COBYLA重み最適化

## CV設計

- Outer: StratifiedKFold 20-fold
- Inner: StratifiedKFold 5-fold（Target Encodingのリーク防止）

