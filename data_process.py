import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

# 讀取 Titanic 數據集
df = pd.read_csv("train.csv")  

# 1️⃣ 處理缺失值
df["Age"] = df["Age"].fillna(df["Age"].median())# 年齡填補中位數
df["Embarked"] = df["Embarked"].fillna("S")# 登船港口填 'S'
df["Fare"] = df["Fare"].fillna(df["Fare"].median())
df["Cabin"] = df["Cabin"].fillna("U").map(
    lambda x: x[0])  # 艙房取首字母（A, B, C...U代表未知）

# 檢查缺失值是否填補成功
# print("✅ 缺失值處理後：")
# print(df.isnull().sum(), "\n")  # 應該全部為 0

# 創建新特徵 FamilySize
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1  # 家庭大小
df["IsAlone"] = (df["FamilySize"] == 1).astype(int)  # 是否獨自旅行
df["FareGroup"] = pd.qcut(df["Fare"], 5, labels=[1, 2, 3, 4, 5])  # 票價分組

# print("✅ 新增特徵後的數據：\n", df[["FamilySize", "IsAlone", "FareGroup"]].head())

# 2️⃣ 轉換類別變數（One-Hot Encoding）
df = pd.get_dummies(
    df, columns=["Sex", "Embarked", "Cabin", "FareGroup"], drop_first=True)
# 檢查 One-Hot Encoding 後的新特徵
# print("✅ One-Hot Encoding 後的特徵：")
# print(df.head(), "\n")

# 3️⃣ 選擇特徵與標籤
features = ["Pclass", "Age", "SibSp", "Parch", "Fare", "FamilySize", "IsAlone"]
features += [col for col in df.columns if col.startswith("Sex_") or col.startswith(
    "Embarked_") or col.startswith("Cabin_") or col.startswith("FareGroup_")]

X = df[features]
y = df["Survived"]
df_processed = df[features]  # 確保 "Survived" 也被存入
df_processed.to_csv("train_processed.csv", index=False)

print("✅ 資料處理完成，已儲存為 train_processed.csv！")

# # 檢查選擇的特徵
# print("✅ 選擇的特徵名稱：")
# print(features, "\n")

# 4️⃣ 標準化數據（Age, Fare）
scaler = StandardScaler()
X[["Age", "Fare"]] = scaler.fit_transform(X[["Age", "Fare"]])

# 檢查標準化後的 Age 和 Fare
print("✅ 標準化後的 Age 和 Fare：")
print(X[["Age", "Fare"]].describe(), "\n")

# 切分訓練集與測試集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
print(f"✅ 訓練集大小: {X_train.shape}, 測試集大小: {X_test.shape}")

# 訓練 XGBoost 模型

model = xgb.XGBClassifier(
    objective="binary:logistic",  # 二分類問題
    max_depth=5,
    learning_rate=0.005,
    eval_metric="logloss",  # 評估指標
    n_estimators=1000,
    random_state=1
)
model.fit(X_train, y_train)
model.save_model("xgboost_model.json")

# 評估模型
accuracy = model.score(X_test, y_test)
print(f"🎯 XGBoost 模型準確率: {accuracy:.4f}")
