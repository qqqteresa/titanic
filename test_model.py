import pandas as pd
import joblib
import xgboost as xgb

# 載入訓練時儲存的模型與特徵名稱
model = joblib.load("xgb_model.pkl")
feature_names = joblib.load("xgb_features.pkl")

# 載入新的 Titanic 測試資料
print('read test.csv')
df = pd.read_csv("test.csv")  

# 做出和訓練時一樣的特徵工程 -----------------------------

print('process feature')
# 缺失值處理
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Fare"] = df["Fare"].fillna(df["Fare"].median())
df["Embarked"] = df["Embarked"].fillna("S")

# FamilySize
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

# Title
df["Title"] = df["Name"].str.extract(" ([A-Za-z]+)\.", expand=False)
df["Title"] = df["Title"].replace(
    ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev',
     'Sir', 'Jonkheer', 'Dona'], 'Rare')
df["Title"] = df["Title"].replace(['Mlle', 'Ms'], 'Miss')
df["Title"] = df["Title"].replace('Mme', 'Mrs')

# One-hot Title
title_dummies = pd.get_dummies(df["Title"], prefix="Title")
for col in ["Title_Mr", "Title_Mrs", "Title_Master", "Title_Rare"]:
    df[col] = title_dummies.get(col, 0)

# Sex
df["Sex_female"] = (df["Sex"] == "female").astype(int)

# Embarked
df["Embarked_C"] = (df["Embarked"] == "C").astype(int)
df["Embarked_S"] = (df["Embarked"] == "S").astype(int)

# TicketFreq
df["TicketFreq"] = df.groupby("Ticket")["Ticket"].transform("count")

print('process feature complete ')
# 確保所有訓練用的特徵都有，若缺就補 0
for col in feature_names:
    if col not in df.columns:
        df[col] = 0

# 排序特徵順序一致
X_test = df[feature_names]

print('start predict .....')
# 預測
y_pred = model.predict(X_test)

# 加入結果
df["Survived"] = y_pred

# 儲存結果
df[["PassengerId", "Survived"]].to_csv("submission.csv", index=False)
print("prediction complete，save as submission.csv")
