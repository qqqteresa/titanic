import pandas as pd

# 讀取 Titanic 原始資料
print('read data')
df = pd.read_csv("train.csv")

# 處理缺失值
print('process missing values')
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Fare"] = df["Fare"].fillna(df["Fare"].median())
df["Embarked"] = df["Embarked"].fillna("S")

# FamilySize = SibSp + Parch + 1
print('process family_size')
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

# Title from Name
print('process Title from Name')
df["Title"] = df["Name"].str.extract(" ([A-Za-z]+)\.", expand=False)

# 將 Title 分為 Mr, Mrs, Master, Rare
df["Title"] = df["Title"].replace(
    ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev',
     'Sir', 'Jonkheer', 'Dona'], 'Rare')
df["Title"] = df["Title"].replace('Mlle', 'Miss')
df["Title"] = df["Title"].replace('Ms', 'Miss')
df["Title"] = df["Title"].replace('Mme', 'Mrs')

# One-hot encoding for Title (只保留 Mr, Mrs, Master, Rare，保留大方向)
print('One-hot encoding')
title_dummies = pd.get_dummies(df["Title"], prefix="Title")
for col in ["Mr", "Mrs", "Master", "Rare"]:
    df[f"Title_{col}"] = title_dummies.get(f"Title_{col}", 0)

# Sex (One-hot)
df["Sex_female"] = (df["Sex"] == "female").astype(int)

# Embarked (One-hot for C, S)
df["Embarked_C"] = (df["Embarked"] == "C").astype(int)
df["Embarked_S"] = (df["Embarked"] == "S").astype(int)

# 篩選出需要的特徵
print('select important features')
selected_columns = [
    "Sex_female", "Pclass", "Title_Mr", "Fare", "Age", "Title_Master",
    "FamilySize", "Title_Mrs", "Embarked_S", "SibSp", "TicketFreq",
    "Embarked_C", "Parch", "Title_Rare"
]

# 注意：某些 title 可能在這筆資料中沒出現，補 0 保險處理
for col in selected_columns:
    if col not in df.columns:
        df[col] = 0

selected_columns = ["PassengerId"] + ["Survived"] + selected_columns
# 輸出最終 DataFrame
final_df = df[selected_columns]

# 顯示前幾筆確認
print(final_df.head())

# 儲存為 CSV
final_df = df[selected_columns]
final_df.to_csv("titanic_selected_features.csv", index=False)

print("save as titanic_selected_features.csv")
