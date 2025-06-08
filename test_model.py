import pandas as pd
import xgboost as xgb
import pickle  # 用來讀取訓練好的模型

# 讀取測試數據
df_test = pd.read_csv("test.csv")

# 保存 PassengerId（最後要用來生成 submission.csv）
passenger_ids = df_test["PassengerId"]

# **處理缺失值（和 train.csv 一樣的處理方式）**
df_test["Age"] = df_test["Age"].fillna(df_test["Age"].median())  # 填補年齡中位數
df_test["Fare"] = df_test["Fare"].fillna(df_test["Fare"].median())  # 填補票價中位數
df_test["Embarked"] = df_test["Embarked"].fillna("S")  # 登船港口填 'S'
df_test["Cabin"] = df_test["Cabin"].fillna("U").map(lambda x: x[0])  # 艙房首字母

# **新增特徵（和 train.csv 一樣的處理方式）**
df_test["FamilySize"] = df_test["SibSp"] + df_test["Parch"] + 1  # 家庭大小
df_test["IsAlone"] = (df_test["FamilySize"] == 1).astype(int)  # 是否獨自旅行
df_test["FareGroup"] = pd.qcut(
df_test["Fare"], 5, labels=[1, 2, 3, 4, 5])  # 票價分組

# **進行 One-Hot Encoding（和 train.csv 一樣的處理方式）**
df_test = pd.get_dummies(
    df_test, columns=["Sex", "Embarked", "Cabin", "FareGroup"], drop_first=True)

# **確保 test 的欄位和 train 一致**
df_train = pd.read_csv("train_processed.csv")  # 讀取處理過的 train 資料
train_columns = df_train.columns.tolist()
# train_columns.remove("Survived")  # 移除 Survived，因為 test 沒有這個欄位

# **確保 test 有所有 train 的欄位**
df_test = df_test.reindex(columns=train_columns, fill_value=0)  # 如果有缺的欄位，補 0

# **儲存處理後的 test 數據**
df_test.to_csv("test_processed.csv", index=False)

# **讀取已訓練的模型**
model = xgb.XGBClassifier()  # 如果是使用 XGBClassifier 儲存
model.load_model("xgboost_model.json")  # 或者是 xgboost_model.bin

# **進行預測**
predictions = model.predict(df_test)

# **生成 Kaggle 提交檔案**
submission = pd.DataFrame(
    {"PassengerId": passenger_ids, "Survived": predictions})
submission.to_csv("submission.csv", index=False)

print("✅ 完成！請上傳 submission.csv 到 Kaggle 🚀")
