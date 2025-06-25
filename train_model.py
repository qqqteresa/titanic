from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
import torch
import joblib
torch.cuda.empty_cache()

# 設定設備 (使用 GPU 加速)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
# 讀取 Titanic 數據集
df = pd.read_csv("titanic_selected_features.csv")

# 分割特徵與標籤
X = df.drop("Survived", axis=1)
y = df["Survived"]

# 切分訓練與測試集（80% train / 20% test）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print('start train data')
# 建立 XGBoost 分類模型
model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.05,
    eval_metric="logloss"
)

# 訓練模型
model.fit(X_train, y_train)

# 預測與準確率
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# 儲存模型
print("save model")
joblib.dump(model, "xgb_model.pkl")

# 儲存特徵欄位
joblib.dump(X.columns.tolist(), "xgb_features.pkl")
print(f" XGBoost accuracy：{accuracy:.4f}")
