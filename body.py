from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pandas as pd

# 加载客户行为数据
data = pd.read_csv("customer_behavior.csv")
X = data.drop(columns=["Risk_Level"])
y = data["Risk_Level"]

# 划分训练和测试集
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 模型训练
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 模型评估
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# 新客户风险评分
new_customer = pd.DataFrame([{"Feature1": 0.8, "Feature2": 0.3, ...}])
risk_score = model.predict_proba(new_customer)[:, 1]
print("客户风险评分:", risk_score)
