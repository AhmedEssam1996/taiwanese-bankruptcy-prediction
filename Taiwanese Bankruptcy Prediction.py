#library
import numpy as np
import pandas as pd
#-----------------------
#visualization library
import seaborn as sns
import matplotlib.pyplot as plt
#-----------------------
#Preprocessing library
from sklearn.preprocessing import MinMaxScaler
#-----------------------
#Model library
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve,mean_squared_error ,r2_score, precision_score, recall_score, f1_score
from xgboost import plot_tree

#----------------------------------------------------------------------------
#Reading and describe data
data=pd.read_csv(r'C:\Users\Elbostan\Desktop\full project\taiwanese+bankruptcy+prediction\data.csv')
print(data.head())
print(data.tail())
print(data.info())
print(data.isna().sum())

summary=data.describe()
fig, ax = plt.subplots(figsize=(40, 10))
sns.heatmap(summary,
            annot=True,
            cbar=True,
            cmap='magma')
plt.show()
#-----------------------------------------------------------------------------
#data preprocessing
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

data = pd.DataFrame(scaled_data, columns=data.columns)
#-----------------------------------------------------------------------------
#MODEL
X=data.iloc[:,1:]
y=data.iloc[:,0:1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
train_data = xgb.DMatrix(X_train, label=y_train)
test_data = xgb.DMatrix(X_test, label=y_test)

#------------------------------------------------------------------------------
params = {
    'objective': 'multi:softmax',  # نوع المشكلة (تصنيف متعدد)
    'num_class': 2,               # عدد الفئات
    'max_depth': 4,               # عمق الأشجار
    'learning_rate': 0.1,         # معدل التعلم
    'n_estimators': 100,          # عدد الأشجار
    'eval_metric': 'mlogloss'     # معيار التقييم
}

model = xgb.train(params, train_data, num_boost_round=100, evals=[(test_data, 'test')], early_stopping_rounds=10)

y_pred = model.predict(test_data)

print(f"الدقة: {accuracy_score(y_test, y_pred):.2f}")

#------------------------------------------------------------------------------

from xgboost import plot_importance


# إعداد الرسم
fig, ax = plt.subplots(figsize=(20, 30))

# رسم أهمية الميزات
plot_importance(model, ax=ax)  # استخدام "ax" لتخصيص الرسم

# عرض الرسم
plt.show()

model.save_model('model.json')