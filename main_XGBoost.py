from xgboost import XGBClassifier
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE

#Coletando os dados
url = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
df = pd.read_csv(url)

#Transformação dos dados
df["Amount_log"] = np.log1p(df["Amount"]) 
scaler = StandardScaler()
df["Amount_scaled"] = scaler.fit_transform(df[["Amount_log"]])
df = df.drop(["Amount", "Amount_log"], axis=1)

#Como o número de caso fraude e não fraude são diferentes
#Vamos fazer um Undersampling(tornar o números de fraude e não fradue iguais)
fraudes = df[df["Class"]==1]
normais = df[df["Class"]==0].sample(len(fraudes), random_state=42)
df_under = (pd.concat([fraudes, normais]).sample(frac=1, random_state=42).reset_index(drop=True))

x = df_under.drop("Class", axis=1)
y = df_under["Class"]

#Modelo
x_train, x_test, y_train, y_test = train_test_split(x,y, stratify=y, test_size=0.3,random_state=42) 

xgb = XGBClassifier(
    scale_pos_weight=10, 
    use_label_encoder=False,
    eval_metrics="logloss"
)

xgb.fit(x_train, y_train)
y_pred_xgb = xgb.predict(x_test)

print(classification_report(y_test, y_pred_xgb))

#              precision    recall  f1-score   support

#          0       0.93      0.94      0.93       148
#          1       0.94      0.93      0.93       148

#   accuracy                           0.93       296
#  macro avg       0.93      0.93      0.93       296
# weighted avg       0.93      0.93      0.93       296
