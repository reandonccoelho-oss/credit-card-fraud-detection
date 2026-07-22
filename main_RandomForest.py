import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

#Coletando os dados
url = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
df = pd.read_csv(url)

#Transformação dos dados
df["Amount_log"] = np.log1p(df["Amount"]) 
scaler = StandardScaler()
df["Amount_scaled"] = scaler.fit_transform(df[["Amount_log"]])
df = df.drop(["Amount", "Amount_log"], axis=1)

#Como o número de caso fraude e não fraude são diferentes
#Vamos fazer um Undesampling(tornar o números de fraude e não fradue iguais)
fraudes = df[df["Class"]==1]
normais = df[df["Class"]==0].sample(len(fraudes), random_state=42)
df_under = (pd.concat([fraudes, normais]).sample(frac=1, random_state=42).reset_index(drop=True))

#Modelo
x = df_under.drop("Class", axis=1)
y = df_under["Class"]

x_train, x_test, y_train, y_test = train_test_split(x,y, stratify=y, test_size=0.3,random_state=42) 
#train_test_split = Conjunto de teste
#stratify é usado para garantir a proporção de fraude e não fraude seja mantida, já que o banco de dados e 

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=50,
    max_depth=10,
    class_weight= "balanced",
    n_jobs=1,
    random_state=42
)
rf.fit(x_train, y_train)
y_pred_rf = rf.predict(x_test)
print(classification_report(y_test, y_pred_rf))



#             precision    recall  f1-score   support

#          0       0.91      0.97      0.94       148
#          1       0.97      0.91      0.94       148

#   accuracy                           0.94       296
#  macro avg       0.94      0.94      0.94       296
#weighted avg       0.94      0.94      0.94       296
