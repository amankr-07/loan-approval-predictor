import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from utils.preprocess import preprocess_data

df = pd.read_csv("data/loan_data.csv")

df = preprocess_data(df)

X = df.drop("loan_status",axis=1)
y = df["loan_status"]

X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.2,random_state=42
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

models = {
    "logistic": LogisticRegression(),
    "decision_tree": DecisionTreeClassifier(),
    "random_forest": RandomForestClassifier()
}

best_model = None
best_acc = 0

for name,model in models.items():

    model.fit(X_train,y_train)

    preds = model.predict(X_test)

    acc = accuracy_score(y_test,preds)

    print(name,acc)

    if acc > best_acc:
        best_acc = acc
        best_model = model

print("Best Model Accuracy:",best_acc)

joblib.dump(best_model,"models/model.pkl")
joblib.dump(scaler,"models/scaler.pkl")