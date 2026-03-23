import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from utils.preprocess import preprocess_data

df = pd.read_csv("data/loan_data_large.csv")
df = preprocess_data(df)

X = df.drop("loan_status", axis=1)
y = df["loan_status"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

best_model = None
best_acc = 0

log_params = {
    "C": [0.01, 0.1, 1, 10],
    "solver": ["liblinear", "lbfgs"],
    "max_iter": [200, 500]
}

log_grid = GridSearchCV(
    LogisticRegression(),
    log_params,
    cv=5,
    n_jobs=-1
)

log_grid.fit(X_train_scaled, y_train)
log_preds = log_grid.predict(X_test_scaled)
log_acc = accuracy_score(y_test, log_preds)

print("Logistic:", log_acc, log_grid.best_params_)

if log_acc > best_acc:
    best_acc = log_acc
    best_model = log_grid.best_estimator_

dt_params = {
    "max_depth": [5, 10, 20, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}

dt_grid = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    dt_params,
    cv=5,
    n_jobs=-1
)

dt_grid.fit(X_train, y_train)
dt_preds = dt_grid.predict(X_test)
dt_acc = accuracy_score(y_test, dt_preds)

print("Decision Tree:", dt_acc, dt_grid.best_params_)

if dt_acc > best_acc:
    best_acc = dt_acc
    best_model = dt_grid.best_estimator_

rf_params = {
    "n_estimators": [100, 200],
    "max_depth": [10, 20, None],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
    "bootstrap": [True, False]
}

rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    rf_params,
    cv=5,
    n_jobs=-1
)

rf_grid.fit(X_train, y_train)
rf_preds = rf_grid.predict(X_test)
rf_acc = accuracy_score(y_test, rf_preds)

print("Random Forest:", rf_acc, rf_grid.best_params_)

if rf_acc > best_acc:
    best_acc = rf_acc
    best_model = rf_grid.best_estimator_

print("Best Model Accuracy:", best_acc)

joblib.dump(best_model, "models/model.pkl")
joblib.dump(scaler, "models/scaler.pkl")