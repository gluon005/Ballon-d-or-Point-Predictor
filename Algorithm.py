import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split 
import joblib

column_names = ['name', 'x1', 'x2', 'x3', 'x4', 'x5', 'output']
data = pd.read_csv("Datas.csv", header=None, names=column_names)

X = data.iloc[:, 1:6].values
y = data.iloc[:, 6].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = MLPRegressor(
    hidden_layer_sizes=(10,),
    activation='relu',
    solver='adam',
    max_iter=5000,
    alpha=0.001,
    early_stopping=True,
    validation_fraction=0.2,
    random_state=42
)

model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print("Test Set Performance:") 
print(f"R² score: {r2:.4f}")
print(f"Mean Squared Error: {mse:.4f}")

joblib.dump((scaler, model), "nn_model.pkl")
print("\nNeural Network trained & saved as nn_model.pkl")