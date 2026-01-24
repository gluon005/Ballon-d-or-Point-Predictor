import pandas as pd
import joblib

scaler, model = joblib.load("nn_model.pkl")

data = pd.read_csv("NewData.csv", header=None)

X_new = data.iloc[:, 1:6].values
X_new_scaled = scaler.transform(X_new)

predictions = model.predict(X_new_scaled)
data["Predicted Points"] = predictions

data_sorted = data.sort_values(by="Predicted Points", ascending=False)

print("\nPredicted Ballon d'Or 2025 Rankings:")
for i, row in enumerate(data_sorted.itertuples(index=False), 1):
    print(f"{i}. {row[0]} - {row[-1]:.2f} points")
    
data.to_csv("PredictedResults.csv", index=False, header=False)
print("Predictions saved to PredictedResults.csv")
