
import pandas as pd
import joblib

# === Load the saved model and scaler ===
model = joblib.load('svc_model.pkl')
scaler = joblib.load('scaler.pkl')

# === Load new dataset ===
# Replace 'your_new_data.csv' with the path to your new dataset
new_data = pd.read_csv('your_new_data.csv')

# === Preprocess the new data ===
# Drop the target column if it's included in the new data
if 'PPMI_COHORT' in new_data.columns:
    new_data = new_data.drop(columns=['PPMI_COHORT'])

# Scale the features
new_data_scaled = scaler.transform(new_data)

# === Predict ===
predictions = model.predict(new_data_scaled)

# Optional: Decode the class labels (0 = Control, 1 = PD)
label_mapping = {0: 'Control', 1: 'PD'}
decoded_predictions = [label_mapping[p] for p in predictions]

# === Output predictions ===
# You can print or save to a CSV
output = pd.DataFrame({
    'Prediction': decoded_predictions
})
print(output)

# Optional: Save to CSV
output.to_csv('predictions.csv', index=False)
