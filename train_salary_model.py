import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# Corrected data path
data_path = r"C:\Users\amulb\OneDrive\Documents\Data Analystics\DataScience projects\AI-Powered Job Assistant for Data Science Roles\dataset"
df = pd.read_csv(os.path.join(data_path, "ds_salaries.csv"))

# Features & target
features = ["work_year", "experience_level", "employment_type", "job_title",
            "employee_residence", "remote_ratio", "company_location", "company_size"]
target = "salary_in_usd"

# Encode categorical columns
encoders = {}
for col in df.select_dtypes(include='object').columns:
    encoder = LabelEncoder()
    df[col] = encoder.fit_transform(df[col])
    encoders[col] = encoder

# Train model
X = df[features]
y = df[target]
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Save model and encoders
joblib.dump(model, os.path.join(data_path, "salary_model.pkl"))
joblib.dump(encoders, os.path.join(data_path, "encoders.pkl"))
