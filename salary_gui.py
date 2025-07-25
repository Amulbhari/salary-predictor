import tkinter as tk
from tkinter import ttk
import pandas as pd
import joblib
import os

# ========== PATH SETUP ========== #
data_path = r"C:\Users\amulb\OneDrive\Documents\Data Analystics\DataScience projects\AI-Powered Job Assistant for Data Science Roles"
model = joblib.load(os.path.join(data_path, "salary_model.pkl"))
encoders = joblib.load(os.path.join(data_path, "encoders.pkl"))
df = pd.read_csv(os.path.join(data_path, "ds_salaries.csv"))

# ========== MAIN WINDOW ========== #
root = tk.Tk()
root.title("💼 AI-Powered Salary Predictor")
root.geometry("700x560")
root.configure(bg="#f4f9fc")

# ========== TITLE ========== #
tk.Label(root, text="Salary Predictor for Data Science Roles",
         font=("Helvetica", 18, "bold"), bg="#f4f9fc", fg="#003366").pack(pady=20)

# ========== FORM FRAME ========== #
form_frame = tk.Frame(root, bg="#f4f9fc")
form_frame.pack()

# ========== FORM INPUTS ========== #
fields = {
    "Work Year": sorted(df["work_year"].unique()),
    "Experience Level": list(encoders["experience_level"].classes_),
    "Employment Type": list(encoders["employment_type"].classes_),
    "Job Title": sorted(df["job_title"].unique()),
    "Employee Residence": list(encoders["employee_residence"].classes_),
    "Remote Ratio": sorted(df["remote_ratio"].unique()),
    "Company Location": list(encoders["company_location"].classes_),
    "Company Size": list(encoders["company_size"].classes_)
}

entries = {}
for i, (label, values) in enumerate(fields.items()):
    tk.Label(form_frame, text=f"{label}:", font=("Arial", 11), bg="#f4f9fc").grid(row=i, column=0, sticky="e", padx=12, pady=8)
    combo = ttk.Combobox(form_frame, values=values, state="readonly", width=45)
    combo.grid(row=i, column=1, padx=12, pady=8)
    combo.current(0)
    entries[label] = combo

# ========== RESULT LABEL ========== #
result_label = tk.Label(root, text="", font=("Arial", 14, "bold"), bg="#f4f9fc", fg="green")
result_label.pack(pady=20)

# ========== PREDICT FUNCTION ========== #
def predict_salary():
    try:
        input_data = {
            "work_year": int(entries["Work Year"].get()),
            "experience_level": entries["Experience Level"].get(),
            "employment_type": entries["Employment Type"].get(),
            "job_title": entries["Job Title"].get(),
            "employee_residence": entries["Employee Residence"].get(),
            "remote_ratio": int(entries["Remote Ratio"].get()),
            "company_location": entries["Company Location"].get(),
            "company_size": entries["Company Size"].get()
        }

        # Apply encoders
        for key in input_data:
            if key in encoders:
                input_data[key] = encoders[key].transform([input_data[key]])[0]

        input_df = pd.DataFrame([input_data])
        prediction = model.predict(input_df)[0]
        result_label.config(text=f"💰 Predicted Salary: ${int(prediction):,} USD", fg="green")

    except Exception as e:
        result_label.config(text=f"❌ Error: {str(e)}", fg="red")

# ========== PREDICT BUTTON ========== #
tk.Button(root, text="Predict Salary", font=("Arial", 12, "bold"),
          bg="#007acc", fg="white", padx=20, pady=8, command=predict_salary).pack(pady=10)

# ========== MAIN LOOP ========== #
root.mainloop()
