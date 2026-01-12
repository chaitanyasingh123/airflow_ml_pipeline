import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression

# 1. Create Sample Data Records (for initial training)
data = {
    'hours_studied': [2, 3, 4, 5, 6, 7, 8, 9, 10],
    'prev_grade': [60, 65, 70, 75, 80, 85, 90, 92, 95],
    'target_score': [65, 68, 72, 78, 82, 88, 92, 95, 98]
}
df = pd.DataFrame(data)

# 2. Save sample records as a CSV for Airflow to "pull"
df.to_csv('sample_student_data.csv', index=False)
print("✅ Created sample_student_data.csv")

# 3. Train a Mock Model
X = df[['hours_studied', 'prev_grade']]
y = df['target_score']
model = LinearRegression().fit(X, y)

# 4. Save as .sav file (Joblib is preferred for models with arrays)
joblib.dump(model, 'student_model.sav')
print("✅ Created student_model.sav")