import pandas as pd
import joblib
import logging
import os

# Set up logging to see errors in Airflow task logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def prepare_and_predict(input_file="sample_student_data.csv", 
                        model_file="student_model.sav", 
                        output_file="output_predictions.csv"):
    try:
        # 1. Validate Input File Existence
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input data not found at: {input_file}")
        
        # 2. Load Data
        logger.info(f"Loading data from {input_file}...")
        df = pd.read_csv(input_file)
        
        if df.empty:
            raise ValueError("The input CSV file is empty.")

        # 3. Data Transformation & Validation
        logger.info("Performing data transformations...")
        required_cols = ['hours_studied', 'prev_grade']
        for col in required_cols:
            if col not in df.columns:
                raise KeyError(f"Missing required column: {col}")

        df['hours_studied'] = df['hours_studied'].fillna(df['hours_studied'].mean())
        df['is_high_effort'] = (df['hours_studied'] > 7).astype(int)
        
        # 4. Load Model
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"Model file not found at: {model_file}")
            
        logger.info(f"Loading model from {model_file}...")
        model = joblib.load(model_file)
        
        # 5. Prediction
        logger.info("Running model inference...")
        features = df[required_cols]
        df['predicted_score'] = model.predict(features)
        
        # 6. Save Results
        df.to_csv(output_file, index=False)
        logger.info(f"Predictions saved successfully to {output_file}")
        
        return f"Successfully processed {len(df)} records."

    except FileNotFoundError as fnf_error:
        logger.error(f"File Error: {fnf_error}")
        raise # Re-raise so Airflow marks the task as FAILED
    except KeyError as k_error:
        logger.error(f"Schema Error: {k_error}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error occurred: {e}")
        raise

if __name__ == "__main__":
    prepare_and_predict()