# Databricks notebook source
# MAGIC %md
# MAGIC # Ensemble Model

# COMMAND ----------

"""
: 1b27fecd743e47da828bf889547d69ab
 
"""

import mlflow
from pyspark.sql.functions import col, when, lit
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
from sklearn.metrics import classification_report

# ==============================================================================
# 1. SETUP AND DATA LOADING
# ==============================================================================
section = "02"
number = "01"
folder_path = f"dbfs:/student-groups/Group_{section}_{number}/checkpoints"

print(f"Loading data from: {folder_path}/transformed_60m_reduced_features.parquet")

# Load the fully preprocessed data
try:
    df_total = spark.read.parquet(f"{folder_path}/transformed_60m_reduced_features.parquet")
    print("Preprocessed data loaded successfully.")
except Exception as e:
    print(f"Error loading data: {e}")
    print("Please ensure the data preprocessing notebook has been run successfully.")

# Isolate the hold-out test set (2019 data)
test_df = df_total.filter(col("Year") == 2019).cache()
print(f"Test set (Year 2019) contains {test_df.count()} records.")


# ==============================================================================
# 2. LOAD TRAINED MODELS FROM MLFLOW
# ==============================================================================
LOGREG_RUN_ID = "f8e1e9de37694bd4a9eeedf4e98cd773" # flight-delay-logreg-cv-yr
MLP_RUN_ID = "ffc4e594d2244185852ef830864dc68f" # flight-delay-mlpc-cv-yr

# Construct the full model URI path
logreg_model_uri = f"runs:/{LOGREG_RUN_ID}/final-model"
mlp_model_uri = f"runs:/{MLP_RUN_ID}/final-model"

print("Loading models...")
try:
    logreg_model = mlflow.spark.load_model(logreg_model_uri)
    print("Advanced Logistic Regression model loaded.")
    mlp_model = mlflow.spark.load_model(mlp_model_uri)
    print("MLP model loaded.")
except Exception as e:
    print(f"Error loading a model: {e}")
    print("Please verify your Run IDs and ensure the models were logged correctly.")

# ==============================================================================
# 3. GENERATE PREDICTIONS FROM BASE MODELS
# ==============================================================================
print("\nGenerating predictions from base models on the test set...")

# Generate predictions for each model
logreg_preds = logreg_model.transform(test_df).select("departure_datetime", "Tail_Number", "Flight_Number_Reporting_Airline", "DepDel15", col("prediction").alias("logreg_prediction"))
mlp_preds = mlp_model.transform(test_df).select("departure_datetime", "Tail_Number", "Flight_Number_Reporting_Airline", col("prediction").alias("mlp_prediction"))

print("Predictions generated.")

# ==============================================================================
# 4. COMBINE PREDICTIONS AND IMPLEMENT VOTING LOGIC
# ==============================================================================
print("\nCombining predictions for ensemble model...")

# Join the predictions into a single DataFrame
join_keys = ["departure_datetime", "Tail_Number", "Flight_Number_Reporting_Airline"]
ensemble_df = logreg_preds.join(mlp_preds, join_keys, "inner")

# Implement a simple majority vote (hard voting)
num_models = 2
vote_threshold = num_models / 2.0

ensemble_df = ensemble_df.withColumn(
    "ensemble_prediction",
    when(
        (col("logreg_prediction") + col("mlp_prediction")) >= lit(vote_threshold),
        1.0
    ).otherwise(0.0)
)

print("Ensemble predictions calculated.")
ensemble_df.select("DepDel15", "logreg_prediction", "mlp_prediction", "ensemble_prediction").show(10)

# ==============================================================================
# 5. EVALUATE THE ENSEMBLE MODEL
# ==============================================================================
print("\nEvaluating ensemble model performance...")

results_df = ensemble_df.select(col("DepDel15").alias("label"), col("ensemble_prediction").alias("prediction"))
results_pd = results_df.toPandas()

y_true = results_pd['label']
y_pred = results_pd['prediction']

# Use scikit-learn for a detailed classification report
report = classification_report(y_true, y_pred, target_names=['Not Delayed', 'Delayed'])
print("\nClassification Report for Ensemble Model:\n")
print(report)

# Use PySpark Evaluator for a high-level evaluation
evaluator_f1 = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
evaluator_accuracy = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
evaluator_precision = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedPrecision")
evaluator_recall = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedRecall")

f1 = evaluator_f1.evaluate(results_df)
accuracy = evaluator_accuracy.evaluate(results_df)
precision = evaluator_precision.evaluate(results_df)
recall = evaluator_recall.evaluate(results_df)

print("\n--- Summary Metrics for Ensemble Model ---")
print(f"Accuracy: {accuracy:.3f}")
print(f"F1 Score (Weighted): {f1:.3f}")
print(f"Precision (Weighted): {precision:.3f}")
print(f"Recall (Weighted): {recall:.3f}")
print("----------------------------------------")



# COMMAND ----------

