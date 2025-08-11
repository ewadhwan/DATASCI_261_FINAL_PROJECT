# Databricks notebook source
# Create folder (RUN THIS ONCE)
section = "02"
number = "01"
folder_path = f"dbfs:/student-groups/Group_{section}_{number}/checkpoints"

# Check if folder exists
try:
    dbutils.fs.ls(folder_path)
    print(f"Folder already exists: {folder_path}")
except Exception as e:
    # If folder doesn't exist, create it
    dbutils.fs.mkdirs(folder_path)
    print(f"Created folder: {folder_path}")

# EXAMPLE USAGE - Save df_weather as a parquet file
#df_weather.write.parquet(f"{folder_path}/df_weather.parquet")

# COMMAND ----------

# List files in the specified directory
files = dbutils.fs.ls("dbfs:/student-groups/Group_02_01/checkpoints")
display(files)

# COMMAND ----------

otpw_60m = spark.read.parquet(f"{folder_path}/07_final_features/")
display(otpw_60m)

# COMMAND ----------

row_count = otpw_60m.count()
column_count = len(otpw_60m.columns)
print(f"Shape: ({row_count}, {column_count})")

# COMMAND ----------

from pyspark.sql.functions import isnan, when, count, col, isnull
display(otpw_60m.select([count(when(col(c).isNull(), c)).alias(c) for c in otpw_60m.columns]))


# COMMAND ----------

#drop rows with crucial identifying data missing
from pyspark.sql.functions import col, isnan, when, count

required_cols = ["DepDel15",'Tail_Number',"departure_datetime", "Flight_Number_Reporting_Airline"] 

otpw_60m_clean = otpw_60m.dropna(subset=required_cols)
otpw_60m_clean.select([
    count(when(col(c).isNull(), c)).alias(f"{c}_missing")
    for c in required_cols
]).show()

row_count = otpw_60m_clean.count()
column_count = len(otpw_60m_clean.columns)
print(f"Shape: ({row_count}, {column_count})")

# COMMAND ----------

#not necessarily applicable for the 1 year data, because there were no nulls, but calculate Year, Quarter, Month, and DayofWeek from FlightDate if necessary


# COMMAND ----------

#drop duplicates
otpw_60m_clean = otpw_60m_clean.dropDuplicates(subset=required_cols)
row_count = otpw_60m_clean.count()
column_count = len(otpw_60m_clean.columns)
print(f"Shape: ({row_count}, {column_count})")

# COMMAND ----------

#filter out cancelled data

otpw_60m_clean = otpw_60m_clean.filter((col("CANCELLED") != 1) | col("CANCELLED").isNull())
otpw_60m_clean.count()

# COMMAND ----------

# drop columns where more than 50% of the entries are null

from pyspark.sql.functions import col, count, when
# Create list to hold null stats
drop_cols = []
# Loop through columns
total_rows = otpw_60m_clean.count()
null_threshold = 0.5
for c in otpw_60m_clean.columns:
    # Use only isNull() for all data types (safe for strings, dates, numerics)
    null_count = otpw_60m_clean.filter(col(c).isNull()).count()
    null_pct = null_count / total_rows
    if null_pct >= null_threshold:
        drop_cols.append(c)

otpw_60m_filtered = otpw_60m_clean.drop(*drop_cols)

row_count = otpw_60m_filtered.count()
column_count = len(otpw_60m_filtered.columns)
print(f"Shape: ({row_count}, {column_count})")

# COMMAND ----------

display(otpw_60m_filtered)

# COMMAND ----------

otpw_60m_filtered.columns

# COMMAND ----------

#drop redundant and leaky columns. Keep columns used for data splitting/time series cross evaulation
keep_columns = ['DepDel15','Year', 'Quarter', 'Month', 'DayOfWeek','departure_datetime', 'Reporting_Airline', 'Flight_Number_Reporting_Airline', 'Tail_Number', 'OriginAirportID', 'DestAirportID', 'Distance', 'Prev_TaxiIn', 'Prev_TaxiOut', 'Prev_ArrDelay', 'Prev_ArrTime', 'Prev_FlightDate', 'Turnaround_Time', 'short_term_ontime_arrival_pct', 'short_term_ontime_departure_pct','IsWeekend', 'Hour', 'SeasonQuarter', 'IsHolidayMonth','HourlyWindSpeed', 'HourlyVisibility', 'HourlyWetBulbTemperature', 'Oncoming_flights']
reduced_features = otpw_60m_filtered.select(keep_columns)
row_count = reduced_features.count()
column_count = len(reduced_features.columns)
print(f"Shape: ({row_count}, {column_count})")

# COMMAND ----------

#fill UNK for categorical features that are null
import pyspark.sql.functions as F
reduced_features = reduced_features.fillna("UNK", subset=["Reporting_Airline", "OriginAirportID", "DestAirportID","SeasonQuarter"])


# COMMAND ----------

#delete the checkpoint if it already exists
dbutils.fs.rm(f"{folder_path}/preprocessed_60m.parquet", recurse=True)
#checkpoint the dataframe
reduced_features.write.parquet(f"{folder_path}/preprocessed_60m.parquet")

# COMMAND ----------

reduced_features=spark.read.parquet(f"{folder_path}/preprocessed_60m.parquet")

# COMMAND ----------

reduced_features.select("Year").distinct().show()

# COMMAND ----------

#split into years 2015-2018 for training data and 2019 for testing data
from pyspark.sql.functions import col
train_df_unscaled_imbalanced = reduced_features.filter((col("Year") >= 2015) & (col("Year") < 2019))
test_df_unscaled = reduced_features.filter(col("Year") == 2019)




# COMMAND ----------

#this data is highly unbalanced
# (train_df_unscaled_imbalanced.groupBy("DepDel15").count().withColumn("ratio", col("count") / train_df_unscaled.count())).select("DepDel15", "ratio").show()
(train_df_unscaled_imbalanced.groupBy("DepDel15").count().withColumn("ratio", col("count") / train_df_unscaled_imbalanced.count())).select("DepDel15", "ratio").show()

# COMMAND ----------

#look at the class imbalance for each training block
blocks=[2015, 2016, 2017, 2018]
for block in blocks:
    section = train_df_unscaled_imbalanced.filter(col("Year") == block)
    section.groupBy("DepDel15").count().withColumn("ratio", col("count") / section.count()).select("DepDel15", "ratio").show()


# COMMAND ----------

#perform undersampling of the majority class
#all of the delayed flights will be kept in the final training dataframe
train_df_unscaled = train_df_unscaled_imbalanced.filter(col("DepDel15") == 1)
#for each training block, filter only the ontime flights, then randomly sample the same number of late flights as the ontime flights
blocks=[2015, 2016, 2017, 2018]
for block in blocks:
    #filter for the flights in each training block for time series cross validation
    section = train_df_unscaled_imbalanced.filter(col("Year") == block)
    #calculate the sampling fraction
    delayed = section.filter(col("DepDel15") == 1)
    on_time = section.filter(col("DepDel15") == 0)
    sample_fraction = delayed.count() / on_time.count()
    #add the randomly sampled ontime flights for the block to the dataframe containing all the delayed flights
    train_df_unscaled = train_df_unscaled.union(on_time.sample(withReplacement=False, fraction=sample_fraction, seed=42))

# COMMAND ----------

#verify the classes are now more balanced
(train_df_unscaled.groupBy("DepDel15").count().withColumn("ratio", col("count") / train_df_unscaled.count())).select("DepDel15", "ratio").show()

# COMMAND ----------

blocks=[2015, 2016, 2017, 2018]
for block in blocks:
    section = train_df_unscaled.filter(col("Year") == block)
    section.groupBy("DepDel15").count().withColumn("ratio", col("count") / section.count()).select("DepDel15", "ratio").show()

# COMMAND ----------

#impute missing values. Use meadian imputation for numerical columns and unknown for categorical columns
from pyspark.ml.feature import Imputer
#impute with median for numerical columns
imputer1 = Imputer(strategy="median", inputCols=["Prev_TaxiIn", "Prev_ArrDelay", "Prev_ArrTime", "Turnaround_Time", "short_term_ontime_arrival_pct", "short_term_ontime_departure_pct", 'Hour', 'Oncoming_flights'], outputCols=["Prev_TaxiIn_imputed","Prev_ArrDelay_imputed", "Prev_ArrTime_imputed", "Turnaround_Time_imputed", "short_term_ontime_arrival_pct_imputed", "short_term_ontime_departure_pct_imputed",'Hour_imputed', 'Oncoming_flights_imputed'])

#fill with mode for the boolean categorical columns
#imputer2 = Imputer(strategy="mode", inputCols=["IsWeekend", "IsHolidayMonth"], outputCols=["IsWeekend_imputed","IsHolidayMonth_imputed"])


# COMMAND ----------

#min-max scaling for numerical features
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import MinMaxScaler

# Combine the numerical columns into a single vector column
assembler = VectorAssembler(inputCols=["Prev_TaxiIn_imputed","Prev_ArrDelay_imputed", "Prev_ArrTime_imputed", "Turnaround_Time_imputed", "short_term_ontime_arrival_pct_imputed", "short_term_ontime_departure_pct_imputed",'Hour_imputed', 'Oncoming_flights_imputed'], outputCol="features")
scaler = MinMaxScaler(inputCol="features", outputCol="scaled_features")



# COMMAND ----------

#hash the categorical features
#from pyspark.ml.feature import FeatureHasher
#input_cols = ["Reporting_Airline", "Flight_Number_Reporting_Airline", "Tail_Number", "OriginAirportID", "DestAirportID", "SeasonQuarter"]
#hasher = FeatureHasher(inputCols=input_cols, outputCol="categorical_features", numFeatures=1000)


# COMMAND ----------

# #pipe through the scaling and hashing steps
# from pyspark.ml import Pipeline
# #train the pipeline on all the data
# pipeline = Pipeline(stages=[imputer1, imputer2, assembler, scaler, hasher])
# pipe = pipeline.fit(train_df_unscaled_imbalanced)
# #apply the pipeline to the balanced training data and the test data
# train_df = pipe.transform(train_df_unscaled)
# test_df = pipe.transform(test_df_unscaled)
# #append test_df to the end of train_df for compatiblity with the model fit function
# df_total = train_df.union(test_df)

# COMMAND ----------

#new pipeline for the reduced feature set - no hashing and only one imputer
from pyspark.ml import Pipeline
#train the pipeline on all the data
pipeline = Pipeline(stages=[imputer1, assembler, scaler])
pipe = pipeline.fit(train_df_unscaled_imbalanced)
#apply the pipeline to the balanced training data and the test data
train_df = pipe.transform(train_df_unscaled)
test_df = pipe.transform(test_df_unscaled)
#append test_df to the end of train_df for compatiblity with the model fit function
df_total = train_df.union(test_df)

# COMMAND ----------

#delete the checkpoint if it already exists
dbutils.fs.rm(f"{folder_path}/transformed_60m.parquet", recurse=True)
#checkpoint the dataframe
df_total.write.parquet(f"{folder_path}/transformed_60m.parquet")

# COMMAND ----------

#checkpoint the dataframe with the smaller feature set
dbutils.fs.rm(f"{folder_path}/transformed_60m_reduced_features.parquet", recurse=True)
#checkpoint the dataframe
df_total.write.parquet(f"{folder_path}/transformed_60m_reduced_features.parquet")

# COMMAND ----------

df_total = spark.read.parquet(f"{folder_path}/transformed_60m_reduced_features.parquet")

# COMMAND ----------

#import log regression fitting functions from 03_p2_eda_vizualizations_and_ml_functions.py
%run "./Phase 2/03_p2_eda_vizualizations_and_ml_functions"


# COMMAND ----------

from pyspark.sql.functions import col
#fit the logistic regression model with time series cross validation folding
feature_cols = ['Month', 'scaled_features']
if "features" in train_df.columns:
    train_df = train_df.drop("features")
if "features" in test_df.columns:
    test_df = test_df.drop("features")

run_logreg_baseline(
    df_train=train_df,
    df_test=test_df,
    feature_cols=feature_cols,
    label_col="DepDel15",
    split_col="Month",
    train_months=range(1, 10),
    test_months=range(10, 13),
    experiment_name="/Users/ronghuang0604@berkeley.edu/simple-flight-delay-logreg-cv",  # set your Databricks user or shared experiment path
    run_name="simple_logreg_baseline"
)

# COMMAND ----------

from pyspark.sql.functions import col
#fit the logistic regression model with time series cross validation folding
feature_cols = ['Month', 'scaled_features']
run_blocked_ts_logreg_cv_yr(
    df=df_total,
    feature_cols=feature_cols,
    label_col="DepDel15",
    year_col="Year",
    blocks=[2015, 2016, 2017, 2018],
    test_year=2019,
    param_grid=[
        {"regParam": 0.01, "elasticNetParam": 0.0},   # Ridge
        {"regParam": 0.01, "elasticNetParam": 0.5},   # Elastic Net
        {"regParam": 0.1,  "elasticNetParam": 0.0},   # Ridge
        {"regParam": 0.1,  "elasticNetParam": 0.5},   # Elastic Net
        {"regParam": 1.0,  "elasticNetParam": 0.0},   # Ridge
        {"regParam": 1.0,  "elasticNetParam": 0.5},   # Elastic Net
    ],
    experiment_name="/Users/ronghuang0604@berkeley.edu/flight-delay-logreg-cv-yr",  # change to your workspace path
    run_name="blocked_ts_logreg_cv_yr"
)

# COMMAND ----------

#run basic training function for MLP model with smaller feature set

from pyspark.sql.functions import col

#fit the MLP model with time series cross validation folding
feature_cols = ['Month', 'scaled_features']

# # No need to assemble or get input_size here—the function does it!

param_grid = [
    {"hidden_layers": [10, 2], "maxIter": 50, "stepSize": 0.03},              # One hidden layer, 10 neurons
    {"hidden_layers": [20, 10, 2], "maxIter": 80, "stepSize": 0.05},          # Two hidden layers, 20 and 10 neurons
    {"hidden_layers": [50, 25, 10, 2], "maxIter": 100, "stepSize": 0.1},      # Three hidden layers, 50, 25, 10 neurons
]

run_blocked_ts_mlpc_cv_yr_basic(
    df=df_total,
    feature_cols=feature_cols,
    label_col="DepDel15",
    year_col="Year",
    blocks=[2015, 2016, 2017, 2018],
    test_year=2019,
    param_grid=param_grid,
    experiment_name="/Users/ronghuang0604@berkeley.edu/flight-delay-mlpc-cv-yr-basic",  # change to your workspace path
    run_name="blocked_ts_mlpc_cv_yr_1"
)

# COMMAND ----------

from pyspark.sql.functions import col

#fit the MLP model with time series cross validation folding
feature_cols = ['Month', 'scaled_features']

param_grid = [
    {"hidden_layers": [10, 2], "maxIter": 50, "stepSize": 0.03},
    {"hidden_layers": [20, 10, 2], "maxIter": 80, "stepSize": 0.05},
    {"hidden_layers": [50, 25, 10, 2], "maxIter": 100, "stepSize": 0.1},
]

final_model, gridRes_df = train_blocked_ts_mlpc_cv_yr(
    df=df_total,
    feature_cols=feature_cols,
    label_col="DepDel15",
    year_col="Year",
    month_col="Month",
    blocks=[2015, 2016, 2017, 2018],
    test_year=2019,
    param_grid=param_grid,
    experiment_name="/Users/ronghuang0604@berkeley.edu/flight-delay-mlpc-cv-yr",
    run_name="blocked_ts_mlpc_cv_yr"
)


# COMMAND ----------

#load the model

import mlflow
logged_model = 'runs:/dfaab1d931244f729265699a95291143/final-model'

# Load model
loaded_model = mlflow.spark.load_model(logged_model)


# COMMAND ----------

from pyspark.sql.functions import col
test_df = df_total.filter(col("Year") == 2019)

# COMMAND ----------


# find best threshold by F1
# best_threshold, threshold_results = find_best_threshold_mlpc(loaded_model, test_df, "DepDel15", metric="f1")
best_threshold, threshold_results = find_best_threshold_mlpc(final_model, test_df, "DepDel15", metric="f1")

# post process and report with the best threshold
postprocess_blocked_ts_mlpc(final_model, test_df, "DepDel15", threshold_value=best_threshold)

# COMMAND ----------

chm_cols = ["DepDel15", "Prev_TaxiIn_imputed","Prev_ArrDelay_imputed", "Prev_ArrTime_imputed", "Turnaround_Time_imputed", "short_term_ontime_arrival_pct_imputed", "short_term_ontime_departure_pct_imputed",'Hour_imputed', 'Oncoming_flights_imputed']
plot_correlation_heatmap(
    df_total, 
    chm_cols, 
    sample_size=5000, 
    title="Correlation Matrix Heatmap"
)

# COMMAND ----------

gridRes_df.display()

# COMMAND ----------

#run GBTree classifier to get a sense of feature importance
features_gbt = ['Month', 'DayOfWeek', "Distance_imputed","Prev_TaxiIn_imputed", "Prev_TaxiOut_imputed", "Prev_ArrDelay_imputed", "Prev_ArrTime_imputed", "Turnaround_Time_imputed", "short_term_ontime_arrival_pct_imputed", "short_term_ontime_departure_pct_imputed", "IsWeekend_imputed", "Hour_imputed", "IsHolidayMonth_imputed",'HourlyWindSpeed_imputed', 'HourlyVisibility_imputed', 'HourlyWetBulbTemperature_imputed', 'categorical_features']
plot_gbt_feature_importances(
    df_total,
    features_gbt,
    label_col="DepDel15",
    maxDepth=5,
    maxIter=20,
    stepSize=0.1,
    subsamplingRate=1.0,
    featureSubsetStrategy="all",
    seed=31,
    top_n=20,
    figsize=(12,6)
)

# COMMAND ----------

# MAGIC %md
# MAGIC # Neural Network with Tensorflow

# COMMAND ----------

# Create folder (RUN THIS ONCE)
section = "02"
number = "01"
folder_path = f"dbfs:/student-groups/Group_{section}_{number}/checkpoints"
df_total = spark.read.parquet(f"{folder_path}/transformed_60m_reduced_features.parquet")
train_blocks=[2015, 2016, 2017, 2018]
validation_year = 2018
test_year=2019

feature_cols = ['Month', 'DayOfWeek', "Distance_imputed","Prev_TaxiIn_imputed", "Prev_TaxiOut_imputed", "Prev_ArrDelay_imputed", "Prev_ArrTime_imputed", "Turnaround_Time_imputed", "short_term_ontime_arrival_pct_imputed", "short_term_ontime_departure_pct_imputed", "IsWeekend_imputed", "Hour_imputed", "IsHolidayMonth_imputed",'HourlyWindSpeed_imputed', 'HourlyVisibility_imputed', 'HourlyWetBulbTemperature_imputed']
label_col="DepDel15",
year_col="Year"   

display(df_total)

# COMMAND ----------

from pyspark.sql.functions import col
from pyspark.ml.functions import vector_to_array

# Convert vector column to array so it can be used in Pandas
df_array = df_total.withColumn("features_array", vector_to_array("scaled_features"))

# Select relevant columns
df_filtered = df_array.select("features_array", "DepDel15", "YEAR")

# COMMAND ----------

import numpy as np

df_train = df_filtered.filter(col("YEAR").between(2015, 2017)).toPandas()
df_val = df_filtered.filter(col("YEAR") == 2018).toPandas()
df_test = df_filtered.filter(col("YEAR") == 2019).toPandas()

#Remove this after GPU assignment
df_train =df_train.sample(frac=0.1, random_state=12)
df_val = df_val.sample(frac=0.1, random_state=12)
df_test = df_test.sample(frac=0.1, random_state=12)

# Extract features and labels
X_train = np.vstack(df_train["features_array"].values)
y_train = df_train["DepDel15"].values

X_val = np.vstack(df_val["features_array"].values)
y_val = df_val["DepDel15"].values

X_test = np.vstack(df_test["features_array"].values)
y_test = df_test["DepDel15"].values

# COMMAND ----------

print(X_train.shape, y_train.shape)
print(X_val.shape, y_val.shape)
print(X_test.shape, y_test.shape)

# COMMAND ----------

from hyperopt import hp

search_space = {
    "units_1": hp.choice("units_1", [64, 128, 256]),
    "units_2": hp.choice("units_2", [32, 64]),
    "dropout_rate": hp.uniform("dropout_rate", 0.1, 0.5),
    "learning_rate": hp.loguniform("learning_rate", -5, -2),  # ~[0.006, 0.135]
    "batch_size": hp.choice("batch_size", [32, 64, 128]),
    "epochs": hp.choice("epochs", [10, 20, 30])
}


# COMMAND ----------

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import optimizers
def f1_score(y_true, y_pred):
    y_pred_binary = K.round(y_pred)
    
    tp = K.sum(K.cast(y_true * y_pred_binary, 'float'), axis=0)
    fp = K.sum(K.cast((1 - y_true) * y_pred_binary, 'float'), axis=0)
    fn = K.sum(K.cast(y_true * (1 - y_pred_binary), 'float'), axis=0)

    precision = tp / (tp + fp + K.epsilon())
    recall = tp / (tp + fn + K.epsilon())

    f1 = 2 * precision * recall / (precision + recall + K.epsilon())
    return K.mean(f1)

# COMMAND ----------

from hyperopt import STATUS_OK
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam,SGD
from tensorflow.keras.metrics import Accuracy
from sklearn.metrics import classification_report, f1_score, mean_squared_error, mean_absolute_error


import mlflow.keras

def objective(params):
    with mlflow.start_run(nested=True):
        model = Sequential()
        model.add(Dense(params['units_1'], activation='relu', input_shape=(X_train.shape[1],)))
        model.add(Dropout(params['dropout_rate']))
        model.add(Dense(params['units_2'], activation='relu'))
        model.add(Dense(1, activation='sigmoid'))

        optimizer = SGD(learning_rate=params["learning_rate"])
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

        model.fit(X_train, y_train,
                  validation_data=(X_val, y_val),
                  epochs=params["epochs"],
                  batch_size=params["batch_size"],
                  verbose=0)

        # Predict
        y_pred_probs = model.predict(X_val).ravel()
        y_pred = (y_pred_probs >= 0.3).astype(int)

        #Calculate Metrics
        f1 = f1_score(y_val, y_pred)
        report = classification_report(y_val, y_pred, output_dict=True)
        mse = mean_squared_error(y_val, y_pred_probs)
        mae = mean_absolute_error(y_val, y_pred_probs)
        recall_delayed = report['1.0']['recall']

        # Log all
        mlflow.log_params(params)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("f1_delayed", report['1.0']['f1-score'])
        mlflow.log_metric("accuracy", report['accuracy'])
        mlflow.log_metric("recall_delayed", report['1.0']['recall'])
        mlflow.log_metric("precision_delayed", report['1.0']['precision'])
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("mae", mae)

        # Save model
        mlflow.keras.log_model(model, artifact_path="model")
       
        return {"recall": -recall_delayed, "status": STATUS_OK}
        # loss = history.history['val_loss'][-1]  # Example of returning validation loss
        # return loss


# COMMAND ----------

from hyperopt import fmin, tpe, Trials

# mlflow.create_experiment("/Users/ronghuang0604@berkeley.edu/FlightDelay_NN_Hyperopt")
mlflow.end_run()
mlflow.set_experiment("/Users/ronghuang0604@berkeley.edu/FlightDelay_NN_Hyperopt_0.3B")

with mlflow.start_run(run_name="Hyperopt_Tuning_0.3_Threshold"):
    trials = Trials()
    best_result = fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=10,  
        trials=trials
    )

    mlflow.log_params(best_result)


# COMMAND ----------

best_result

# COMMAND ----------

# Save best model
best_params = best_result
final_model = Sequential()
final_model.add(Dense(int(best_params['units_1']), input_shape=(X_train.shape[1],), activation='relu'))
final_model.add(Dropout(best_params['dropout_rate']))
final_model.add(Dense(int(best_params['units_2']), activation='relu'))
final_model.add(Dropout(best_params['dropout_rate']))
final_model.add(Dense(1, activation='sigmoid'))

final_model.compile(optimizer=Adam(learning_rate=best_params['learning_rate']),
                    loss='binary_crossentropy',
                    metrics=['accuracy'])

# Train on train + validation for best results
X_train_full = np.vstack((X_train, X_val))
y_train_full = np.concatenate((y_train, y_val))

final_model.fit(X_train_full, y_train_full, epochs=int(best_params['epochs']),
                batch_size=int(best_params['batch_size']), verbose=1)

# Save locally
final_model.save("/Users/ronghuang0604@berkeley.edu/best_nn_model.h5")

# Or log in MLflow
with mlflow.start_run(run_name="Best_Recall_Model"):
    mlflow.keras.log_model(final_model, "nn_model")
    mlflow.log_params(best_params)

# COMMAND ----------

#Test evaluation

# run_id = "19a39cae933148f09a0e3efc3d057538"  # Replace with your run ID
# model_uri = f"runs:/{run_id}/youthful-fly-301"  # 'nn_model' is the name you used in mlflow.keras.log_model()
model_uri="dbfs:/databricks/mlflow-tracking/2431760182593315/19a39cae933148f09a0e3efc3d057538/artifacts/model"

# Load the model
model = mlflow.keras.load_model(model_uri)

y_test_pred_probs = model.predict(X_test).ravel()
y_test_pred = (y_test_pred_probs >= 0.5).astype(int)

test_report = classification_report(y_test, y_test_pred, output_dict=True)
test_mse = mean_squared_error(y_test, y_test_pred_probs)
test_mae = mean_absolute_error(y_test, y_test_pred_probs)

pos_class_test = str(sorted(set(y_test))[-1])  # positive class key dynamically

mlflow.log_metric("test_accuracy", test_report['accuracy'])
mlflow.log_metric("test_precision_delayed", test_report[pos_class_test]['precision'])
mlflow.log_metric("test_recall_delayed", test_report[pos_class_test]['recall'])
mlflow.log_metric("test_f1_delayed", test_report[pos_class_test]['f1-score'])
mlflow.log_metric("test_mse", test_mse)
mlflow.log_metric("test_mae", test_mae)

# Log the trained model
# mlflow.keras.log_model(model, "nn_model_eval")