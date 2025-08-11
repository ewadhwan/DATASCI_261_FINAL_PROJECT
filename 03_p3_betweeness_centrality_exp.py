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

otpw_60m = spark.read.parquet(f"{folder_path}/07_final_features/")

# COMMAND ----------

from pyspark.sql.functions import col
train_df_unscaled_imbalanced = otpw_60m.filter((col("Year") >= 2015) & (col("Year") < 2019))
test_df_unscaled = otpw_60m.filter(col("Year") == 2019)

# COMMAND ----------

import networkx as nx
import matplotlib.pyplot as plt

# Sample Spark DataFrame
flights_df = train_df_unscaled_imbalanced.select("Origin", "Dest").dropna().distinct()

# Convert to Pandas for NetworkX
flights_pd = flights_df.toPandas()

# Create directed graph
G = nx.DiGraph()

# Add edges from flight data
#G.add_edges_from(flights_pd.values)

# Compute betweenness centrality
#bc = nx.betweenness_centrality(G)

# COMMAND ----------

bc

# COMMAND ----------

# Add edges from flight data
G.add_edges_from(flights_pd.values)

# Compute betweenness centrality
deg = nx.degree_centrality(G)

# Add centrality as node attribute
nx.set_node_attributes(G, deg, "degree")

# Get top 10 airports by centrality
top_deg = sorted(deg.items(), key=lambda x: -x[1])[:10]
print("Top 10 Airports by Degree Centrality:")
for airport, score in top_deg:
    print(f"{airport}: {score:.4f}")

# Optional: Draw the graph (small subset recommended)
subgraph_nodes = [x[0] for x in top_deg]
subG = G.subgraph(subgraph_nodes)

plt.figure(figsize=(20, 6))
pos = nx.spring_layout(subG)
node_sizes = [5000 * deg[n] for n in subG.nodes]
nx.draw(subG, pos, with_labels=True, node_size=node_sizes, node_color='lightblue', edge_color='gray')
plt.title("Top 10 Airports by Degree Centrality")
plt.show()


# COMMAND ----------

from pyspark.sql.functions import col, udf
from pyspark.sql.types import StructType, StructField, IntegerType, FloatType
sc = spark.sparkContext

#Broadcast the dictionary
broadcasted_dict = sc.broadcast(bc)
# Define a UDF to map the dictionary values
def get_betweenness_centrality(origin_airport_id):
    return broadcasted_dict.value.get(origin_airport_id, None)

# Register the UDF
get_betweenness_centrality_udf = udf(get_betweenness_centrality, FloatType())

# Apply the UDF to the DataFrames
train_df_unscaled_imbalanced = train_df_unscaled_imbalanced.withColumn(
    "betweeness_centrality", 
    get_betweenness_centrality_udf(col("OriginAirportID"))
)

test_df_unscaled = test_df_unscaled.withColumn(
    "betweeness_centrality", 
    get_betweenness_centrality_udf(col("OriginAirportID"))
)

display(train_df_unscaled_imbalanced)
display(test_df_unscaled)

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

(train_df_unscaled.groupBy("DepDel15").count().withColumn("ratio", col("count") / train_df_unscaled.count())).select("DepDel15", "ratio").show()

# COMMAND ----------

#impute missing values. Use meadian imputation for numerical columns and unknown for categorical columns
from pyspark.ml.feature import Imputer
#impute with median for numerical columns
imputer1 = Imputer(strategy="median", inputCols=["Prev_TaxiIn", "Prev_ArrDelay", "Prev_ArrTime", "Turnaround_Time", "short_term_ontime_arrival_pct", "short_term_ontime_departure_pct", 'Hour', 'betweeness_centrality'], outputCols=["Prev_TaxiIn_imputed","Prev_ArrDelay_imputed", "Prev_ArrTime_imputed", "Turnaround_Time_imputed", "short_term_ontime_arrival_pct_imputed", "short_term_ontime_departure_pct_imputed",'Hour_imputed', 'betweeness_centrality_imputed'])

#fill with mode for the boolean categorical columns
#imputer2 = Imputer(strategy="mode", inputCols=["IsWeekend", "IsHolidayMonth"], outputCols=["IsWeekend_imputed","IsHolidayMonth_imputed"])


# COMMAND ----------

#min-max scaling for numerical features
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import MinMaxScaler

# Combine the numerical columns into a single vector column
assembler = VectorAssembler(inputCols=["Prev_TaxiIn_imputed","Prev_ArrDelay_imputed", "Prev_ArrTime_imputed", "Turnaround_Time_imputed", "short_term_ontime_arrival_pct_imputed", "short_term_ontime_departure_pct_imputed",'Hour_imputed', 'betweeness_centrality_imputed'], outputCol="features")
scaler = MinMaxScaler(inputCol="features", outputCol="scaled_features")

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

#import log regression fitting functions from 03_p2_eda_vizualizations_and_ml_functions.py
%run "./Phase 2/03_p2_eda_vizualizations_and_ml_functions"


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
    run_name="blocked_ts_mlpc_cv_yr_2"
)