# Databricks notebook source
# MAGIC %md
# MAGIC # NEW DATAFRAME
# MAGIC 5-year of OTPW. (Expected for Phase 3)

# COMMAND ----------

# Listing of data
data_BASE_DIR = "dbfs:/student-groups/Group_02_01/checkpoints/07_final_features/"
display(dbutils.fs.ls(f"{data_BASE_DIR}"))

# COMMAND ----------

# MAGIC %md
# MAGIC # Load Joined Data

# COMMAND ----------

# Create folder
section = "02"
number = "01"
folder_path = f"dbfs:/student-groups/Group_{section}_{number}/"
df_otpw_5y=spark.read.parquet(f"{folder_path}/checkpoints/07_final_features/")

# COMMAND ----------

data_BASE_DIR = "dbfs:/mnt/mids-w261/OTPW_60M_Backup"
display(dbutils.fs.ls(f"{data_BASE_DIR}"))
df_otpw_5y_orig=spark.read.parquet(f"{data_BASE_DIR}")

# COMMAND ----------

# MAGIC %md
# MAGIC # Basic Data Tables

# COMMAND ----------

row_count = df_otpw_5y.count()
col_count = len(df_otpw_5y.columns)
display({"rows": row_count, "columns": col_count})

# COMMAND ----------

display(df_otpw_5y)

# COMMAND ----------

display(df_otpw_5y.select("DepDel15").groupBy("DepDel15").count().orderBy("DepDel15", ascending=False))

# COMMAND ----------


# More detailed duplicate analysis
duplicates_df = df_otpw_5y.groupBy(df.columns).count().filter(col("count") > 1)
print(f"Unique duplicate patterns: {duplicates_df.count()}")

# Check duplicates on specific columns (e.g., flight key)
flight_key_duplicates = df.groupBy("FL_DATE", "OP_UNIQUE_CARRIER", "FL_NUM") \
                         .count().filter(col("count") > 1)

# COMMAND ----------

import matplotlib.pyplot as plt

dep_delay = pd.to_numeric(df_otpw_5y_orig.select("DEP_DELAY").dropna().toPandas()["DEP_DELAY"], errors='coerce')
dep_delay = dep_delay[(dep_delay < 100) & (dep_delay > -50)]

plt.figure(figsize=(10,6))
n, bins, patches = plt.hist(dep_delay, bins=200, color='skyblue', edgecolor='black')
plt.axvline(x=15, color='red', linestyle='--', linewidth=2)
plt.annotate('15 min', xy=(15, max(n)*0.9), xytext=(17, max(n)*0.9),
             arrowprops=dict(facecolor='red', shrink=0.05), color='red')
plt.axvline(x=0, color='green', linestyle='--', linewidth=2)
plt.annotate('0 min', xy=(0, max(n)*0.9), xytext=(5, max(n)*0.9),
             arrowprops=dict(facecolor='green', shrink=0.05), color='green')
plt.xlabel("DEP_DELAY (minutes)")
plt.ylabel("Count")
plt.title("Histogram of DEP_DELAY")

plt.tight_layout()
plt.show()

# COMMAND ----------

dep_delay2 = pd.to_numeric(df_otpw_5y_orig.select("DEP_DELAY").dropna().toPandas()["DEP_DELAY"], errors='coerce')
display(dep_delay2[dep_delay2 >= 15].count())
display("\n")
display(dep_delay2[(dep_delay2 > 0) & (dep_delay2 < 15)].count())
display("\n")
display(dep_delay2[dep_delay2 <= 0].count())

# COMMAND ----------

import matplotlib.pyplot as plt

df_airport_counts = (
    df_otpw_5y_orig
    .groupBy("ORIGIN", "origin_airport_lat", "origin_airport_lon")
    .count()
    .withColumn("origin_airport_lat", df_otpw_5y_orig["origin_airport_lat"].cast("double"))
    .withColumn("origin_airport_lon", df_otpw_5y_orig["origin_airport_lon"].cast("double"))
    .filter("origin_airport_lat IS NOT NULL AND origin_airport_lon IS NOT NULL")
)

pdf = df_airport_counts.toPandas()

plt.figure(figsize=(12,8))
plt.scatter(
    pdf["origin_airport_lon"], 
    pdf["origin_airport_lat"], 
    s=pdf["count"]/10, 
    alpha=0.6, 
    c='blue', 
    edgecolor='k'
)
plt.title("US Airports by Flight Count")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.xlim([-130, -65])
plt.ylim([23, 50])
plt.grid(True)
plt.tight_layout()
plt.show()

# COMMAND ----------

display(df_airport_counts)

# COMMAND ----------

# MAGIC %md
# MAGIC #Univariate Analysis

# COMMAND ----------

# MAGIC %md
# MAGIC ## Counts - rows, columns, duplicates

# COMMAND ----------

row_count = df_otpw_5y.count()
col_count = len(df_otpw_5y.columns)
display({"rows": row_count, "columns": col_count})

# COMMAND ----------

# MAGIC %md
# MAGIC ## Table Newly Joined

# COMMAND ----------

display(df_otpw_5y)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Table OTPW ORIG

# COMMAND ----------

display(df_otpw_5y_orig)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Duplicate check

# COMMAND ----------

# More detailed duplicate analysis
from pyspark.sql.functions import col
duplicates_df = df_otpw_5y_orig.groupBy(df_otpw_5y_orig.columns).count().filter(col("count") > 1)

print(f"Unique duplicate patterns: {duplicates_df.count()}")

# Check duplicates on specific columns (e.g., flight key)
flight_key_duplicates = df_otpw_5y_orig.groupBy("FL_DATE", "DEP_TIME","OP_UNIQUE_CARRIER","OP_CARRIER_FL_NUM" ,"TAIL_NUM","ORIGIN","DEST") \
                         .count().filter(col("count") > 1)
display(flight_key_duplicates)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Null values

# COMMAND ----------

# Summary Statistics

import pandas as pd
import numpy as np
from pyspark.sql.functions import col, sum as spark_sum

# Check for nulls
df_otpw_5y_count = df_otpw_5y.count()
null_counts = df_otpw_5y.select([spark_sum(col(c).isNull().cast("float")).alias(c) for c in df_otpw_5y.columns])

null_counts_df = null_counts.toPandas()
null_counts_df = null_counts_df.transpose()
null_counts_df.columns = ['null_counts']
null_counts_df = null_counts_df.reset_index()
null_counts_df['null_counts'] = pd.to_numeric(null_counts_df['null_counts'], errors='coerce')
null_counts_df['percentage'] = round((null_counts_df['null_counts'] / df_otpw_5y_count) * 100, 2)

# Calculate stats
stats = df_otpw_5y.select([col(c).cast("double").alias(c) for c in df_otpw_5y.columns])
summary = stats.summary("mean", "stddev", "min", "max").toPandas().set_index('summary').transpose()
summary.columns = ['mean', 'stddev', 'min', 'max']
medians = stats.approxQuantile(stats.columns, [0.5], 0.01)
median_series = pd.Series([m[0] if m else np.nan for m in medians], index=stats.columns, name='median')

summary['median'] = median_series
summary = summary.reset_index().rename(columns={'index': 'index'})

# Merge stats
null_counts_df = null_counts_df.merge(summary, left_on='index', right_on='index', how='left')
null_counts_df['mean'] = pd.to_numeric(null_counts_df['mean'], errors='coerce').round(2)
null_counts_df['stddev'] = pd.to_numeric(null_counts_df['stddev'], errors='coerce').round(2)
null_counts_df = null_counts_df[null_counts_df['percentage'] > 0].round(2)
null_counts_df = null_counts_df.sort_values(by=['percentage', 'index'], ascending=[True, True])
display(null_counts_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Box plot for numeric

# COMMAND ----------

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Select numeric columns only for boxplot
datetime_cols=['Year','Quarter','Month','DayofMonth','DayOfWeek','Hour','CRSDepTime','ARR_TIME','DEP_TIME','CRS_ARR_TIME','PrevArrTime']
categorical_cols=['Flight_Number_Reporting_Airline','OriginAirportID','DestAirportID','OriginWac','DestWac','DistanceGroup','CANCELLED','delay_flag','cancel_flag','IsWeekend','IsBusinessHours','SeasonQuarter','IsHolidayMonth']
features_cols = ['historical_ontime_arrival_pct','historical_ontime_departure_pct','recent_ontime_arrival_pct','recent_ontime_departure_pct','short_term_ontime_arrival_pct','short_term_ontime_departure_pct','OntimeArrivalPct','OntimeDeparturePct']

numeric_cols = [c for c, t in df_otpw_5y.dtypes if t in ('int', 'bigint', 'double', 'float', 'decimal')]
numeric_cols = [c for c in numeric_cols if c not in datetime_cols+categorical_cols+features_cols]
if "DepDel15" not in numeric_cols:
    numeric_cols.append("DepDel15")

# Limit to a sample for plotting efficiency
pdf = df_otpw_5y.select(numeric_cols).dropna().sample(fraction=0.2, seed=42).toPandas()

# Melt dataframe for seaborn boxplot
melted = pd.melt(pdf, id_vars="DepDel15", value_vars=[c for c in numeric_cols if c != "DepDel15"])

plt.figure(figsize=(16, max(6, len(numeric_cols)*0.7)))
sns.boxplot(x="variable", y="value", hue="DepDel15", data=melted, showfliers=False)
plt.title("Boxplot of Numeric Features by DepDel15")
plt.xlabel("Feature")
plt.ylabel("Value")
plt.legend(title="DepDel15")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# COMMAND ----------

# Select numeric columns only for boxplot
numeric_cols = ['TAXI_OUT','Num_airport_wide_delays','Num_airport_wide_cancelations','ARR_DELAY','PREV_ArrDelay','DISTANCE', 'HourlyWindSpeed','HourlyWetBulbTemperature']
if "DepDel15" not in numeric_cols:
    numeric_cols.append("DepDel15")

# Limit to a sample for plotting efficiency
pdf = df_otpw_5y.select(numeric_cols).dropna().sample(fraction=0.05, seed=42).toPandas()

# Melt dataframe for seaborn boxplot
melted = pd.melt(pdf, id_vars="DepDel15", value_vars=[c for c in numeric_cols if c != "DepDel15"])

plt.figure(figsize=(16, max(6, len(numeric_cols)*0.7)))
sns.boxplot(x="variable", y="value", hue="DepDel15", data=melted, showfliers=False)
plt.title("Boxplot of Numeric Features by DepDel15")
plt.xlabel("Feature")
plt.ylabel("Value")
plt.legend(title="DepDel15")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Histograms

# COMMAND ----------

numeric_cols = [c for c, t in df_otpw_5y.dtypes if t in ('int', 'bigint', 'double', 'float', 'decimal')]
pdf = df_otpw_5y.select(numeric_cols).dropna().sample(fraction=0.05, seed=42).toPandas()

import matplotlib.pyplot as plt

n_cols = 3
n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
plt.figure(figsize=(6 * n_cols, 4 * n_rows))

for i, col in enumerate(numeric_cols, 1):
    plt.subplot(n_rows, n_cols, i)
    plt.hist(pdf[col], bins=30, color='skyblue', edgecolor='black')
    plt.title(col)
    plt.xlabel('Value')
    plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Shapiro Wilk

# COMMAND ----------

from scipy.stats import shapiro

shapiro_results = {}
for col in numeric_cols:
    # Shapiro-Wilk test requires at least 3 and at most 5000 samples
    data = pdf[col].dropna()
    sample = data.sample(n=min(len(data), 5000), random_state=42) if len(data) > 5000 else data
    if len(sample) >= 3:
        stat, p = shapiro(sample)
        shapiro_results[col] = {'column': col, 'statistic': stat, 'p_value': p}
    else:
        shapiro_results[col] = {'column': col, 'statistic': None, 'p_value': None}

import pandas as pd
shapiro_df = pd.DataFrame.from_dict(shapiro_results, orient='index')
shapiro_df = shapiro_df.sort_values(by='statistic', ascending=False).reset_index(drop=True)
display(shapiro_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Heatmap of null values

# COMMAND ----------

import matplotlib.pyplot as plt
import seaborn as sns

missing_df = null_counts_df[null_counts_df['null_counts'] > 0].copy()

plt.figure(figsize=(8, 6))
sns.heatmap(
    data=missing_df.set_index('index')[['null_counts']],
    cmap='Blues',
    annot=True,
    fmt='.0f',
    cbar=True
)
plt.title('Missing Values by Column')
plt.xlabel('Missing')
plt.ylabel('Column')
plt.tight_layout()
plt.show()

# COMMAND ----------

display(df_otpw_5y)

# COMMAND ----------

df_otpw_5y.groupBy("Origin").count().orderBy("count", ascending=False).show()
df_otpw_5y.groupBy("HourlyWetBulbTemperature").count().orderBy("count", ascending=False).show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Basics EDA
# MAGIC ## B) Data dictionary of the raw features 
# MAGIC |Column|Data Type|Variable Type|Comment|Action
# MAGIC |------|--------------------------|-----------|-------------------------|-----------|
# MAGIC |'DAY_OF_MONTH',|integer|Ordinal and Cyclic|Make cyclic: df['day_of_month_sin'] = np.sin(2 * np.pi * df['day_of_month'] / 31),df['day_of_month_cos'] = np.cos(2 * np.pi * df['day_of_month'] / 31)
# MAGIC |'DAY_OF_WEEK',|integer|Ordinal and Cyclic|1-Monday,2-Tuesday,3-Wednesday,4-Thursday,5-Friday,6-Saturday,7-Sunday <br/><br/> Make cyclic: df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7) df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)|
# MAGIC |'QUARTER',|integer|Ordinal and Cyclic|Make cyclic: df['quarter_sin'] = np.sin(2 * np.pi * df['quarter'] / 4) df['quarter_cos'] = np.cos(2 * np.pi * df['quarter'] / 4)
# MAGIC |'FL_DATE',|date|Date|Will be split to DAY_OF_MONTH, MONTH and YEAR 
# MAGIC |'OP_UNIQUE_CARRIER',|String|Nominal| Represented by OP_CARRIER_AIRLINE_ID
# MAGIC |'OP_CARRIER_AIRLINE_ID',|integer|Nominal| Shows the carrier and can be useful in airline behaviro <br/><br/>avg_delay_by_carrier = df.groupBy('OP_CARRIER').agg(mean('DEP_DELAY').alias('avg_dep_delay_carrier'))
# MAGIC |'OP_CARRIER',|string|Nominal| Represented by OP_CARRIER_AIRLINE_ID
# MAGIC |'TAIL_NUM',|string|Nominal|Uniquely identifies each airplane<br/><br/> Use embeddings and create dictionary
# MAGIC |'OP_CARRIER_FL_NUM',|integer|Identifier Text|Flight numbers are almost unique per day and do not generalize well.
# MAGIC |'ORIGIN_AIRPORT_ID',|integer|Nominal|"ORIGIN_AIRPORT_ID" will be used for modelling because it provides a cleaner, integer representation of the airport as per the alphabetic 3 letter code provided in ORIG."Use this field for airport analysis across a range of years because an airport can change its airport code and airport codes can be reused."
# MAGIC |'ORIGIN_AIRPORT_SEQ_ID',|integer|Nominal|"Origin Airport, Airport Sequence ID. An identification number assigned by US DOT to identify a unique airport at a given point of time. Airport attributes, such as airport name or coordinates, may change over time."
# MAGIC |'ORIGIN_CITY_MARKET_ID',|integer|Nominal|"ORIGIN_CITY_MARKET_ID": Consolidates multiple airports in the same city (e.g., LAX, BUR, and LGB are all part of the Los Angeles market).Allows analysis at the city level rather than the individual airport level.Reduces Noise From Multiple Airports and allows for consistent grouping.
# MAGIC |'ORIGIN',|string|Nominal|Important as it defines the labels of airports
# MAGIC |'ORIGIN_CITY_NAME',|string|Identifier text|Not important as it is represented by `ORIGIN`
# MAGIC |'ORIGIN_STATE_ABR',|string|Categorical|Will be represented by `ORIGIN_STATE_FIPS` 
# MAGIC |'ORIGIN_STATE_FIPS',|integer|Nominal|A numerical representation of states
# MAGIC |'ORIGIN_STATE_NM',|string|Text|The name of the state already given by ORIGIN_STATE_FIPS and ORIGIN_STATE_ABR 
# MAGIC |'ORIGIN_WAC',|integer|Nominal|Is not the same as the ORIGIN IDS
# MAGIC |'DEST_AIRPORT_ID',|integer|Nominal|"DEST_AIRPORT_ID" will be used for modelling because it provides a cleaner, integer representation of the airport as per the alphabetic 3 letter code provided in DEST.
# MAGIC |'DEST_AIRPORT_SEQ_ID',|integer|Nominal|"Airport Sequence ID. An identification number assigned by US DOT to identify a unique airport at a given point of time. Airport attributes, such as airport name or coordinates, may change over time."
# MAGIC |'DEST_CITY_MARKET_ID',|integer|Nominal|"DEST_CITY_MARKET_ID": Consolidates multiple airports in the same city (e.g., LAX, BUR, and LGB are all part of the Los Angeles market).Allows analysis at the city level rather than the individual airport level.Reduces Noise From Multiple Airports and allows for consistent grouping.
# MAGIC |'DEST',|string|Nominal|Important as it defines the discrete categories of airports.Will be represented by DEST_AIRPORT_ID
# MAGIC |'DEST_CITY_NAME',|string|Text|Not important as it is represented by `DEST`
# MAGIC |'DEST_STATE_ABR',|string|Categorical|Will be represented by `DEST_STATE_FIPS` 
# MAGIC |'DEST_STATE_FIPS',|integer|Nominal|A numerical representation of states
# MAGIC |'DEST_STATE_NM',|string|Text|The name of the state already given by DEST_STATE_FIPS and DEST_STATE_ABR 
# MAGIC |'DEST_WAC',|integer|Nominal|Is not the same as the ORIGIN IDS
# MAGIC |'CRS_DEP_TIME',|integer|Ordinal and Cyclic|CRS Departure Time (local time: hhmm)<br/><br/> Convert to cyclic as time after midnight
# MAGIC |'DEP_TIME',|integer|Ordinal and Cyclic|Actual Departure Time (local time: hhmm)<br/><br/> Convert to cyclic as time after midnight.However, this is unknown 2 hours before flight
# MAGIC |'DEP_DELAY',|double|Integer|Difference in minutes between scheduled and actual departure time. Early departures show negative numbers.However, this is unknown 2 hours before flight
# MAGIC |'DEP_DELAY_NEW',|double|Integer|Difference in minutes between scheduled and actual departure time. Early departures set to 0.
# MAGIC |'DEP_DEL15',|double|Categorical|Departure Delay Indicator, 15 Minutes or More (1=Yes)
# MAGIC |'DEP_DELAY_GROUP',|integer|Nominal|However, this is unknown 2 hours before flight.
# MAGIC |'DEP_TIME_BLK',|string|Nominal|CRS Departure Time Block, Hourly Intervals. Change to categories Morning, Afternoon and Night
# MAGIC |'TAXI_OUT',|double|Integer|Taxi Time, in Minutes.However, this is unknown 2 hours before flight hence  use rolling features for the airplane or airport 
# MAGIC |'WHEELS_OFF',|integer|Ordinal and Cyclic|Time (local time: hhmm)<br/><br/> Convert to cyclic as time after midnight.However, this is unknown 2 hours before flight  use rolling features for the airplane or airport 
# MAGIC |'WHEELS_ON',|integer|Ordinal and Cyclic|Time (local time: hhmm)<br/><br/> Convert to cyclic as time after midnight. However, this is unknown 2 hours before flight hence  use rolling features for the airplane or airport 
# MAGIC |'TAXI_IN',|double|Integer|Taxi Time, in Minutes.However, this is unknown 2 hours before flight hence use rolling features for the airplane or airport 
# MAGIC |'CRS_ARR_TIME',|integer|Ordinal and Cyclic|Scheduled Arrival Time. It represents the time that an airline expects a flight to arrive at its destination, according to the schedule. Time (local time: hhmm)<br/><br/> Convert to cyclic as time after midnight
# MAGIC |'ARR_TIME',|integer|Ordinal and Cyclic|Time (local time: hhmm)<br/><br/> Convert to cyclic as time after midnight. However, this is can be covered by ARR_DELAY 
# MAGIC |'ARR_DELAY',|double|Integer|Arrival delay unknown 2 hours before flight. However, can  use rolling features for the airplane or airport 
# MAGIC |'ARR_DELAY_NEW',|double|Integer|Arrival delay unknown 2 hours before flight. However, can  use rolling features for the airplane or airport 
# MAGIC |'ARR_DEL15',|double|Ordinal|Arrival delay unknown 2 hours before flight. However, can  use rolling features for the airplane or airport 
# MAGIC |'ARR_DELAY_GROUP',|integer|Nominal|Arrival delay unknown 2 hours before flight. However, can  use rolling features for the airplane or airport 
# MAGIC |'ARR_TIME_BLK',|string|Nominal|Arrival time unknown 2 hours before flight. However, can  use rolling features for the airplane or airport 
# MAGIC |'CANCELLED',|double|Nominal|Cancelled flights can be useful for checking on carrier or airplane behaviour
# MAGIC |'DIVERTED',|double|Nominal|Diverted flights can be used in rolling features for the airplane or airport 
# MAGIC |'CRS_ELAPSED_TIME',|double|Integer|Elapsed Time of Flight, in Minutes.However, However, this can be covered with ARR_DELAY.
# MAGIC |'ACTUAL_ELAPSED_TIME',|double|Integer|Elapsed Time of Flight, in Minutes.However, this can be covered with ARR_DELAY.
# MAGIC |'AIR_TIME',|double|Integer|Flight Time, in Minutes.However, this is unknown 2 hours before flight.
# MAGIC |'FLIGHTS',|double|Integer|Mostly a count of 1.Does not highlight anything meaningful
# MAGIC |'DISTANCE',|double|Integer|Distance between airports (miles)
# MAGIC |'DISTANCE_GROUP',|integer|Integer|Distance Intervals, every 250 Miles, for Flight Segment. Can be useful in depcting location behvaior
# MAGIC |'YEAR',|integer|Ordinal|Important for multiyear analysis, adds no value for 1 year
# MAGIC |'MONTH',|integer|Ordinal and cyclic|df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
# MAGIC |'origin_airport_name',|string|Text|Already catered for by ORIGIN_AIRPORT_ID
# MAGIC |'origin_station_name',|string|Text|Already catered for by STATION
# MAGIC |'origin_station_id',|long|Nominal|Identifier for origin_station_id.
# MAGIC |'origin_iata_code',|string|Text|Already catered for by AIRPORT_ID
# MAGIC |'origin_icao',|string|Text|Already catered for by AIRPORT_ID
# MAGIC |'origin_type',|string|Nominal|Provides additional information on airport parameters
# MAGIC |'origin_region',|string|Nominal|Already catered for by ORIGIN 
# MAGIC |'origin_station_lat',|double|double|Origin station latitude. Already catered for by STATION
# MAGIC |'origin_station_lon',|double|double|Origin station longitude. Already catered for by STATION
# MAGIC |'origin_airport_lat',|double|double|Origin airport latitude.Already catered for by AIRPORT_ID
# MAGIC |'origin_airport_lon',|double|double|Origin airport longitude.Already catered for by AIRPORT_ID
# MAGIC |'origin_station_dis',|double|double|Extremely imbalanced:~99.5% of records have the value 0.0
# MAGIC |'dest_airport_name',|string|Text|Catered for by dest_station_id
# MAGIC |'dest_station_name',|string|Text|Catered for by dest_station_id
# MAGIC |'dest_station_id',|long|Nominal|Represents the dest station identifier
# MAGIC |'dest_iata_code',|string|Text|Catered for by dest_station_id
# MAGIC |'dest_icao',|string|Text|Catered for by dest_station_id
# MAGIC |'dest_type',|string|Text|Shows the type of airport as small,medium,large. Possibly can be catered for by dest_station_id or used for group related feature engineering for airport attributes
# MAGIC |'dest_region',|string|Text|Catered for by dest_station_id
# MAGIC |'dest_station_lat',|double|double|Destination Station latitude. Already catered for by dest_station_id
# MAGIC |'dest_station_lon',|double|double|Destination Station longitude. Already catered for by dest_station_id
# MAGIC |'dest_airport_lat',|double|double|Destination Airport latitude. Already catered for by dest_station_id
# MAGIC |'dest_airport_lon',|double|double|Destination longitude longitude. Already catered for by dest_station_id
# MAGIC |'dest_station_dis',|double|double|Extremely imbalanced:~99.5% of records have the value 0.0
# MAGIC |'sched_depart_date_time',|timestamp|Ordinal and Cyclic|Catered for by CRS_DEP_TIME
# MAGIC |'sched_depart_date_time_UTC',|timestamp|Ordinal and Cyclic|Catered for by CRS_DEP_TIME
# MAGIC |'four_hours_prior_depart_UTC',|timestamp|Ordinal and Cyclic|Catered for by CRS_DEP_TIME. Assumed to be a derived feature for joining
# MAGIC |'two_hours_prior_depart_UTC',|timestamp|Ordinal and Cyclic|Catered for by CRS_DEP_TIME. Assumed to be a derived feature for joining
# MAGIC |'STATION',|long|Nominal|Represented by origin_station_id 
# MAGIC |'DATE',|timestamp|Ordinal and Cyclic|Assumed to be a derived feature for joining
# MAGIC |'LATITUDE',|double|Nominal|Represented by origin_station_id 
# MAGIC |'LONGITUDE',|double|Nominal|Represented by origin_station_id 
# MAGIC |'ELEVATION',|double|double|refers to the height of a location above sea level, and it significantly influences temperature and other atmospheric conditions.
# MAGIC |'NAME',|string|Text|Text identifier name
# MAGIC |'REPORT_TYPE',|string|Nominal|In weather reporting, FM-15 refers to the METAR code (Aerodrome Routine Meteorological Report), and FM-16 refers to the SPECI code (Aerodrome Selected Special Meteorological Report). Records with FM16 suggest special weather conditions|
# MAGIC |'SOURCE',|string|Nominal| likely relate to ice conditions and the presence of icebergs or growlers.Might be useful for sea conditions, maybe not flights 
# MAGIC |'HourlyAltimeterSetting',|string|Float| the barometric pressure used to calibrate an aircraft's altimeter so it indicates the altitude above sea level
# MAGIC |'HourlyDewPointTemperature',|string|Float| dew point is a crucial factor for pilots as it helps predict fog, clouds, and even icing conditions. 
# MAGIC |'HourlyDryBulbTemperature',|string|Float|the temperature that a standard thermometer would read when not affected by moisture. In weather, it's a fundamental measurement for understanding atmospheric conditions, and in aviation, it plays a role in aircraft performance and calculations. 
# MAGIC |'HourlyPrecipitation',|string|Float|Light rain may have minimal impact, but heavier precipitation can lead to slower taxi speeds, congestion, and potential delays due to reduced visibility and slippery runways.
# MAGIC |'HourlyRelativeHumidity',|string|Float|It influences precipitation, fog, and thunderstorm development in weather, while also impacting takeoff distances, climb performance, 
# MAGIC |'HourlySkyConditions',|string|Float|Pilots use sky condition information to determine safe altitudes and flight routes, avoiding areas with low ceilings or poor visibility. 
# MAGIC |'HourlySeaLevelPressure',|string|Float|Pilots use sea level pressure to calibrate their altimeters, which indicate altitude. 
# MAGIC |'HourlyStationPressure',|string|Float|Measured directly at a weather station's location, it's the true atmospheric pressure at that specific altitude. 
# MAGIC |'HourlyVisibility',|string|Float|It refers to the horizontal distance at which objects can be seen and recognized. Poor visibility, often caused by weather phenomena like fog, rain, or snow, can lead to delays, diversions, and even accidents. 
# MAGIC |'HourlyWetBulbTemperature',|string|Float|wet-bulb temperature can affect aircraft performance and safety, particularly at takeoff and landing. 
# MAGIC |'HourlyWindDirection',|string|Float|it dictates which runway a plane will use for takeoff and landing, affecting flight paths and potentially delays. 
# MAGIC |'HourlyWindSpeed',|string|Float| Strong crosswinds and tailwinds can affect takeoff and landing, potentially causing delays or even diversions.
# MAGIC  'REM',|string|Text|Remarks Section. Unavailable at point of departure
# MAGIC  '_row_desc'|integer|integer|Looks like a count of 1
# MAGIC
# MAGIC
# MAGIC ## D) Newly Joined Dataset Summary Statistcis
# MAGIC |index|null_counts|percentage|mean|stddev|min|max|median|
# MAGIC |:----|:----|:----|:----|:----|:----|:----|:----|
# MAGIC |CANCELLED|0|0|0.02|0.13|0.0|1.0|0|
# MAGIC |CRSDepTime|0|0|1330.26|492.99|1.0|2359.0|1320|
# MAGIC |CRSElapsedTime|135|0|141.9|72.35|1.0|948.0|125|
# MAGIC |CRS_ARR_TIME|0|0|1485.79|521.47|1.0|2400.0|1511|
# MAGIC |DayOfWeek|0|0|3.94|2|1.0|7.0|4|
# MAGIC |DayofMonth|0|0|15.73|8.76|1.0|31.0|16|
# MAGIC |DepTimeBlk|0|0|null|null|null|null|null|
# MAGIC |Dest|0|0|null|null|null|null|null|
# MAGIC |DestAirportID|0|0|12648.81|1523.82|10135.0|16869.0|12889|
# MAGIC |DestCityName|0|0|null|null|null|null|null|
# MAGIC |DestState|0|0|null|null|null|null|null|
# MAGIC |DestWac|0|0|54.03|26.16|1.0|93.0|44|
# MAGIC |Distance|0|0|800.54|592.51|31.0|5095.0|631|
# MAGIC |DistanceGroup|0|0|3.68|2.33|1.0|11.0|3|
# MAGIC |FlightDate|0|0|null|null|null|null|null|
# MAGIC |Flight_Number_Reporting_Airline|0|0|2557.2|1799.41|1.0|7933.0|2134|
# MAGIC |Hour|0|0|13.03|4.91|0.0|23.0|13|
# MAGIC |IsBusinessHours|0|0|0.97|0.18|0.0|1.0|1|
# MAGIC |IsHolidayMonth|0|0|0.24|0.43|0.0|1.0|0|
# MAGIC |IsWeekend|0|0|0.29|0.45|0.0|1.0|0|
# MAGIC |Month|0|0|6.58|3.4|1.0|12.0|7|
# MAGIC |Num_airport_wide_cancelations|0|0|0.74|3.35|0.0|118.0|0|
# MAGIC |Num_airport_wide_delays|0|0|8.04|11.17|0.0|161.0|4|
# MAGIC |Oncoming_flights|0|0|41.61|39.39|0.0|195.0|32|
# MAGIC |Origin|0|0|null|null|null|null|null|
# MAGIC |OriginAirportID|0|0|12648.88|1523.85|10135.0|16869.0|12889|
# MAGIC |OriginCityName|0|0|null|null|null|null|null|
# MAGIC |OriginState|0|0|null|null|null|null|null|
# MAGIC |OriginWac|0|0|54.03|26.16|1.0|93.0|44|
# MAGIC |Quarter|0|0|2.53|1.11|1.0|4.0|3|
# MAGIC |Reporting_Airline|0|0|null|null|null|null|null|
# MAGIC |SeasonQuarter|0|0|null|null|null|null|null|
# MAGIC |Year|0|0|2019|0|2019.0|2019.0|2019|
# MAGIC |arrival_datetime|98|0|1562217917.2|8980149.83|1.54630086E9|1.57783674E9|1562280900|
# MAGIC |cancel_flag|0|0|0.02|0.13|0.0|1.0|0|
# MAGIC |delay_flag|0|0|0.18|0.39|0.0|1.0|0|
# MAGIC |departure_datetime|0|0|1562212131.77|8980407.11|1.54630104E9|1.57783674E9|1561999620|
# MAGIC |Prev_FlightDate|5892|0.08|null|null|null|null|null|
# MAGIC |OntimeArrivalPct|18170|0.24|81.49|5.9|0.0|100.0|81.82|
# MAGIC |OntimeDeparturePct|18170|0.24|79.39|7.52|0.0|100.0|79.3|
# MAGIC |Tail_Number|17837|0.24|null|null|null|null|null|
# MAGIC |historical_ontime_arrival_pct|18170|0.24|81.49|5.9|0.0|100.0|81.82|
# MAGIC |historical_ontime_departure_pct|18170|0.24|79.39|7.52|0.0|100.0|79.3|
# MAGIC |short_term_ontime_arrival_pct|18569|0.25|81.84|6.57|0.0|100.0|82.21|
# MAGIC |short_term_ontime_departure_pct|18569|0.25|79.81|8.16|0.0|100.0|80.2|
# MAGIC |recent_ontime_arrival_pct|33628|0.45|81.89|9.95|0.0|100.0|82.95|
# MAGIC |recent_ontime_departure_pct|33628|0.45|79.9|11.38|0.0|100.0|81.16|
# MAGIC |DEP_TIME|130086|1.75|1334.61|507.2|1.0|2400.0|1322|
# MAGIC |DepDel15|130110|1.75|0.19|0.39|0.0|1.0|0|
# MAGIC |TAXI_OUT|133977|1.81|17.39|10|1.0|227.0|15|
# MAGIC |ARR_TIME|137646|1.85|1462.96|542.45|1.0|2400.0|1500|
# MAGIC |TAXI_IN|137647|1.85|7.74|6.19|1.0|316.0|6|
# MAGIC |Prev_TaxiOut|139791|1.88|17.39|10|1.0|227.0|15|
# MAGIC |Prev_ArrTime|143455|1.93|1462.75|542.34|1.0|2400.0|1459|
# MAGIC |Prev_TaxiIn|143456|1.93|7.74|6.19|1.0|316.0|6|
# MAGIC |ARR_DELAY|153805|2.07|5.41|51.07|-99.0|2695.0|-6|
# MAGIC |Prev_ArrDelay|159600|2.15|5.42|51.06|-99.0|2695.0|-6|
# MAGIC |Turnaround_Time|223079|3.01|401.35|692.75|-1741.0|4758.0|106|
# MAGIC |HourlyDryBulbTemperature|275091|3.71|56.11|22.16|-54.0|140.0|58.33|
# MAGIC |HourlyRelativeHumidity|291842|3.93|69.43|22.32|3.0|100.0|73|
# MAGIC |HourlyDewPointTemperature|293918|3.96|44.22|20.51|-44.0|83.0|46|
# MAGIC |HourlyWindSpeed|314091|4.23|7.72|5.83|0.0|84.0|7|
# MAGIC |HourlyVisibility|348351|4.69|8.9|2.79|0.0|90.0|10|
# MAGIC |HourlyWetBulbTemperature|591505|7.97|50.11|19.11|-36.0|84.5|52.5|
# MAGIC |HourlySkyConditions|879194|11.85|28.52|21.33|0.0|74.0|26|
# MAGIC |HourlySeaLevelPressure|4354714|58.67|29.99|0.24|28.35|31.29|29.99|
# MAGIC |HourlyPrecipitation|5422691|73.06|0.01|0.07|0.0|2.03|0|
# MAGIC |HourlyWindGustSpeed|5798893|78.13|23.08|6.37|11.0|82.0|22|
# MAGIC |HourlyPresentWeatherType|6147814|82.83|null|null|null|null|null|

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC Considerations for feature engineering
# MAGIC
# MAGIC |Category,               |Feature,            |Use           |
# MAGIC |------------------------|--------------------|--------------|
# MAGIC |Time-based,             |CRS_DEP_TIME DAY_OF_WEEK MONTH QUARTER,|Encode cyclical patterns|
# MAGIC |Location,               |ORIGIN, DEST, ORIGIN_REGION, ORIGIN_TYPE,|Capture congestion/weather risks|
# MAGIC |Carrier behavior,       |OP_CARRIER, historical delay rates by carrier,|Carriers vary in performance|
# MAGIC |Historical congestion,  |Mean # flights scheduled in past 2 hours at ORIGIN,|Proxy for traffic at airport|
# MAGIC |Historical delay context,|% delayed flights in past 2h at ORIGIN,|Captures real-time risk signal|
# MAGIC |Weather,                |Weather features 2h prior to CRS_DEP_TIME,|Strong predictor for delay|
# MAGIC |Rolling features,       |Mean TAXI_OUT, WHEELS_OFF delay in past 2h|Can simulate expected congestion|
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC #Bivariate Analysis

# COMMAND ----------

# MAGIC %md
# MAGIC ## Correlation Heatmap

# COMMAND ----------

import matplotlib.pyplot as plt
import seaborn as sns

numeric_cols = [field for (field, dtype) in df_otpw_5y.dtypes if dtype in ('int', 'double', 'float', 'bigint')]
pandas_df = df_otpw_5y.select(numeric_cols).sample(fraction=0.1).toPandas()
corr = pandas_df.corr(method='pearson')

mask = np.triu(np.ones_like(corr, dtype=bool))
plt.figure(figsize=(10, 8))
sns.heatmap(corr, mask=mask, annot=False, cmap='coolwarm', square=True, linewidths=.5)
plt.title('Correlation Heatmap')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pairplot

# COMMAND ----------

plt.figure(figsize=(10, 8))
df=df_otpw_5y[['CRS_ARR_TIME', 'DepDel15', 'CRSDepTime']].sample(fraction=0.1).toPandas()
sns.pairplot(df)
plt.title('Pairplot for some fields')
plt.show()

# COMMAND ----------

import matplotlib.pyplot as plt

df_pd = df_otpw_5y.select("CRS_ARR_TIME", "CRSDepTime").sample(fraction=0.1).dropna().toPandas()

plt.figure(figsize=(8,6))
plt.scatter(df_pd["CRSDepTime"], df_pd["CRS_ARR_TIME"], alpha=0.5)
plt.xlabel("CRSDepTime")
plt.ylabel("CRS_ARR_TIME")
plt.title("Relationship between CRS_ARR_TIME and CRSDepTime")
plt.grid(True)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC # Trends Analysis

# COMMAND ----------

# MAGIC %md
# MAGIC ## Target Variable by monrning, evening,night
# MAGIC  

# COMMAND ----------

df_otpw_5y.groupBy("DepDel15").count().orderBy("count", ascending=False).display()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Trends by Month/Week/Hour
# MAGIC

# COMMAND ----------

from pyspark.sql.functions import concat_ws

df_count_by_month = df_otpw_5y.groupBy("Month", "Year", "DayofMonth", "DayOfWeek", "Hour", "DepDel15").count() \
    .withColumn("Month_Year", concat_ws("-", "Month", "Year")) \
    .orderBy("Year", "Month", "DayofMonth", "DayOfWeek", "Hour", ascending=True)

display(df_count_by_month)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Early Morning, Morning, Afternoon, Evening

# COMMAND ----------

from pyspark.sql.functions import when

df_otpw_5y_timeblock = df_otpw_5y.withColumn(
    "HourBlock",
    when((df_otpw_5y["Hour"] >= 0) & (df_otpw_5y["Hour"] < 6), "1.Early Morning")
    .when((df_otpw_5y["Hour"] >= 6) & (df_otpw_5y["Hour"] < 12), "2.Morning")
    .when((df_otpw_5y["Hour"] >= 12) & (df_otpw_5y["Hour"] < 18), "3.Afternoon")
    .otherwise("4. Night")
)

display(
    df_otpw_5y_timeblock.groupBy("Year", "HourBlock","DepDel15").count().orderBy("Year", "HourBlock")
)

# COMMAND ----------

# df_otpw_1y.groupBy("ORIGIN").count().orderBy("count", ascending=False)
display(df_otpw_5y.groupBy("ORIGIN").count().orderBy("count", ascending=False))

# COMMAND ----------

# df_otpw_1y.groupBy("DEST").count().orderBy("count", ascending=False).display()
display(df_otpw_5y.groupBy("Dest").count().orderBy("count", ascending=False))

# COMMAND ----------

df_orig_dest = df_otpw_1y.groupBy("OriginState","Origin", "OriginAirportID","DestState","Dest","DestAirportID","DepDel15").count()

display(df_orig_dest)

# COMMAND ----------

display(df_orig_dest.groupBy("OriginState").sum('count'))

# COMMAND ----------

import plotly.express as px

df_state_counts = df_orig_dest.groupBy("OriginState").sum('count').toPandas()

fig = px.choropleth(
    df_state_counts,
    locations="OriginState",
    locationmode="USA-states",
    color="sum(count)",
    scope="usa",
    color_continuous_scale="Blues",
    labels={"sum(count)": "Flight Count"},
    width=1200,
    height=800
)

for i, row in df_state_counts.iterrows():
    fig.add_scattergeo(
        locations=[row["OriginState"]],
        locationmode="USA-states",
        text=[f"{row['OriginState']}({row['sum(count)']})"],
        mode="text",
        textfont=dict(size=8, color="black"),
        showlegend=False
    )

fig.update_layout(
    margin={"r":0,"t":0,"l":0,"b":0},
    geo=dict(
        scope="usa",
        projection= dict(type='albers usa'),
        showlakes=True,
        lakecolor='rgb(255, 255, 255)'
    )
)

display(fig)

# COMMAND ----------

display(df_otpw_5y.groupBy('Dest').sum('count'))

# COMMAND ----------

# MAGIC %md
# MAGIC The higher the number of airports the higher the total flights.

# COMMAND ----------

# MAGIC %md
# MAGIC # Graph Trends

# COMMAND ----------

# MAGIC %md
# MAGIC ## Betweenness

# COMMAND ----------

import networkx as nx
import matplotlib.pyplot as plt

# Sample Spark DataFrame
flights_df = df_otpw_5y.select("Origin", "Dest").dropna().distinct()

# Convert to Pandas for NetworkX
flights_pd = flights_df.sample(fraction=0.7).toPandas()

# Create directed graph
G = nx.DiGraph()

# Add edges from flight data
G.add_edges_from(flights_pd.values)

# Compute betweenness centrality
bc = nx.betweenness_centrality(G)

# Add centrality as node attribute
nx.set_node_attributes(G, bc, "betweenness")

# Get top 10 airports by centrality
top_bc = sorted(bc.items(), key=lambda x: -x[1])[:10]
print("Top 10 Airports by Betweenness Centrality:")
for airport, score in top_bc:
    print(f"{airport}: {score:.4f}")

# Optional: Draw the graph (small subset recommended)
subgraph_nodes = [x[0] for x in top_bc]
subG = G.subgraph(subgraph_nodes)

plt.figure(figsize=(20, 6))
pos = nx.spring_layout(subG)
node_sizes = [5000 * bc[n] for n in subG.nodes]
nx.draw(subG, pos, with_labels=True, node_size=node_sizes, node_color='lightblue', edge_color='gray')
plt.title("Top Airports by Betweenness Centrality")
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC # EDA Phase 1B
# MAGIC
# MAGIC 1. Drop null values in Target Variable `DepDel15`

# COMMAND ----------

df_otpw_1y = df_otpw_1y.na.drop(subset=["DepDel15"])

# COMMAND ----------

df_otpw_1y.groupBy("DepDel15").count().orderBy("count", ascending=False).display()

# COMMAND ----------

df_orig_dest_nadrop = df_otpw_1y.groupBy("OriginState","OriginAirportID","DestState","DestAirportID","Origin","Dest","Origin").count()
display(df_orig_dest_nadrop)

# COMMAND ----------

display(df_otpw_1y.groupBy("Dest").count().orderBy("count", ascending=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Check the nulls remaining in dataset

# COMMAND ----------

# Summary Statistics

import pandas as pd
import numpy as np
from pyspark.sql.functions import col, sum as spark_sum

# Check for nulls
df_otpw_1y_count = df_otpw_1y.count()
null_counts = df_otpw_1y.select([spark_sum(col(c).isNull().cast("float")).alias(c) for c in df_otpw_1y.columns])

null_counts_df = null_counts.toPandas()
null_counts_df = null_counts_df.transpose()
null_counts_df.columns = ['null_counts']
null_counts_df = null_counts_df.reset_index()
null_counts_df['null_counts'] = pd.to_numeric(null_counts_df['null_counts'], errors='coerce')
null_counts_df['percentage'] = round((null_counts_df['null_counts'] / df_otpw_1y_count) * 100, 2)

# Calculate stats
stats = df_otpw_1y.select([col(c).cast("double").alias(c) for c in df_otpw_1y.columns])
summary = stats.summary("mean", "stddev", "min", "max").toPandas().set_index('summary').transpose()
summary.columns = ['mean', 'stddev', 'min', 'max']
medians = stats.approxQuantile(stats.columns, [0.5], 0.01)
median_series = pd.Series([m[0] if m else np.nan for m in medians], index=stats.columns, name='median')

summary['median'] = median_series
summary = summary.reset_index().rename(columns={'index': 'index'})

# Merge stats
null_counts_df = null_counts_df.merge(summary, left_on='index', right_on='index', how='left')
null_counts_df['mean'] = pd.to_numeric(null_counts_df['mean'], errors='coerce').round(2)
null_counts_df['stddev'] = pd.to_numeric(null_counts_df['stddev'], errors='coerce').round(2)
null_counts_df = null_counts_df.round(2)
null_counts_df = null_counts_df.sort_values(by=['percentage', 'index'], ascending=[True, True])
display(null_counts_df)

# COMMAND ----------

df_delay=df_otpw_1y.filter(col("DepDel15") == 1).groupBy("Month").count().orderBy("Month")
display(df_delay)

import matplotlib.pyplot as plt

# Convert to pandas for plotting
df_delay_pd = df_delay.toPandas()



# COMMAND ----------

df_delay_pd['count'].mean()

# COMMAND ----------

# Plot
plt.figure(figsize=(8, 5))
plt.bar(df_delay_pd['Month'], df_delay_pd['count'], color='blue')
plt.xlabel('Month')
plt.ylabel('Delayed Flights Count')
plt.title('Monthly Delayed Flights Count')
plt.axhline(y=87978, color='orange', linestyle='--', label='The mean value (113,513)') 
# plt.annotate('Highest Delayed Flights', 
#              (df_delay_pd['MONTH'].iloc[-1], df_delay_pd['count'].iloc[-1])

# # Annotate points where there is growth
# for i in range(1, len(df_delay_pd)):
#     if df_delay_pd['count'][i] > df_delay_pd['count'][i-1]:
#         plt.annotate('Growth', 
#                      (df_delay_pd['MONTH'][i], df_delay_pd['count'][i]),
#                      textcoords="offset points", xytext=(0,10), ha='center', color='red')
plt.legend()
plt.grid(True)
plt.show()

# COMMAND ----------

cols_with_nulls = null_counts_df[null_counts_df['percentage'] > 0]['index'].tolist()
numeric_cols_with_nulls = null_counts_df[(null_counts_df['percentage'] > 0) & (null_counts_df['median'].notnull())]['index'].tolist()
medians_dict = null_counts_df.set_index('index')['median'].to_dict()

from pyspark.sql.functions import when, lit

for col_name in numeric_cols_with_nulls:
    median_value = medians_dict[col_name]
    df_otpw_1y = df_otpw_1y.withColumn(
        col_name,
        when(df_otpw_1y[col_name].isNull(), lit(median_value)).otherwise(df_otpw_1y[col_name])
    )


df_otpw_1y.write.mode('overwrite').parquet(f"{folder_path}/df_otpw_1y_fillnull_columns.parquet")

display(df_otpw_1y.select(*cols_with_nulls))


# COMMAND ----------

display(df_otpw_1y)

# COMMAND ----------

# Summary Statistics

import pandas as pd
import numpy as np
from pyspark.sql.functions import col, sum as spark_sum

# Check for nulls
df_otpw_1y_count = df_otpw_1y.count()
null_counts = df_otpw_1y.select([spark_sum(col(c).isNull().cast("float")).alias(c) for c in df_otpw_1y.columns])

null_counts_df = null_counts.toPandas()
null_counts_df = null_counts_df.transpose()
null_counts_df.columns = ['null_counts']
null_counts_df = null_counts_df.reset_index()
null_counts_df['null_counts'] = pd.to_numeric(null_counts_df['null_counts'], errors='coerce')
null_counts_df['percentage'] = round((null_counts_df['null_counts'] / df_otpw_1y_count) * 100, 2)

# Calculate stats
stats = df_otpw_1y.select([col(c).cast("double").alias(c) for c in df_otpw_1y.columns])
summary = stats.summary("mean", "stddev", "min", "max").toPandas().set_index('summary').transpose()
summary.columns = ['mean', 'stddev', 'min', 'max']
medians = stats.approxQuantile(stats.columns, [0.5], 0.01)
median_series = pd.Series([m[0] if m else np.nan for m in medians], index=stats.columns, name='median')

summary['median'] = median_series
summary = summary.reset_index().rename(columns={'index': 'index'})

# Merge stats
null_counts_df = null_counts_df.merge(summary, left_on='index', right_on='index', how='left')
null_counts_df['mean'] = pd.to_numeric(null_counts_df['mean'], errors='coerce').round(2)
null_counts_df['stddev'] = pd.to_numeric(null_counts_df['stddev'], errors='coerce').round(2)
null_counts_df = null_counts_df.round(2)
null_counts_df = null_counts_df.sort_values(by=['percentage', 'index'], ascending=[True, True])
display(null_counts_df)


# COMMAND ----------

# MAGIC %md
# MAGIC # EDA Phase 1C
# MAGIC
# MAGIC 1. Drop additional columns that are repetitive and can cause data leakage
# MAGIC 2. Classify the remaining ones into continuous, ordinal and nominal values
# MAGIC 3. Create a corr matrix to view relationships

# COMMAND ----------

# phase_1_c_drop=[
# 'FL_DATE',
# 'OP_UNIQUE_CARRIER',
# 'OP_CARRIER',
# 'OP_CARRIER_FL_NUM',
# 'ORIGIN_CITY_NAME',
# 'ORIGIN_STATE_ABR',
# 'ORIGIN_STATE_NM',
# 'DEST',
# 'DEST_CITY_NAME',
# 'DEST_STATE_ABR',
# 'DEST_STATE_NM',
# 'DEP_TIME',
# 'DEP_DELAY',
# 'DEP_DELAY_NEW',
# 'DEP_DELAY_GROUP',
# 'ARR_DELAY_NEW',
# 'ARR_DEL15',
# 'ARR_DELAY_GROUP',
# 'ARR_TIME_BLK',
# 'CRS_ELAPSED_TIME',
# 'ACTUAL_ELAPSED_TIME',
# 'AIR_TIME',
# 'FLIGHTS',
# 'origin_airport_name',
# 'origin_station_name',
# 'origin_iata_code',
# 'origin_icao',
# 'origin_region',
# 'origin_station_lat',
# 'origin_station_lon',
# 'origin_airport_lat',
# 'origin_airport_lon',
# 'origin_station_dis',
# 'dest_airport_name',
# 'dest_station_name',
# 'dest_iata_code',
# 'dest_icao',
# 'dest_type',
# 'dest_region',
# 'dest_station_lat',
# 'dest_station_lon',
# 'dest_airport_lat',
# 'dest_airport_lon',
# 'dest_station_dis',
# 'sched_depart_date_time',
# 'sched_depart_date_time_UTC',
# 'four_hours_prior_depart_UTC',
# 'two_hours_prior_depart_UTC',
# 'STATION',
# 'DATE',
# 'LATITUDE',
# 'LONGITUDE',
# 'NAME',
# 'SOURCE',
# 'REM',
# '_row_desc',
# 'DISTANCE_GROUP',


# ]

# df_otpw_1y = df_otpw_1y.drop(*phase_1_c_drop)


# COMMAND ----------

# Summary Statistics

import pandas as pd
import numpy as np
from pyspark.sql.functions import col, sum as spark_sum

# Check for nulls
df_otpw_1y_count = df_otpw_1y.count()
null_counts = df_otpw_1y.select([spark_sum(col(c).isNull().cast("float")).alias(c) for c in df_otpw_1y.columns])

null_counts_df = null_counts.toPandas()
null_counts_df = null_counts_df.transpose()
null_counts_df.columns = ['null_counts']
null_counts_df = null_counts_df.reset_index()
null_counts_df['null_counts'] = pd.to_numeric(null_counts_df['null_counts'], errors='coerce')
null_counts_df['percentage'] = round((null_counts_df['null_counts'] / df_otpw_1y_count) * 100, 2)

# Calculate stats
stats = df_otpw_1y.select([col(c).cast("double").alias(c) for c in df_otpw_1y.columns])
summary = stats.summary("mean", "stddev", "min", "max").toPandas().set_index('summary').transpose()
summary.columns = ['mean', 'stddev', 'min', 'max']
medians = stats.approxQuantile(stats.columns, [0.5], 0.01)
median_series = pd.Series([m[0] if m else np.nan for m in medians], index=stats.columns, name='median')

summary['median'] = median_series
summary = summary.reset_index().rename(columns={'index': 'index'})

# Merge stats
null_counts_df = null_counts_df.merge(summary, left_on='index', right_on='index', how='left')
null_counts_df['mean'] = pd.to_numeric(null_counts_df['mean'], errors='coerce').round(2)
null_counts_df['stddev'] = pd.to_numeric(null_counts_df['stddev'], errors='coerce').round(2)
null_counts_df = null_counts_df.round(2)
null_counts_df = null_counts_df.sort_values(by=['percentage', 'index'], ascending=[True, True])
display(null_counts_df)

# COMMAND ----------

continuous_cols =[ 
'CRSDepTime',
'CRSElapsedTime',
'CRS_ARR_TIME',
'DestAirportID',
'DestWac',
'Distance',
'DistanceGroup',
'Flight_Number_Reporting_Airline',
'Num_airport_wide_cancelations',
'Num_airport_wide_delays',
'Oncoming_flights',
'OriginAirportID',
'OriginWac',
'arrival_datetime',
'departure_datetime',
'OntimeArrivalPct',
'OntimeDeparturePct',
'historical_ontime_arrival_pct',
'historical_ontime_departure_pct',
'short_term_ontime_arrival_pct',
'short_term_ontime_departure_pct',
'recent_ontime_arrival_pct',
'recent_ontime_departure_pct',
'DEP_TIME',
'TAXI_OUT',
'ARR_TIME',
'TAXI_IN',
'Prev_TaxiOut',
'Prev_ArrTime',
'Prev_TaxiIn',
'ARR_DELAY',
'Prev_ArrDelay',
'Turnaround_Time',
'HourlyDryBulbTemperature',
'HourlyRelativeHumidity',
'HourlyDewPointTemperature',
'HourlyWindSpeed',
'HourlyVisibility',
'HourlyWetBulbTemperature',
'HourlySkyConditions',
'HourlySeaLevelPressure',
'HourlyPrecipitation',
'HourlyWindGustSpeed'
]

ordinal_cols=[
'Hour',
'DayOfWeek',
'DayofMonth',
'Month',
'Quarter',
'Year',

]

nominal_cols =[
'DepDel15',
'cancel_flag',
'delay_flag',
'IsBusinessHours',
'IsHolidayMonth',
'IsWeekend',
'CANCELLED',
]

# COMMAND ----------

#Fill in with median values
def fill_nulls(df, cols):
    for c in cols:
        df = df.withColumn(c, col(c).cast("float"))
        median_value = df.approxQuantile(c, [0.5], 0.01)[0]
        df = df.fillna({c: median_value})


    return df
df_flights_Weather = fill_nulls(df_otpw_1y, continuous_cols)
# df_flights_Weather = df_otpw_1y.select(continuous_cols).fillna(df_otpw_1y.continuous_cols.median())

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.stat import Correlation
from pyspark.sql import functions as F

# Combine all columns for correlation
corr_cols = continuous_cols #+ ordinal_cols + nominal_cols

# Select only numeric columns (correlation only works for numeric types)
numeric_cols = [c for c in corr_cols if dict(df_flights_Weather.dtypes)[c] in ['int', 'bigint', 'double', 'float', 'long', 'short']]

# Randomly sample 50000 rows
df_sampled = (
    df_flights_Weather
    .select(numeric_cols)
    .dropna()
    .orderBy(F.rand())
    .limit(50000)
)

# Assemble features into a vector
assembler = VectorAssembler(inputCols=numeric_cols, outputCol="features", handleInvalid='keep')
df_vector = assembler.transform(df_sampled).select("features")

# Compute correlation matrix
corr_matrix = Correlation.corr(df_vector, "features", "pearson").head()[0].toArray()

# Convert to pandas DataFrame for display
import pandas as pd
corr_df = pd.DataFrame(corr_matrix, columns=numeric_cols, index=numeric_cols)
display(corr_df)

# COMMAND ----------

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(20, 16))
sns.heatmap(corr_df, annot=True, fmt=".2f", cmap="coolwarm", square=True, cbar=True, annot_kws={"size": 6})
plt.title("Pearson Correlation Matrix Heatmap")
plt.tight_layout()
plt.savefig("pearson_correlation_matrix.jpg", format="jpg", dpi=300, bbox_inches="tight")
display(plt.gcf())
plt.close()

# COMMAND ----------

# Correlation For Ordinal variables
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.stat import Correlation
from pyspark.sql import functions as F

# Combine all columns for correlation
corr_cols = ordinal_cols + continuous_cols + nominal_cols

# Select only numeric columns (correlation only works for numeric types)
numeric_cols = [c for c in corr_cols if dict(df_flights_Weather.dtypes)[c] in ['int', 'bigint', 'double', 'float', 'long', 'short']]

# Randomly sample 50000 rows
df_sampled = (
    df_flights_Weather
    .select(numeric_cols)
    .dropna()
    .orderBy(F.rand())
    .limit(50000)
)

# Assemble features into a vector
assembler = VectorAssembler(inputCols=numeric_cols, outputCol="features", handleInvalid='keep')
df_vector = assembler.transform(df_sampled).select("features")

# Compute correlation matrix
corr_matrix = Correlation.corr(df_vector, "features", "spearman").head()[0].toArray()

# Convert to pandas DataFrame for display
import pandas as pd
corr_df = pd.DataFrame(corr_matrix, columns=numeric_cols, index=numeric_cols)
display(corr_df)


# COMMAND ----------

import matplotlib.pyplot as plt
import seaborn as sns

# Move 'DEP_DEL15' to the start of the columns and rows for distinct display
cols = ['DepDel15'] + [col for col in corr_df.columns if col != 'DepDel15']
corr_df_reordered = corr_df.loc[cols, cols]

plt.figure(figsize=(20, 16))
sns.heatmap(corr_df_reordered, annot=True, fmt=".2f", cmap="coolwarm", square=True, cbar=True, annot_kws={"size": 6})
plt.title("Spearman Correlation Matrix Heatmap")
plt.tight_layout()
plt.savefig("spearman_correlation_matrix.jpg", format="jpg", dpi=300, bbox_inches="tight")
display(plt.gcf())
plt.close()

# COMMAND ----------

import matplotlib.pyplot as plt

# Assume df_flights_Weather is a Spark DataFrame with the relevant columns
# Convert to Pandas for plotting
cols_to_plot = continuous_cols + ordinal_cols
cols_to_plot = [c for c in cols_to_plot if c in df_flights_Weather.columns]
cols_to_plot = [c for c in cols_to_plot if c != 'DepDel15']

df_pd = df_flights_Weather.select(cols_to_plot + ['DepDel15']).toPandas()

num_cols = len(cols_to_plot)
ncols = 3
nrows = (num_cols + ncols - 1) // ncols

fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5*ncols, 4*nrows))
axes = axes.flatten()

for idx, col in enumerate(cols_to_plot):
    data = [df_pd[df_pd['DepDel15'] == val][col].dropna() for val in sorted(df_pd['DepDel15'].dropna().unique())]
    axes[idx].boxplot(data, labels=sorted(df_pd['DepDel15'].dropna().unique()))
    axes[idx].set_title(f"{col} by DepDel15")
    axes[idx].set_xlabel('DepDel15')
    axes[idx].set_ylabel(col)

for idx in range(num_cols, len(axes)):
    fig.delaxes(axes[idx])

plt.tight_layout()
display(plt.gcf())
plt.close()

# COMMAND ----------

display(df_otpw_1y.printSchema())

# COMMAND ----------

# %run ./p2_ML_Pipeline_pt2

# COMMAND ----------

# MAGIC %md
# MAGIC #IGNORE FROM THIS POINT, SOME VARS ARE NOT IN THE NEW DATASET
# MAGIC
# MAGIC ## DEP_TIME_BLK
# MAGIC Add morning, afternoon, night in DEP_TIME_BLK

# COMMAND ----------

from pyspark.sql.functions import when, col

df_flights = df_otpw_1y.withColumn(
    "X_DEP_TIME_BLK",
    when((col("DEP_TIME_BLK").substr(1, 2).cast("int") >=18) & 
         (col("DEP_TIME_BLK").substr(1, 2).cast("int") < 23), "4_night")
    .when((col("DEP_TIME_BLK").substr(1, 2).cast("int") >= 6) & 
          (col("DEP_TIME_BLK").substr(1, 2).cast("int") < 12), "2_morning")
    .when((col("DEP_TIME_BLK").substr(1, 2).cast("int") >= 12) & 
          (col("DEP_TIME_BLK").substr(1, 2).cast("int") < 18), "3_afternoon")
    .otherwise("1_early_morning")
)

# Add hour column for hourly analysis
df_flights = df_flights.withColumn(
    "X_DEP_TIME_BLK_HOUR",
    col("DEP_TIME_BLK").substr(1, 4).cast("int")
)
display(df_flights)

# COMMAND ----------

df_NEW_DEP_TIME_BLK = df_flights.groupBy("MONTH","X_DEP_TIME_BLK","DEP_DEL15","X_DEP_TIME_BLK_HOUR").count().orderBy("count", ascending=False)
display(df_NEW_DEP_TIME_BLK)

# COMMAND ----------

# MAGIC %md
# MAGIC #Phase 1D
# MAGIC
# MAGIC ## Feature importance checks
# MAGIC 1. XGBoost
# MAGIC 2. Random Forest 
# MAGIC 2. Permutation Importance

# COMMAND ----------

#Check point
# df_flights_Weather.write.mode('overwrite').parquet(f"{folder_path}/df_flights_Weather_c.parquet")
df_flights=spark.read.parquet(f"{folder_path}/df_flights_Weather_c.parquet")

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
import xgboost as xgb

# Select numeric columns and label
label_col = "DEP_DEL15"
numeric_cols = [c for c, t in df_flights.dtypes if t in ('int', 'double', 'float') and c != label_col]

# Assemble features
assembler = VectorAssembler(inputCols=numeric_cols, outputCol="features", handleInvalid="skip")
df_vec = assembler.transform(df_flights).select("features", label_col)

# Convert to Pandas for XGBoost
pandas_df = df_vec.toPandas()
X = pandas_df["features"].apply(lambda v: v.toArray()).tolist()
y = pandas_df[label_col].values

# Train XGBoost
dtrain = xgb.DMatrix(X, label=y, feature_names=numeric_cols)
params = {"objective": "binary:logistic", "eval_metric": "logloss"}
model = xgb.train(params, dtrain, num_boost_round=50)

# Get feature importances
importances = model.get_score(importance_type='gain')
importances_sorted = sorted(importances.items(), key=lambda x: x[1], reverse=True)

# Display as DataFrame
import pandas as pd
importance_df = pd.DataFrame(importances_sorted, columns=["feature", "importance"])
display(importance_df)

# COMMAND ----------

from pyspark.sql.functions import col, sum as spark_sum

null_counts = df_flights.select([spark_sum(col(c).isNull().cast("int")).alias(c) for c in df_flights.columns])
null_counts_long = null_counts.selectExpr("stack({0}, {1}) as (column, null_count)".format(
    len(df_flights.columns),
    ', '.join(["'{0}', `{0}`".format(c) for c in df_flights.columns])
)).filter("null_count > 0")

display(null_counts_long)

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.linalg import DenseVector
import numpy as np

from pyspark.sql import DataFrame
from pyspark.sql.functions import countDistinct, col
from pyspark.ml.feature import StringIndexer, OneHotEncoder, Tokenizer, HashingTF, IDF
from pyspark.ml import Pipeline
import re

def process_text_column_spark(df: DataFrame, column: str, cardinality_threshold=20, embedding=False):
    unique_count = df.select(countDistinct(col(column))).collect()[0][0]
    total_rows = df.count()

    if unique_count / total_rows > 0.9:
        print(f"Dropping {column} (likely an ID, {unique_count} unique values)")
        return df.drop(column)

    if unique_count <= cardinality_threshold:
        print(f"Applying One-Hot Encoding to {column} ({unique_count} unique values)")
        indexer = StringIndexer(inputCol=column, outputCol=f"{column}_index")
        encoder = OneHotEncoder(inputCols=[f"{column}_index"], outputCols=[f"{column}_ohe"])
        pipeline = Pipeline(stages=[indexer, encoder])
        model = pipeline.fit(df)
        df = model.transform(df).drop(column, f"{column}_index")
        return df

    if not embedding:
        print(f"Applying Frequency Encoding to {column} ({unique_count} unique values)")
        freq_df = (df.groupBy(column)
                     .count()
                     .withColumnRenamed("count", f"{column}_freq"))
        df = df.join(freq_df, on=column, how="left").drop(column)
        return df

    print(f"Applying TF-IDF Embedding to {column}")
    tokenizer = Tokenizer(inputCol=column, outputCol=f"{column}_words")
    hashingTF = HashingTF(inputCol=f"{column}_words", outputCol=f"{column}_tf", numFeatures=50)
    idf = IDF(inputCol=f"{column}_tf", outputCol=f"{column}_tfidf")
    pipeline = Pipeline(stages=[tokenizer, hashingTF, idf])
    model = pipeline.fit(df)
    df = model.transform(df).drop(column, f"{column}_words", f"{column}_tf")
    return df

def process_all_text_columns_spark(df: DataFrame, cardinality_threshold=20, embedding_columns=None):   
    # display(df)
    if embedding_columns is None:
        embedding_columns = []

    string_columns = [f.name for f in df.schema.fields if str(f.dataType) == "StringType"]

    for col_name in string_columns:
        use_embedding = col_name in embedding_columns
        df = process_text_column_spark(df, col_name, cardinality_threshold, embedding=use_embedding)

    return df

def feature_selection_with_rf(df, label_col, importance_threshold=0.001):
    numeric_cols = [c for c, t in df.dtypes if t in ('int', 'double', 'float') and c != label_col]
    assembler = VectorAssembler(inputCols=numeric_cols, outputCol="features", handleInvalid="skip")
    df_vec = assembler.transform(df)
    if df_vec.count() == 0:
        raise ValueError("The DataFrame is empty after processing. Please check your data and processing steps.")

    rf = RandomForestClassifier(labelCol=label_col, featuresCol="features", numTrees=50)
    rf_model = rf.fit(df_vec)

    importances = rf_model.featureImportances
    selected_features = [numeric_cols[i] for i, imp in enumerate(importances) if imp >= importance_threshold]

    print(f"Selected {len(selected_features)}/{len(numeric_cols)} features based on importance threshold {importance_threshold}")

    df_selected = df.select([label_col] + selected_features)
    return df_selected, selected_features

def process_and_select_features_spark(df, label_col, cardinality_threshold=20, embedding_columns=None, importance_threshold=0.01):
    print("=== Processing text columns ===")
    df_processed = process_all_text_columns_spark(df, cardinality_threshold, embedding_columns)

    print("=== Running feature selection with RandomForest ===")
    df_selected, selected_features = feature_selection_with_rf(df_processed, label_col, importance_threshold)

    return df_selected, selected_features

df = df_flights

final_df, selected_features = process_and_select_features_spark(
    df,
    label_col="DEP_DEL15",
    cardinality_threshold=10,
    embedding_columns=["X_DEP_TIME_BLK","HourlySkyConditions","ORIGIN","TAIL_NUM"],
    importance_threshold=0.01
)

final_df.show(truncate=False)
print("Selected features:", selected_features)

# COMMAND ----------

# MAGIC %md
# MAGIC