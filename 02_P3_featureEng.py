# Databricks notebook source
from datetime import datetime

# Configuration for checkpointing
section = "02"
number = "01"
folder_path = f"dbfs:/student-groups/Group_{section}_{number}"
base_dir = "dbfs:/mnt/mids-w261/"
checkpoint_dir = f"{folder_path}/checkpoints"

# Utility functions for checkpointing
def save_checkpoint(df, checkpoint_name, description=""):
    """Save a dataframe checkpoint with metadata"""
    checkpoint_path = f"{checkpoint_dir}/{checkpoint_name}"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    print(f"[{timestamp}] Saving checkpoint: {checkpoint_name}")
    if description:
        print(f"Description: {description}")
    
    # Save the dataframe
    df.coalesce(10).write.mode("overwrite").parquet(checkpoint_path)
    
    # Save metadata
    row_count = df.count()
    col_count = len(df.columns)
    
    print(f"✓ Checkpoint saved: {row_count:,} rows x {col_count} columns")
    print(f"  Path: {checkpoint_path}")
    return checkpoint_path

def load_checkpoint(checkpoint_name):
    """Load a dataframe from checkpoint"""
    checkpoint_path = f"{checkpoint_dir}/{checkpoint_name}"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    print(f"[{timestamp}] Loading checkpoint: {checkpoint_name}")
    df = spark.read.parquet(checkpoint_path)
    
    row_count = df.count()
    col_count = len(df.columns)
    print(f"✓ Checkpoint loaded: {row_count:,} rows x {col_count} columns")
    
    return df

def checkpoint_exists(checkpoint_name):
    """Check if a checkpoint exists"""
    checkpoint_path = f"{checkpoint_dir}/{checkpoint_name}"
    try:
        # Try to list the path to see if it exists
        dbutils.fs.ls(checkpoint_path)
        return True
    except:
        return False
    
def list_checkpoints():
    """List all checkpoint folders in the checkpoint directory"""
    print(f"Listing checkpoints in: {checkpoint_dir}")
    try:
        files = dbutils.fs.ls(checkpoint_dir)
        checkpoint_names = [f.name.strip("/") for f in files if f.isDir()]
        if checkpoint_names:
            print("Available checkpoints:")
            for name in checkpoint_names:
                print(f"  • {name}")
        else:
            print("No checkpoints found.")
        return checkpoint_names
    except Exception as e:
        print(f"Error listing checkpoints: {e}")
        return []
    
def delete_checkpoint(checkpoint_name):
    """Delete a specific checkpoint folder"""
    checkpoint_path = f"{checkpoint_dir}/{checkpoint_name}"
    print(f"Deleting checkpoint: {checkpoint_path}")
    try:
        dbutils.fs.rm(checkpoint_path, recurse=True)
        print("✓ Checkpoint deleted.")
    except Exception as e:
        print(f"Failed to delete checkpoint: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC # Engineering Features Workbook 
# MAGIC
# MAGIC ### Omar Zu'bi
# MAGIC
# MAGIC ### Overview
# MAGIC This notebook implements a comprehensive feature engineering pipeline for predicting flight delays using historical flight data, weather information, and airport characteristics. The pipeline transforms raw airline data into a time series dataset optimized for machine learning analysis of the binary classification problem: Will a flight be delayed by 15+ minutes?
# MAGIC
# MAGIC ### Dataset Sources
# MAGIC The pipeline integrates three primary data sources:
# MAGIC
# MAGIC - Flight Data: Comprehensive airline operational data including schedules, delays, cancellations, and aircraft information
# MAGIC - Weather Data: Hourly meteorological observations from weather stations near airports
# MAGIC - Airport Data: Geographic information and weather station mappings for airports
# MAGIC
# MAGIC ### Pipeline Steps
# MAGIC - Step 0-1: Data Loading & Test Case
# MAGIC
# MAGIC   - Implements checkpoint system for pipeline resumption
# MAGIC   - Optional test case creation (single aircraft, one month) for development
# MAGIC
# MAGIC - Step 2: Data Cleaning
# MAGIC
# MAGIC   - Standardizes column names and data types across all datasets
# MAGIC   - Filters invalid records and handles missing values
# MAGIC   - Creates consistent temporal and spatial reference systems
# MAGIC
# MAGIC - Step 3: Aircraft Features
# MAGIC
# MAGIC   - Lag Features: Previous flight performance (taxi times, delays)
# MAGIC   - Turnaround Time: Time between aircraft arrival and next departure
# MAGIC   - Uses PySpark Window functions for time-ordered aircraft history
# MAGIC
# MAGIC - Step 4: Airport Congestion
# MAGIC
# MAGIC   - 2-Hour Lookback: Counts delays/cancellations at origin airport
# MAGIC   - Traffic Analysis: Tracks incoming flights affecting ground operations
# MAGIC   - Creates real-time airport activity metrics
# MAGIC
# MAGIC - Step 5: Historical Performance
# MAGIC
# MAGIC   - Multi-timeframe analysis: Lifetime, 90-days, and 14-days performance windows
# MAGIC   - On-time percentages: Arrival and departure performance by aircraft
# MAGIC   - Requires minimum 3 flights for statistical validity
# MAGIC
# MAGIC - Step 6: Weather Integration
# MAGIC
# MAGIC   - Temporal Alignment: Uses weather 2 hours before scheduled departure
# MAGIC   - Station Mapping: Links airports to nearest weather stations
# MAGIC   - Comprehensive Metrics: Wind, visibility, temperature, pressure, precipitation
# MAGIC   - Handles duplicate readings through averaging
# MAGIC
# MAGIC - Step 7: Final Feature Set
# MAGIC
# MAGIC   - Selects 75 features optimized for time series analysis
# MAGIC   - Adds temporal features (hour, weekend, season, holiday indicators)
# MAGIC   - Sorts by departure time for sequential analysis
# MAGIC
# MAGIC - Output: 7.4M+ flight records with 75 engineered features
# MAGIC

# COMMAND ----------

# Check path
notebook_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
print(f"Current notebook path: {notebook_path}")

# COMMAND ----------

# Load dependencies
import os
from datetime import datetime, timedelta

# PySpark core functions - standardized imports
from pyspark.sql import functions as F
from pyspark.sql.types import (
    IntegerType, DoubleType, StringType, TimestampType
)
from pyspark.sql.window import Window

from functools import reduce

# Custom Functions
#from checkpoint_functions import save_checkpoint, load_checkpoint, checkpoint_exists, list_checkpoints, delete_checkpoint

# COMMAND ----------

# List all available checkpoints
list_checkpoints()

# Check base dir
print(base_dir)

#checkpoint = ['06_weather_integrated_fixed']

# checkpoint_names = [
#     "02_airports_clean",
#     "02_flights_clean",
#     "02_weather_clean",
#     "03_aircraft_features",
#     "04_airport_congestion",
#     "05_historical_performance",
#     "06_weather_integrated_fixed",
#     "06a_weather_deduplicated",
#     "07_final_features"
# ]

# for checkpoint in checkpoint_names:
#     delete_checkpoint(checkpoint)

# COMMAND ----------

# =============================================================================
# STEP 00: LOADING TEST CASE OTHERWISE LOAD DATASETS
# =============================================================================

if (checkpoint_exists("00_flights_test") and 
    checkpoint_exists("00_weather_test")):
    print("Found existing test case. Loading...")
    df_flights = load_checkpoint("00_flights_test")
    df_weather = load_checkpoint("00_weather_test")
    df_airports = spark.read.parquet(base_dir + "datasets_final_project_2022/stations_data/stations_with_neighbors.parquet/")
elif (checkpoint_exists("02_flights_clean") and 
    checkpoint_exists("02_weather_clean") and 
    checkpoint_exists("02_airports_clean")):
    print("Found existing cleaned data checkpoints. Skipping step...")
else:
    print("Test files do not exist! Loading raw data.")

    print("Loading datasets...")
    df_flights = spark.read.parquet(base_dir + "datasets_final_project_2022/parquet_airlines_data/")
    df_weather = spark.read.parquet(base_dir + "datasets_final_project_2022/parquet_weather_data/")
    df_airports = spark.read.parquet(base_dir + "datasets_final_project_2022/stations_data/stations_with_neighbors.parquet/")

    print(f"Flights: {df_flights.count():,} rows")
    print(f"Weather: {df_weather.count():,} rows")
    print(f"Airports: {df_airports.count():,} rows")

# COMMAND ----------

# # =============================================================================
# # STEP 1: (OPTIONAL) GENERATING TEST CASE - 1 unique flight tail # for 1 month of data
# # =============================================================================

# # Check if we already have our test case
# if (checkpoint_exists("00_flights_test") and 
#     checkpoint_exists("00_weather_test")):
#     print("Found existing test case. Should have been loaded from prior step.")
# else:
#     print("Generating test files")

#     # Get first unique tail_num
#     first_tail_num = df_flights \
#         .filter((F.col("TAIL_NUM").isNotNull()) & (F.col("TAIL_NUM") != "")) \
#         .select("TAIL_NUM") \
#         .distinct() \
#         .orderBy("TAIL_NUM") \
#         .limit(1) \
#         .collect()[0]["TAIL_NUM"]

#     print(f"First unique tail number: {first_tail_num}")

#     df_flights = df_flights.filter(F.col("TAIL_NUM") == first_tail_num)

#     # Get the first date (earliest flight date)
#     first_date_str = df_flights.agg(F.min("FL_DATE")).collect()[0][0]
#     print(f"First date: {first_date_str}")
    
#     # Datetime value
#     first_date = datetime.strptime(first_date_str, "%Y-%m-%d").date()
#     first_month = first_date.month
#     first_year = first_date.year

#     # Filter df_flights for the same year and month
#     df_flights = df_flights.filter(
#         (F.year(F.col("FL_DATE")) == first_year) & (F.month(F.col("FL_DATE")) == first_month)
#     )
#     save_checkpoint(df_flights, "00_flights_test", "Test case of 1 unique flight path in the first month of 3M dataset")

#     # Filter df_weather for the same year and month
#     df_weather = df_weather.filter(
#         (F.year(F.col("DATE")) == first_year) & (F.month(F.col("DATE")) == first_month)
#     )
#     save_checkpoint(df_weather, "00_weather_test", "Weather associated to test case")

#     print("✓ Step 00 Complete: Test dataset created!")

# COMMAND ----------

# =============================================================================
# STEP 2: CLEAN AND STANDARDIZE DATASETS
# =============================================================================

# Check if we already have cleaned data checkpoints
if (checkpoint_exists("02_flights_clean") and 
    checkpoint_exists("02_weather_clean") and 
    checkpoint_exists("02_airports_clean")):
    print("Found existing cleaned data checkpoints. Loading...")
    df_flights_clean = load_checkpoint("02_flights_clean")
    df_weather_clean = load_checkpoint("02_weather_clean")
    df_airports_clean = load_checkpoint("02_airports_clean")
else:
    print("Cleaning flights data...")
    
    # Clean flights data
    df_flights_clean = df_flights.select(
        # Target variable
        F.col("DEP_DEL15").cast(DoubleType()).alias("DepDel15"),
        
        # Time & Scheduling Features
        F.col("FL_DATE").alias("FlightDate"),
        F.col("YEAR").cast(IntegerType()).alias("Year"),
        F.col("QUARTER").cast(IntegerType()).alias("Quarter"),
        F.col("MONTH").cast(IntegerType()).alias("Month"),
        F.col("DAY_OF_MONTH").cast(IntegerType()).alias("DayofMonth"),
        F.col("DAY_OF_WEEK").cast(IntegerType()).alias("DayOfWeek"),
        F.col("CRS_DEP_TIME").cast(IntegerType()).alias("CRSDepTime"),
        F.col("DEP_TIME_BLK").cast(StringType()).alias("DepTimeBlk"),
        
        # Flight & Airline Features
        F.col("OP_UNIQUE_CARRIER").cast(StringType()).alias("Reporting_Airline"),
        F.col("OP_CARRIER_FL_NUM").cast(IntegerType()).alias("Flight_Number_Reporting_Airline"),
        F.col("TAIL_NUM").cast(StringType()).alias("Tail_Number"),
        
        # Airport Features
        F.col("ORIGIN").cast(StringType()).alias("Origin"),
        F.col("ORIGIN_CITY_NAME").cast(StringType()).alias("OriginCityName"),
        F.col("ORIGIN_STATE_ABR").cast(StringType()).alias("OriginState"),
        F.col("DEST").cast(StringType()).alias("Dest"),
        F.col("DEST_CITY_NAME").cast(StringType()).alias("DestCityName"),
        F.col("DEST_STATE_ABR").cast(StringType()).alias("DestState"),
        F.col("ORIGIN_AIRPORT_ID").cast(IntegerType()).alias("OriginAirportID"),
        F.col("DEST_AIRPORT_ID").cast(IntegerType()).alias("DestAirportID"),
        F.col("ORIGIN_WAC").cast(IntegerType()).alias("OriginWac"),
        F.col("DEST_WAC").cast(IntegerType()).alias("DestWac"),
        
        # Distance & Duration Features
        F.col("DISTANCE").cast(DoubleType()).alias("Distance"),
        F.col("DISTANCE_GROUP").cast(IntegerType()).alias("DistanceGroup"),
        F.col("CRS_ELAPSED_TIME").cast(DoubleType()).alias("CRSElapsedTime"),
        
        # Helper columns for feature engineering
        F.col("ARR_DELAY").cast(DoubleType()),
        F.col("TAXI_IN").cast(DoubleType()),
        F.col("TAXI_OUT").cast(DoubleType()),
        F.col("ARR_TIME").cast(IntegerType()),
        F.col("DEP_TIME").cast(IntegerType()),
        F.col("CRS_ARR_TIME").cast(IntegerType()),
        F.col("CANCELLED").cast(DoubleType())
    ).filter(
        (F.col("FL_DATE").isNotNull()) & 
        (F.col("ORIGIN").isNotNull()) & 
        (F.col("DEST").isNotNull()) &
        (F.col("CRS_DEP_TIME").isNotNull())
    ).dropDuplicates()

    save_checkpoint(df_flights_clean, "02_flights_clean", "Cleaned flights data with standardized columns")
    
    print("Cleaning weather data...")
    
    # Clean weather data
    df_weather_clean = df_weather.select(
        F.to_date(F.col("DATE"), "yyyy-MM-dd'T'HH:mm:ss").alias("weather_date"),
        F.hour(F.to_timestamp(F.col("DATE"), "yyyy-MM-dd'T'HH:mm:ss")).alias("weather_hour"),
        
        # Weather Features
        F.col("HourlyWindSpeed").cast(DoubleType()),
        F.col("HourlyWindGustSpeed").cast(DoubleType()),
        F.col("HourlyVisibility").cast(DoubleType()),
        F.col("HourlySkyConditions").cast(StringType()),
        F.col("HourlyPrecipitation").cast(DoubleType()),
        F.col("HourlyPresentWeatherType").cast(StringType()),
        F.col("HourlyDryBulbTemperature").cast(DoubleType()),
        F.col("HourlyWetBulbTemperature").cast(DoubleType()),
        F.col("HourlyDewPointTemperature").cast(DoubleType()),
        F.col("HourlyRelativeHumidity").cast(DoubleType()),
        F.col("HourlySeaLevelPressure").cast(DoubleType()),
        
        # Spatial & Station Features
        F.col("STATION").cast(StringType()),
        F.col("LATITUDE").cast(DoubleType()).alias("origin_station_lat"),
        F.col("LONGITUDE").cast(DoubleType()).alias("origin_station_lon")
    ).filter(
        (F.col("DATE").isNotNull()) & 
        (F.col("STATION").isNotNull())
    ).dropDuplicates()
    
    save_checkpoint(df_weather_clean, "02_weather_clean", "Cleaned weather data with standardized columns")
    
    print("Cleaning airports data...")
    
    # Clean airports data
    df_airports_clean = df_airports.select(
        F.col("station_id").cast(StringType()),
        F.col("lat").alias("origin_airport_lat"),
        F.col("lon").alias("origin_airport_lon"),
        F.col("neighbor_call").alias("airport_code"),
        F.col("distance_to_neighbor").alias("origin_station_dis")
    ).filter(
        (F.col("neighbor_call").isNotNull()) & 
        (F.col("station_id").isNotNull()) &
        (F.col("neighbor_call") != "")
    ).dropDuplicates()
    
    save_checkpoint(df_airports_clean, "02_airports_clean", "Cleaned airports data with standardized columns")

print("✓ Step 2 Complete: All datasets cleaned and standardized")

# COMMAND ----------

# =============================================================================
# STEP 3: ENGINEER AIRCRAFT-SPECIFIC FEATURES
# =============================================================================

if checkpoint_exists("03_aircraft_features"):
    print("Found existing aircraft features checkpoint. Loading...")
    df_flights_with_aircraft_features = load_checkpoint("03_aircraft_features")
else:
    print("Engineering aircraft-specific features...")
    
    # Create window for aircraft history (ordered by date and time)
    aircraft_window = Window.partitionBy("Tail_Number").orderBy("FlightDate", "CRSDepTime")
    
    # Create features for previous flight performance
    df_flights_with_aircraft_features = df_flights_clean.withColumn(
        "Prev_TaxiIn", 
        F.lag("TAXI_IN", 1).over(aircraft_window)
    ).withColumn(
        "Prev_TaxiOut", 
        F.lag("TAXI_OUT", 1).over(aircraft_window)
    ).withColumn(
        "Prev_ArrDelay", 
        F.lag("ARR_DELAY", 1).over(aircraft_window)
    ).withColumn(
        "Prev_ArrTime", 
        F.lag("ARR_TIME", 1).over(aircraft_window)
    ).withColumn(
        "Prev_FlightDate", 
        F.lag("FlightDate", 1).over(aircraft_window)
    )
    
    # Calculate turnaround time (time between previous arrival and current scheduled departure)
    df_flights_with_aircraft_features = df_flights_with_aircraft_features.withColumn(
        "Turnaround_Time",
        F.when(
            F.col("Prev_ArrTime").isNotNull() &
            (F.datediff(F.col("FlightDate"), F.col("Prev_FlightDate")) == 0),
            F.col("CRSDepTime") - F.col("Prev_ArrTime")
        ).when(
            F.col("Prev_ArrTime").isNotNull() &
            (F.datediff(F.col("FlightDate"), F.col("Prev_FlightDate")) == 1),
            (F.col("CRSDepTime") + 2400) - F.col("Prev_ArrTime")
        ).otherwise(None))

    save_checkpoint(df_flights_with_aircraft_features, "03_aircraft_features", 
                   "Flights data with aircraft-specific lag features and turnaround time")

print("✓ Step 3 Complete: Aircraft-specific features engineered")

# COMMAND ----------

# =============================================================================
# STEP 4: ENGINEER AIRPORT CONGESTION FEATURES
# =============================================================================

if checkpoint_exists("04_airport_congestion"):
    print("Found existing airport congestion checkpoint. Loading...")
    df_with_oncoming = load_checkpoint("04_airport_congestion")
else:
    print("Engineering airport-level congestion features...")

    # Create all necessary columns upfront
    df_prepared = df_flights_with_aircraft_features.withColumn(
        "departure_datetime",
        F.to_timestamp(F.concat_ws("", F.col("FlightDate"), F.lpad(F.col("CRSDepTime"), 4, "0")), "yyyy-MM-ddHHmm")
    ).withColumn(
        "arrival_datetime",
        F.to_timestamp(F.concat_ws("", F.col("FlightDate"), F.lpad(F.col("CRS_ARR_TIME"), 4, "0")), "yyyy-MM-ddHHmm")
    ).withColumn(
        "delay_flag", F.when(F.col("DepDel15") == 1, 1).otherwise(0)
    ).withColumn(
        "cancel_flag", F.when(F.col("CANCELLED") == 1, 1).otherwise(0)
    ).withColumn(
        "departure_epoch", F.unix_timestamp(F.col("departure_datetime"))
    ).withColumn(
        "arrival_epoch", F.unix_timestamp(F.col("arrival_datetime"))
    ).filter(F.col("departure_datetime").isNotNull())

    # Calculate departure-based metrics using window functions
    departure_window = Window.partitionBy("Origin").orderBy("departure_epoch").rangeBetween(-7200, -1)
    
    df_with_departure_metrics = df_prepared.withColumn(
        "Num_airport_wide_delays",
        F.coalesce(F.sum("delay_flag").over(departure_window), F.lit(0))
    ).withColumn(
        "Num_airport_wide_cancelations",
        F.coalesce(F.sum("cancel_flag").over(departure_window), F.lit(0))
    )

    # Calculate arrival-based metrics using window functions
    # Create a unified view of all airport events (departures + arrivals)
    
    # Departures as events
    departure_events = df_with_departure_metrics.select(
        F.col("Origin").alias("airport"),
        F.col("departure_epoch").alias("event_time"),
        F.lit("departure").alias("event_type"),
        F.col("Tail_Number").alias("aircraft"),
        # Include flight-specific columns for later join
        F.col("FlightDate"),
        F.col("CRSDepTime"),
        *[c for c in df_with_departure_metrics.columns if c not in ["Origin", "departure_epoch", "Tail_Number"]]
    )
    
    # Arrivals as events
    arrival_events = df_prepared.filter(F.col("arrival_datetime").isNotNull()).select(
        F.col("Dest").alias("airport"),
        F.col("arrival_epoch").alias("event_time"),
        F.lit("arrival").alias("event_type"),
        F.col("Tail_Number").alias("aircraft"),
        F.lit(None).alias("FlightDate").cast("string"),
        F.lit(None).alias("CRSDepTime").cast("integer")
    )
    
    # Combine all events and sort by time
    all_events = departure_events.select("airport", "event_time", "event_type", "aircraft").union(
        arrival_events.select("airport", "event_time", "event_type", "aircraft")
    )
    
    # Use window function to count arrivals in preceding 2-hour window
    arrival_count_window = Window.partitionBy("airport").orderBy("event_time").rangeBetween(-7200, 0)
    
    events_with_counts = all_events.withColumn(
        "arrivals_in_window",
        F.sum(F.when(F.col("event_type") == "arrival", 1).otherwise(0)).over(arrival_count_window)
    ).withColumn(
        "own_arrival_adjustment",
        F.when(F.col("event_type") == "arrival", 1).otherwise(0)
    ).withColumn(
        "Oncoming_flights",
        F.greatest(F.col("arrivals_in_window") - F.col("own_arrival_adjustment"), F.lit(0))
    )
    
    # Join back with departure data to get oncoming flights for each departure
    df_with_oncoming = df_with_departure_metrics.join(
        events_with_counts.filter(F.col("event_type") == "departure").select(
            "airport", "event_time", "aircraft", "Oncoming_flights"
        ),
        (F.col("Origin") == F.col("airport")) & 
        (F.col("departure_epoch") == F.col("event_time")) & 
        (F.col("Tail_Number") == F.col("aircraft")),
        "left"
    ).drop("airport", "event_time", "aircraft", "departure_epoch", "arrival_epoch")
    
    # Handle nulls
    df_with_oncoming = df_with_oncoming.fillna({"Oncoming_flights": 0})

    save_checkpoint(df_with_oncoming, "04_airport_congestion",
                    "Complete airport congestion features")

print("✓ Step 4 Complete: Airport congestion features engineered")

# COMMAND ----------

# =============================================================================
# STEP 5: ENGINEER HISTORICAL PERFORMANCE FEATURES
# =============================================================================

if checkpoint_exists("05_historical_performance"):
    print("Found existing historical performance checkpoint. Loading...")
    df_with_ontime_metrics = load_checkpoint("05_historical_performance")
else:
    print("Engineering historical performance features...")
    
    # Data preparation & type conversion
    df_prepared = df_with_oncoming \
        .withColumn("FlightDate", F.to_date("FlightDate")) \
        .withColumn("CRSDepTime", F.col("CRSDepTime").cast("int")) \
        .withColumn("departure_epoch", F.col("departure_datetime").cast("long")) \
        .withColumn(
            "temp_ontime_arrival",
            F.when(F.coalesce(F.col("ARR_DELAY"), F.lit(0)) <= 15, 1).otherwise(0)
        ).withColumn(
            "temp_ontime_departure",
            F.when(F.col("DepDel15") == 0, 1).otherwise(0)
        ).cache()  # Cache since we'll use this multiple times
    
    # Single window definition with optimized range
    w_base = Window.partitionBy("Tail_Number").orderBy("departure_epoch")
    
    # Define time boundaries in seconds
    TWO_HOURS = 7200
    NINETY_DAYS = 90 * 24 * 60 * 60  # 90 days in seconds
    TWO_WEEKS = 14 * 24 * 60 * 60    # 14 days in seconds
    
    # Compute all historical metrics in one pass
    df_with_ontime_metrics = df_prepared.withColumn(
        # EXPANDING WINDOW - All historical data (2+ hours before)
        "historical_ontime_arrival_pct",
        F.when(
            F.count("temp_ontime_arrival").over(
                w_base.rangeBetween(Window.unboundedPreceding, -TWO_HOURS)
            ) >= 3,
            F.avg("temp_ontime_arrival").over(
                w_base.rangeBetween(Window.unboundedPreceding, -TWO_HOURS)
            ) * 100
        ).otherwise(None)
    ).withColumn(
        "historical_ontime_departure_pct",
        F.when(
            F.count("temp_ontime_departure").over(
                w_base.rangeBetween(Window.unboundedPreceding, -TWO_HOURS)
            ) >= 3,
            F.avg("temp_ontime_departure").over(
                w_base.rangeBetween(Window.unboundedPreceding, -TWO_HOURS)
            ) * 100
        ).otherwise(None)
    ).withColumn(
        # RECENT WINDOW - Last two weeks (2+ hours before)
        "recent_ontime_arrival_pct",
        F.when(
            F.count("temp_ontime_arrival").over(
                w_base.rangeBetween(-TWO_WEEKS, -TWO_HOURS)
            ) >= 3,
            F.avg("temp_ontime_arrival").over(
                w_base.rangeBetween(-TWO_WEEKS, -TWO_HOURS)
            ) * 100
        ).otherwise(None)
    ).withColumn(
        "recent_ontime_departure_pct",
        F.when(
            F.count("temp_ontime_departure").over(
                w_base.rangeBetween(-TWO_WEEKS, -TWO_HOURS)
            ) >= 3,
            F.avg("temp_ontime_departure").over(
                w_base.rangeBetween(-TWO_WEEKS, -TWO_HOURS)
            ) * 100
        ).otherwise(None)
    ).withColumn(
        # SHORT-TERM WINDOW - Last 90 days (2+ hours before)
        "short_term_ontime_arrival_pct",
        F.when(
            F.count("temp_ontime_arrival").over(
                w_base.rangeBetween(-NINETY_DAYS, -TWO_HOURS)
            ) >= 3,
            F.avg("temp_ontime_arrival").over(
                w_base.rangeBetween(-NINETY_DAYS, -TWO_HOURS)
            ) * 100
        ).otherwise(None)
    ).withColumn(
        "short_term_ontime_departure_pct",
        F.when(
            F.count("temp_ontime_departure").over(
                w_base.rangeBetween(-NINETY_DAYS, -TWO_HOURS)
            ) >= 3,
            F.avg("temp_ontime_departure").over(
                w_base.rangeBetween(-NINETY_DAYS, -TWO_HOURS)
            ) * 100
        ).otherwise(None)
    ).withColumn(
        # LEGACY COLUMNS (backwards compatibility)
        "OntimeArrivalPct", F.col("historical_ontime_arrival_pct")
    ).withColumn(
        "OntimeDeparturePct", F.col("historical_ontime_departure_pct")
    ).drop(
        "temp_ontime_arrival", 
        "temp_ontime_departure", 
        "departure_epoch"
    )
    
    # Unpersist cached DataFrame
    df_prepared.unpersist()
    
    save_checkpoint(df_with_ontime_metrics, "05_historical_performance", 
                   "Historical on-time performance metrics by tail number")

print("✓ Step 5 Complete: Historical performance features engineered")

# COMMAND ----------

df_weather_clean = load_checkpoint("02_weather_clean")
df_airports_clean = load_checkpoint("02_airports_clean")
df_with_ontime_metrics = load_checkpoint("05_historical_performance")

# COMMAND ----------

# =============================================================================
# STEP 6: INTEGRATE WEATHER DATA
# =============================================================================

def deduplicate_weather_by_averaging(df_weather_clean_filtered):
    """Average duplicate weather readings"""
    return df_weather_clean_filtered.groupBy(
        "STATION", "weather_date", "weather_hour"
    ).agg(
        F.avg("HourlyWindSpeed").alias("HourlyWindSpeed"),
        F.avg("HourlyWindGustSpeed").alias("HourlyWindGustSpeed"),
        F.avg("HourlyVisibility").alias("HourlyVisibility"),
        F.avg("HourlyDryBulbTemperature").alias("HourlyDryBulbTemperature"),
        F.avg("HourlyWetBulbTemperature").alias("HourlyWetBulbTemperature"),
        F.avg("HourlyDewPointTemperature").alias("HourlyDewPointTemperature"),
        F.avg("HourlyRelativeHumidity").alias("HourlyRelativeHumidity"),
        F.avg("HourlySeaLevelPressure").alias("HourlySeaLevelPressure"),
        F.avg("origin_station_lat").alias("origin_station_lat"),
        F.avg("origin_station_lon").alias("origin_station_lon"),
        F.first("HourlySkyConditions").alias("HourlySkyConditions"),
        F.first("HourlyPresentWeatherType").alias("HourlyPresentWeatherType"),
        F.first("HourlyPrecipitation").alias("HourlyPrecipitation")
    )

for year in range(2015, 2022):  # 2015–2021
    checkpoint_name = f"06_weather_integrated_{year}"
    
    if checkpoint_exists(checkpoint_name):
        print(f"✅ Checkpoint exists for {year}, skipping.")
        continue

    print(f"\n=== Processing Year: {year} ===")

    # 1. Filter flight and weather data
    df_flights_year = df_with_ontime_metrics \
        .filter(F.year("FlightDate") == year) \
        .dropDuplicates() \
        .cache()

    df_weather_year = df_weather_clean \
        .filter(F.year("weather_date") == year)

    # 2. Deduplicate weather
    df_weather_deduplicated = deduplicate_weather_by_averaging(df_weather_year)

    # 3. Get min/max flight dates
    min_max_dates = df_flights_year.agg(
        F.min("FlightDate").alias("min_date"),
        F.max("FlightDate").alias("max_date")
    ).collect()[0]

    min_date = min_max_dates["min_date"]
    max_date = min_max_dates["max_date"]

    # 4. Broadcast airport to station mapping
    airport_to_station = df_airports_clean \
        .filter(F.col("airport_code").startswith("K")) \
        .select(
            F.regexp_replace("airport_code", "^K", "").alias("airport_code"),
            F.col("station_id").alias("weather_station")
        ).dropna().dropDuplicates()

    airport_to_station_broadcast = F.broadcast(airport_to_station)

    # 5. Join weather with airport code
    weather_joined_df = df_weather_deduplicated.join(
        airport_to_station_broadcast,
        df_weather_deduplicated["STATION"] == airport_to_station["weather_station"],
        "inner"
    ).select(
        "airport_code",
        "weather_date",
        "weather_hour",
        *[c for c in df_weather_deduplicated.columns if c not in ("STATION", "weather_date", "weather_hour")]
    )

    # 6. Filter weather by ±1 day of flight dates
    weather_joined_df = weather_joined_df.filter(
        F.col("weather_date").between(
            F.date_sub(F.lit(min_date), 1),
            F.date_add(F.lit(max_date), 1)
        )
    )

    # 7. Join with only relevant airports
    distinct_origins = df_flights_year.select("origin").distinct()

    filtered_weather_df = weather_joined_df.join(
        F.broadcast(distinct_origins),
        weather_joined_df["airport_code"] == distinct_origins["origin"],
        "inner"
    ).drop("origin", "origin_station_lat", "origin_station_lon", "weather_station") \
    .dropDuplicates(["airport_code", "weather_date", "weather_hour"])

    # 8. Add lookup columns
    df_flights_year = df_flights_year.withColumn(
        "flight_hour", F.hour("departure_datetime")
    ).withColumn(
        "weather_hour_lookup", ((F.col("flight_hour") - 2) % 24)
    ).withColumn(
        "weather_date_lookup", F.when(
            F.col("flight_hour") < 2,
            F.date_sub(F.col("FlightDate"), 1)
        ).otherwise(F.col("FlightDate"))
    )

    # 9. Join flights with weather
    df_with_weather_year = df_flights_year.join(
        filtered_weather_df,
        (df_flights_year["origin"] == filtered_weather_df["airport_code"]) &
        (df_flights_year["weather_date_lookup"] == filtered_weather_df["weather_date"]) &
        (df_flights_year["weather_hour_lookup"] == filtered_weather_df["weather_hour"]),
        "left"
    )

    # 10. Save the result for this year
    save_checkpoint(df_with_weather_year.repartition(200), checkpoint_name, f"Weather data integrated for {year}")

    print(f"✅ Saved weather-integrated data for {year}")

    # 11. Cleanup memory
    df_flights_year.unpersist()
    del df_flights_year, df_weather_year, df_weather_deduplicated
    del min_max_dates, min_date, max_date
    del airport_to_station, airport_to_station_broadcast
    del weather_joined_df, distinct_origins, filtered_weather_df
    del df_with_weather_year
    import gc
    gc.collect()


# COMMAND ----------

# # =============================================================================
# # STEP 6: INTEGRATE WEATHER DATA
# # =============================================================================

# def deduplicate_weather_by_averaging(df_weather_clean_filtered):
#     """Average duplicate weather readings"""
#     return df_weather_clean_filtered.groupBy(
#         "STATION", "weather_date", "weather_hour"
#     ).agg(
#         F.avg("HourlyWindSpeed").alias("HourlyWindSpeed"),
#         F.avg("HourlyWindGustSpeed").alias("HourlyWindGustSpeed"),
#         F.avg("HourlyVisibility").alias("HourlyVisibility"),
#         F.avg("HourlyDryBulbTemperature").alias("HourlyDryBulbTemperature"),
#         F.avg("HourlyWetBulbTemperature").alias("HourlyWetBulbTemperature"),
#         F.avg("HourlyDewPointTemperature").alias("HourlyDewPointTemperature"),
#         F.avg("HourlyRelativeHumidity").alias("HourlyRelativeHumidity"),
#         F.avg("HourlySeaLevelPressure").alias("HourlySeaLevelPressure"),
#         F.avg("origin_station_lat").alias("origin_station_lat"),
#         F.avg("origin_station_lon").alias("origin_station_lon"),
#         F.first("HourlySkyConditions").alias("HourlySkyConditions"),
#         F.first("HourlyPresentWeatherType").alias("HourlyPresentWeatherType"),
#         F.first("HourlyPrecipitation").alias("HourlyPrecipitation")
#     )

# if checkpoint_exists("06_weather_integrated_fixed"):
#     df_with_weather = load_checkpoint("06_weather_integrated_fixed")
# else:

#     # Deduplicate and cache
#     if checkpoint_exists("06a_weather_deduplicated"):
#         df_weather_deduplicated = load_checkpoint("06a_weather_deduplicated")
#     else:
#         df_weather_clean = load_checkpoint("02_weather_clean")
#         df_weather_deduplicated = deduplicate_weather_by_averaging(df_weather_clean)
#         save_checkpoint(df_weather_deduplicated, "06a_weather_deduplicated", "Deduplicated weather data")

    
#     # 1. Cache df_with_ontime_metrics if reused
#     df_with_ontime_metrics = df_with_ontime_metrics.dropDuplicates().cache()

#     # 2. Get min/max flight dates (this should be fast)
#     min_max_dates = df_with_ontime_metrics.agg(
#         F.min("FlightDate").alias("min_date"),
#         F.max("FlightDate").alias("max_date")
#     ).collect()[0]

#     min_date = min_max_dates["min_date"]
#     max_date = min_max_dates["max_date"]

#     # 3. Prepare airport -> weather station mapping and broadcast it
#     airport_to_station = df_airports_clean \
#         .filter(F.col("airport_code").startswith("K")) \
#         .select(
#             F.regexp_replace("airport_code", "^K", "").alias("airport_code"),
#             F.col("station_id").alias("weather_station")
#         ).dropna().dropDuplicates()

#     airport_to_station_broadcast = F.broadcast(airport_to_station)

#     # 4. Join weather to airport stations early (improve filtering efficiency)
#     weather_joined_df = df_weather_deduplicated.join(
#         airport_to_station_broadcast,
#         df_weather_deduplicated["STATION"] == airport_to_station["weather_station"],
#         "inner"
#     ).select(
#         "airport_code",
#         "weather_date",
#         "weather_hour",
#         *[c for c in df_weather_deduplicated.columns if c not in ("STATION", "weather_date", "weather_hour")]
#     )

#     # 5. Filter weather by date range (expand by ±1 day)
#     weather_joined_df = weather_joined_df.filter(
#         F.col("weather_date").between(
#             F.date_sub(F.lit(min_date), 1),
#             F.date_add(F.lit(max_date), 1)
#         )
#     )

#     # 6. Filter weather_joined_df to include only airports present in flights
#     distinct_origins = df_with_ontime_metrics.select("origin").distinct()

#     # Use broadcast join to avoid shuffling
#     filtered_weather_df = weather_joined_df.join(
#         F.broadcast(distinct_origins),
#         weather_joined_df["airport_code"] == distinct_origins["origin"],
#         "inner"
#     ).drop("origin", "origin_station_lat", "origin_station_lon", "weather_station") \
#     .dropDuplicates(["airport_code", "weather_date", "weather_hour"])  # Avoid excessive rows

#     # 7. Prepare flight dataset with weather lookup columns
#     df_with_ontime_metrics = df_with_ontime_metrics.withColumn(
#         "flight_hour", F.hour("departure_datetime")
#     ).withColumn(
#         "weather_hour_lookup", ((F.col("flight_hour") - 2) % 24)
#     ).withColumn(
#         "weather_date_lookup", F.when(
#             F.col("flight_hour") < 2,
#             F.date_sub(F.col("FlightDate"), 1)
#         ).otherwise(F.col("FlightDate"))
#     )

#     # 8. Join flights with weather
#     df_with_weather = df_with_ontime_metrics.join(
#         filtered_weather_df,
#         (df_with_ontime_metrics["origin"] == filtered_weather_df["airport_code"]) &
#         (df_with_ontime_metrics["weather_date_lookup"] == filtered_weather_df["weather_date"]) &
#         (df_with_ontime_metrics["weather_hour_lookup"] == filtered_weather_df["weather_hour"]),
#         "left"
#     )

#     save_checkpoint(df_with_weather, "06_weather_integrated_fixed", "Weather data integrated with deduplication")

# print("Weather integration completed with deduplication")

# COMMAND ----------

from functools import reduce

# List of years to load
years = list(range(2015, 2020))

# Load all yearly checkpoints
dfs = [load_checkpoint(f"06_weather_integrated_{year}") for year in years]

# Combine them using unionByName (preserves schema order and columns)
df_with_weather = reduce(lambda df1, df2: df1.unionByName(df2), dfs)

# Optional: Save the combined dataset as a new checkpoint
save_checkpoint(df_with_weather.repartition(200), "06_weather_integrated_all_years", "Merged weather data for 2015–2019")

print("✅ Merged DataFrame created and saved as '06_weather_integrated_all_years'")

# COMMAND ----------

# =============================================================================
# STEP 7: CREATE FINAL FEATURE SET FOR TIME SERIES STUDY
# =============================================================================
if checkpoint_exists("07_final_features"):
    print("Found existing final features checkpoint. Loading...")
    df_final = load_checkpoint("07_final_features")
else:
    print("Selecting final feature set for time series analysis...")
    
    # Select features
    final_features = [
        # === TARGET VARIABLE ===
        "DepDel15",
        
        # === TIME & TEMPORAL FEATURES (Critical for Time Series) ===
        "FlightDate",                    # Primary time dimension
        "Year",
        "Quarter", 
        "Month",
        "DayofMonth",
        "DayOfWeek",                     # Day of week seasonality
        "CRSDepTime",                    # Time of day (important for delays)
        "DepTimeBlk",                    # Time block categorization
        "departure_datetime",            # Full timestamp
        "arrival_datetime",              # Arrival timestamp
        
        # === FLIGHT & AIRLINE IDENTIFIERS ===
        "Reporting_Airline",             # Airline effects
        "Flight_Number_Reporting_Airline",
        "Tail_Number",                   # Aircraft-specific effects
        
        # === ROUTE & AIRPORT FEATURES ===
        "Origin",                        # Origin airport code
        "OriginCityName",
        "OriginState",
        "Dest",                          # Destination airport
        "DestCityName", 
        "DestState",
        "OriginAirportID",
        "DestAirportID",
        "OriginWac",                     # World Area Codes
        "DestWac",
        
        # === OPERATIONAL FEATURES ===
        "Distance",                      # Flight distance
        "DistanceGroup",
        "CRSElapsedTime",               # Scheduled flight time
        "ARR_DELAY",                    # Arrival delay
        "TAXI_IN",                      # Taxi in time
        "TAXI_OUT",                     # Taxi out time
        "ARR_TIME",                     # Actual arrival time
        "DEP_TIME",                     # Actual departure time
        "CRS_ARR_TIME",                 # Scheduled arrival time
        "CANCELLED",                    # Cancellation flag
        "delay_flag",                   # Delay indicator
        "cancel_flag",                  # Cancellation indicator
        
        # === WEATHER FEATURES ===
        "HourlyWindSpeed",
        "HourlyWindGustSpeed",
        "HourlyVisibility",
        "HourlySkyConditions",
        "HourlyPrecipitation", 
        "HourlyPresentWeatherType",
        "HourlyDryBulbTemperature",
        "HourlyWetBulbTemperature",
        "HourlyDewPointTemperature",
        "HourlyRelativeHumidity",
        "HourlySeaLevelPressure",
        
        # === HISTORICAL/ENGINEERED FEATURES ===
        "Prev_TaxiIn",                   # Previous flight performance
        "Prev_TaxiOut",
        "Prev_ArrDelay",
        "Prev_ArrTime",                  # Previous arrival time
        "Prev_FlightDate",               # Previous flight date
        "Turnaround_Time",               # Aircraft turnaround time
        "Num_airport_wide_delays",       # Airport congestion
        "Num_airport_wide_cancelations",
        "Oncoming_flights",              # Airport traffic
        
        # === ON-TIME PERFORMANCE METRICS ===
        "historical_ontime_arrival_pct",     # Long-term trends
        "historical_ontime_departure_pct",
        "recent_ontime_arrival_pct",          # Recent performance
        "recent_ontime_departure_pct", 
        "short_term_ontime_arrival_pct",      # Short-term trends
        "short_term_ontime_departure_pct",
        "OntimeArrivalPct",                   # Current metrics
        "OntimeDeparturePct"
    ]
    
    # Remove features that don't exist 
    available_columns = df_with_weather.columns
    final_features_filtered = [col for col in final_features if col in available_columns]
    
    # Report which features were not found
    missing_features = [col for col in final_features if col not in available_columns]
    if missing_features:
        print(f"⚠️  The following features were not found in the dataset: {missing_features}")
        print("Proceeding with available features only.")
    
    print(f"Selected {len(final_features_filtered)} features for time series analysis")
    
    # Create final dataset
    df_final = df_with_weather.select(*final_features_filtered)
    
    # Add time series specific features
    print("Creating additional time series features...")
    
    df_final = df_final.withColumn(
        "Hour", F.hour(F.col("departure_datetime"))
    ).withColumn(
        "IsWeekend", (F.col("DayOfWeek").isin([1, 7])).cast("integer")  # Sunday=1, Saturday=7
    ).withColumn(
        "IsBusinessHours", 
        ((F.col("Hour") >= 6) & (F.col("Hour") <= 22)).cast("integer")  # 6AM-10PM
    ).withColumn(
        "SeasonQuarter", 
        F.when(F.col("Month").isin([12, 1, 2]), "Winter")
        .when(F.col("Month").isin([3, 4, 5]), "Spring") 
        .when(F.col("Month").isin([6, 7, 8]), "Summer")
        .otherwise("Fall")
    ).withColumn(
        "IsHolidayMonth",
        F.col("Month").isin([11, 12, 1]).cast("integer")  # Holiday season
    )
    
    # Sort by time for time series analysis
    df_final = df_final.orderBy("departure_datetime")
    
    save_checkpoint(df_final, "07_final_features", 
                   "Final time series dataset with engineered features")

print("✓ Step 7 Complete: Time series feature set created")


# COMMAND ----------

# =============================================================================
# FINAL DATASET SUMMARY
# =============================================================================
print("\n=== FINAL DATASET SUMMARY ===")
print(f"Total rows: {df_final.count()}")
print(f"Total features: {len(df_final.columns)}")

# Show data types
print("\nFeature types:")
for field in df_final.schema.fields:
    print(f"  {field.name}: {field.dataType}")

# Show null counts for key features
print("\nNull value summary:")
null_counts = df_final.select([
    F.count(F.when(F.col(c).isNull(), c)).alias(c) 
    for c in df_final.columns[:20]  # First 20 columns
]).collect()[0].asDict()

for col_name, null_count in null_counts.items():
    if null_count > 0:
        print(f"  {col_name}: {null_count} nulls")

print("\nTime range:")
df_final.select(
    F.min("departure_datetime").alias("start_date"),
    F.max("departure_datetime").alias("end_date") 
).show()

print("✅ Dataset ready for time series analysis!")

# COMMAND ----------

null_counts_df = df_final.select([
    F.count(F.when(F.col(c).isNull(), c)).alias(c) for c in df_final.columns
])
display(null_counts_df)

# COMMAND ----------

# List all available checkpoints
list_checkpoints()
