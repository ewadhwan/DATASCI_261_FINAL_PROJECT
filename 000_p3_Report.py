# Databricks notebook source
# MAGIC %md
# MAGIC # Forecasting Flight Departure Delays: A Cross-validation in Time Series Machine Learning Approach
# MAGIC | Elana Wadhwani | Hildah Ngondoki | Omar Zu'bi | Roz Huang | Sohail Khan | Trevor Lang | 
# MAGIC |-------|------|-------|-------|------|-------|
# MAGIC | <img src="https://github.com/ewadhwan/w261_final_project_images/blob/main/headshot%20(2).jpeg?raw=true" width="200" height="200"> | <img src="https://avatars.githubusercontent.com/u/18394882?s=400&u=d00434789ef6027f30af67449e22a575f402a82b&v=4" width="200" height="200"> | <img src="https://github.com/DrZubi.png" width="200" height="200"> | <img src="https://github.com/ronghuang0604.png" width="200" height="200"> | <img src="https://media.licdn.com/dms/image/v2/D4E03AQEFRn2KuAHD4g/profile-displayphoto-shrink_400_400/profile-displayphoto-shrink_400_400/0/1727407069512?e=1758153600&v=beta&t=OG9nznHQpzDDq3eoZhUH8ppymppZMgCWK9x3x2xPf9M" width="200" height="200"> | <img src="https://www.ischool.berkeley.edu/sites/default/files/styles/fullscreen/public/profile_pictures/berkeley_id_photo_0.png?itok=ABOKiqjL" width="200" height="200"> |
# MAGIC | ewadhwan@usc.edu | hngondoki@berkeley.edu | ozubi@berkeley.edu | ronghuang0604@berkeley.edu | sohailk2@illinois.edu | tlang@berkeley.edu |
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Phase Leader Plan:
# MAGIC |We are at | Week # | Date | Leader | Due | Due Date | 
# MAGIC |----------|----------|----------|----------|----------|----------|
# MAGIC |          | Week 10 | 7/7-7/13 | Omar Zu'bi | P1 Report ✅  | 7/13 Sunday 11:59pm |
# MAGIC |          | Week 11 | 7/14 - 7/20 |  Elana Wadhwani | | |
# MAGIC |          | Week 12 | 7/21-7/27 | Roz Huang | Presentation ✅ | 7/22 Tuesday In-class |
# MAGIC |          |         |           | | P2 Report ✅ | 7/27 Sunday 11:59pm |               
# MAGIC |          | Week 13 - HW 5 | 7/28-8/3 |Hildah Ngondoki | HW 5✅ | 8/3 Sunday 11:59pm | 
# MAGIC |          | Week 13 | 7/28-8/3 | Trevor Lang | | | 
# MAGIC | ------>  | Week 14 | 8/4-8/10 | Sohail Khan | Presentation✅ | 8/5 Tuesday In-class |
# MAGIC |          |         |          | | P3 Report | 8/9 Saturday 11:59pm |

# COMMAND ----------

# MAGIC %md
# MAGIC ## Credit Assignment Plan:
# MAGIC | Name | Phase 1 | Phase 2 | Phase 3 |
# MAGIC |----------|----------|----------|----------|
# MAGIC | Roz Huang    | Data EDA, Data Imbalance Strategy, Data Cleaning   | Data preprocessing code (2h) Report write up (abstract & data preprocessing) (1h) Phase 2 report preparation, structure, and clean up (2h) Prepare phase 2 presentation (PPT front page & ML pipeline block diagram) (2h) Meeting planning, office hour questions, and cordinating between members (1.5h) |Ensemble model code & debug (5h) report write up(abstract, Machine Learning & Evaluation, appendix, and miscellaneous cleanup)(4h total) fix ML pipeline diagram format to make it fit in one picture and add to github(40min), create presentation slide diagram(40min)|
# MAGIC | Sohail Khan    | Machine learning algorithims |Conclusion Slide (\~1 hour)|Conclusion Slide (\~1 hour)|
# MAGIC | Trevor Lang    | 10 hours spent: Abstract, Correlation Heatmap, Feature Importance  | Machine Algorithims and Metrics, Machine Learning Pipeline (\~6 hours), Block diagram (ML) edits (\~1 hour), Results writeup (\~6 hours)| Machine Algorithims and Metrics (\~6 hours), Machine Learning Pipeline (\~1 hours), Results writeup (\~3 hours)|
# MAGIC | Hildah Ngondoki    | ~26 hours in EDA Plan, Data Cleaning and Missing Value Strategy, Defining Cross Validation on Time Series Approach| hours Work on presentation, Reformatting EDA to suit 1YR OTPW data (\~10)| Adjust to 3Y OTPW data(\~4hours), HW5 assignment, compilation, cleanup and submission (\~5), Project and HW5 Group meetings (\~2hrs),Tensor Flow NN (\~8hrs ) Hyperparameter tuning model(s)(\~6 hours) Documentation and presentation (\~3hrs)  |
# MAGIC | Elana Wadhwani    | ~ 8 hrs spent on Machine Algorithims and Metrics section, Machine Learning Pipeline section, Gnatt Chart, ML Pipeline Block Flow Diagram, and group meeetings    | Initial data cleaning, null value imputation, scaling, encoding, and building Spark MLlib pipelines (12h), report write-up (2h), block diagram edits (1h), research and planning for Introductory and Motivation slides (1h) | Data Balancing (1 hr) Training of 1 layer MLP NN (4 h), discussion of data leakage (2 h), experimenting with graph features (4 hrs) |
# MAGIC | Omar Zu'bi  |P1 leadership + coordination, Feature selection and overview research and writeup, conclusions writeup, documentation for the data ingestion section, and preliminary EDA visualizations for the flights_db table and checkpointing writeup (\~8 hours total including meetings)|Developed engineered time-based features (\~5 hours), temporal filtering & cleaning for feature engineering (\~4 hours), weather data integration (joins) (\~6 hours), block diagram (\~2 hour). section II, section IV, section VII writeup (\~3 hours), feature engineering slides and speaking notes prep for mid-project presentations (\~1 hour) | Feature engineering and joins with the 5Y dataset (\~4 hours), section II, section IV, section VII writeup (\~3 hours), feature engineering slides and speaking notes prep for final-project presentations (\~1 hour), Homework 5 (\~2 hours), group meetings (\~3 hours) |
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC # I) Abstract
# MAGIC _____
# MAGIC
# MAGIC Flight delays create substantial operational and financial challenges for airports and airlines, leading to disrupted schedules, higher costs, and diminished passenger satisfaction. To help airport operations teams act proactively, we aim to predict which flights will be delayed by 15 minutes or more, using data available two hours before scheduled departure. Our primary evaluation metric is recall, which measures the proportion of true delays correctly identified. In this business context, maximizing recall is critical, as failing to anticipate a delay carries a higher operational cost than reacting to a false alarm.
# MAGIC
# MAGIC
# MAGIC
# MAGIC Our methodology integrated three key datasets: on-time performance data from the U.S. Department of Transportation, historical weather observations from the National Oceanic and Atmospheric Administration (NOAA), and airport metadata, also from the Department of Transportation. We followed an iterative progression of increasing model complexity to maximize predictive power, beginning with a logistic regression baseline enhanced by a blocked time-series cross-validation framework. Recognizing the non-linear dynamics of flight operations, we then implemented a Multi-Layer Perceptron (MLP) to capture more intricate patterns. The final stage of our pipeline involved creating a hard-voting ensemble model, which combines the predictions of our enhanced logistic regression and MLP models to leverage their distinct strengths and produce a more accurate and reliable forecast.
# MAGIC
# MAGIC The results demonstrate a clear improvement in performance with each stage of modeling. While our initial baseline models provided a solid foundation, our final ensemble model yielded the most significant gains. On the holdout test set, it achieved a recall of **0.73** for the delayed class, successfully identifying a higher percentage of true delays than any individual model. This work validates that a multi-tiered modeling approach, culminating in an ensemble, is a highly effective strategy for forecasting flight delays and provides a valuable tool for proactive airport management.
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC # II) Project Description
# MAGIC _____
# MAGIC This report outlines a comprehensive workflow for predicting flight delays using an engineered dataset that incorporates weather and flight data from across the United States, covering five years of aviation activity. The structure moves from data exploration to model development, with each section building on the previous to deliver both technical depth and operational relevance.
# MAGIC
# MAGIC The analysis begins in Section III with Exploratory Data Analysis (EDA). Here, the OTPW dataset is unpacked to uncover key patterns and challenges. Summary statistics, missing value distributions, and correlation analyses provide a foundational understanding of the data. Notably, the EDA reveals a pronounced class imbalance—only 18% of flights are delayed—along with significant missingness in weather fields. These issues prompted the creation of a new dataset, constructed using custom joins, as detailed in the join strategy section. Temporal trends also emerge, such as increased delays during evening hours and holiday travel periods, underscoring the need for time-aware features.
# MAGIC
# MAGIC Section IV details the data engineering pipeline, built in PySpark to scale with the 31 million-record dataset. This section explains how raw aviation and weather data are transformed into 58 predictive features through time decomposition, aircraft history tracking, airport congestion metrics, and enriched weather joins. A checkpointing strategy is incorporated to improve runtime efficiency and resilience, particularly during complex multi-source joins and iterative processing.
# MAGIC
# MAGIC Section V transitions into the modeling phase, outlining the machine learning pipeline and evaluation framework. Preprocessing steps, feature family structures, and model selection rationale are discussed in detail. Emphasis is placed on recall as the primary metric, given the operational cost of missed delay predictions. The modeling path progresses from a logistic regression baseline to more expressive architectures, including multilayer perceptrons, with targeted strategies to mitigate the effects of class imbalance.
# MAGIC
# MAGIC In Section VI, results are presented and analyzed, drawing from both the training period (2015 to 2017 data) and a blind test set (2018 data). This section includes side-by-side model performance comparisons, feature importance rankings, and an assessment of how each architecture addresses the imbalance in delay classification. Our findings offer practical insight into which modeling approaches are best suited for real-world deployment.
# MAGIC
# MAGIC Section VII concludes the report, summarizing the key takeaways and reflecting on their practical significance for aviation stakeholders. The success of the feature engineering approach, the challenges of working with imbalanced data, and future opportunities for model enhancement and operational integration are discussed.
# MAGIC
# MAGIC Finally, an Appendix provides supplementary material, including extended technical details on custom joins and links to supporting code notebooks to ensure full reproducibility. Throughout, the report aims to deliver not only methodological rigor but also insights that are actionable within the context of airline delay management systems.
# MAGIC
# MAGIC
# MAGIC ### Machine Description:
# MAGIC All modeling and data processing tasks were conducted on Databricks Runtime 16.4 LTS, which includes Apache Spark 3.5.2 and Scala 2.12, with Photon acceleration enabled for enhanced performance. The pipeline was executed on r5d.xlarge worker nodes, each with 32 GB of memory and 4 vCPUs.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC # III) Exploratory Data Analysis (EDA)
# MAGIC _____
# MAGIC
# MAGIC ### Summary description
# MAGIC To build a predictive model for flight departure delays, we integrated multiple datasets that capture both operational and environmental conditions affecting flight performance. These datasets span aviation operations, weather observations, and geospatial metadata, and were sourced from reputable public agencies and repositories. Below is a summary of each dataset used in the project and its role in the modeling pipeline:
# MAGIC
# MAGIC - Airline On-Time Performance Data (Flights): The Airline On-Time Performance Data is provided by the Bureau of Transportation Statistics (BTS). It contains historical flight performance records for U.S. domestic flights. Each row represents an individual flight, including scheduled and actual departure/arrival times, delays and their causes (such as carrier, weather, or NAS), flight details (airline, origin, destination), and cancellation information. This dataset is important in analyzing flight delays and more importantly in understanding patterns across airlines, routes, and times of day.
# MAGIC
# MAGIC - Weather Data: The Weather Data is provided from NOAA. It consists of historical weather observations across the U.S.A, including temperature, humidity, precipitation, wind speed, visibility, and weather events such as storms and fog. The data is available on an hourly, daily or monthly basis per station, making it a key input for predicting delays influenced by weather conditions.
# MAGIC
# MAGIC - Weather Stations Data: It provides metadata about weather stations, including station IDs, latitude, longitude, elevation, and names, enabling a link between weather observations and specific geographic locations. It is useful in associating weather data with airports.
# MAGIC
# MAGIC - Airport Codes Data: Sourced from Airport Codes Dataset (DataHub), it contains details for airports worldwide, such as IATA and ICAO codes, airport names, cities, countries, and geographic coordinates. This dataset allows mapping airport codes in flight data to their geographic positions, supporting geospatial analysis and merging with weather data.
# MAGIC
# MAGIC ### Preliminary EDA Findings
# MAGIC
# MAGIC Before building predictive models, we conducted exploratory data analysis (EDA) to better understand the structure, quality, and distributions within the dataset. This step helped identify data issues, assess feature variability, and evaluate the balance of the target variable. Our analysis covered dataset shape, missing values, statistical distributions, and outcome class proportions. Additionally, a new dataset was formed by joining the raw datasets, thus creating a new dataset that will hereby be referred to as the _newly joined dataset_.The joining strategies will be discussed in the subsequent section under feature engineering.
# MAGIC
# MAGIC Key findings from the EDA of the 5 year (60month) the Newly Joined Dataset are summarized below:
# MAGIC
# MAGIC **Basic EDA**: The Shape of the dataset was 31,746,841 rows and 69 columns with a combination of airline and weather columns. The 69 columns needed further scrutiny for missing values analysis, duplicates and feature importance. 
# MAGIC
# MAGIC **Univariate Analysis**: In the Missing Values Analysis, a key finding in the dataset was that 32 columns contained missing values however, only 4 columns(HourlySeaLevelPressure, HourlyPrecipitation,HourlyWindGustSpeed and HourlyPresentWeatherType) had over 60% missing values, which suggested potential gaps in data sources or integration processes.  Key findings showed high variability in flight distances and weather conditions, along with outliers that likely reflect rare but valid events such as long-haul flights or severe weather.
# MAGIC
# MAGIC <figure style="display: flex; flex-direction: column; align-items: center;">
# MAGIC   <img src="https://raw.githubusercontent.com/hngondoki/forecasting-flight-departure-delays/main/images/5y_boxplot_numeric.png" width="560" alt="Box plot for numeric variables">
# MAGIC </figure>
# MAGIC
# MAGIC _Figure: Box plot for Numeric Variables_
# MAGIC
# MAGIC  **A Shapiro-Wilk test for normality** was applied to numeric features in the newly joined dataset. Most variables failed to meet normality assumptions (p-value < 0.05) due to the large sample size, which amplifies even minor deviations from normality. Despite the non-normality, the W-statistic exceeded 0.9 for several variables, suggesting that while not perfectly normal, some distributions are close enough.The summary statistics and scale showed that the dataset had high variance and large deviations especially for continuous variables like DISTANCE and some of the weather parameters. Additionally, outliers were highly prominent, as illustrated in boxplots. These outliers may not be errors but could represent rare events, such as unusually long-haul flights or extreme weather delays. Since the data did not follow a normal distribution as would be expected, so standard scaling methods during modelling may not apply. 
# MAGIC
# MAGIC **Outcome Variable Analysis**:
# MAGIC In the dataset, the target variable (`DepDel15` - binary: delayed vs. on-time) revealed class imbalance, with only 18% of flights classified as delayed (label 1), compared to 80.5% being on time (label 0) and a lesser 1.5% being missing values. Such imbalance poses challenges for predictive modeling, as most classifiers will favor the majority. Label 0 (On time/ Early) (80.5%) : 25,576,004; 1 (Delayed) (18%): 5,693,541 and Null (Missing Values) (1.5%) : 477,296 with a **Total** of **31,746,841**
# MAGIC The histogram showcased that a majority of the flights arrive early and on time (64%), 17% are just in time to avoid being flagged as delayed and Delayed Flights are 18%.
# MAGIC  
# MAGIC <figure style="display: flex; flex-direction: column; align-items: center;">
# MAGIC   <img src="https://raw.githubusercontent.com/hngondoki/forecasting-flight-departure-delays/main/images/5y_DepDel_histogram.png" alt="Delay Histogram" width="600">
# MAGIC </figure>
# MAGIC
# MAGIC _Figure: Histogram Plot of Delays_
# MAGIC
# MAGIC **Bivariate Analysis**: The Bivariate Analysis showed high correlation parameters (e.g., CRS_DEP_TIME, DEP_TIME, and DEP_DELAY), which may result in multicollinearity if used together. Strong correlations were also observed between weather variables (like HourlyVisibility, HourlyWindSpeed) and delay-related metrics, highlighting their predictive potential providing useful  information for feature engineering and model selection.
# MAGIC
# MAGIC - Correlation analysis
# MAGIC
# MAGIC To understand the interrelationships between key features in the Newly Joined Dataset, a select subset of continuous and nominal variables was chosen based on their relevance to the target variable `DepDel15` and domain knowledge of aviation operations. These included numerical features such as DISTANCE, HourlyWindSpeed, HourlyPrecipitation, and categorical-but-numeric-coded variables like `DayOfWeek`,`DayofMonth` and 'CRS_ARR_TIME'.There were relatively weak or neutral correlations between certain temporal variables and our target variable `DepDel15`, suggesting that time-of-day effects may be better captured using engineered cyclical features. Notably, most of the time parameters such as `ARR_TIME`, `WHEELS_OFF`, and `WHEELS_ON` were highly correlated, reflecting the sequential and interdependent nature of flight operations where delays or shifts in one timestamp often propagate to others.
# MAGIC
# MAGIC <figure style="display: flex; flex-direction: column; align-items: center;">
# MAGIC   <img 
# MAGIC     src="https://raw.githubusercontent.com/hngondoki/forecasting-flight-departure-delays/main/images/5Y_correlation.png" 
# MAGIC     alt="Pearson Correlation Matrix" 
# MAGIC     style="width: 600px; height: auto; border-radius: 8px; box-shadow: 0 0 8px rgba(0,0,0,0.15);"
# MAGIC   >
# MAGIC </figure>
# MAGIC
# MAGIC
# MAGIC _Figure: Correlation diagram of the numeric features_
# MAGIC
# MAGIC **Temporal Analysis and Trends**:There was a temporal and trend analysis to highlight any variability due to seasonality. Delay reflected normal seasonality across the months with high counts reflected in .However, finer-grained hourly patterns revealed interesting insights highlighting the temporal relationships inherent in the data set. A closer look at the delayed flights count by month showed that there was a visible decrease in delays between the months of September to November
# MAGIC
# MAGIC <div style="text-align: center">
# MAGIC   <h6 style="font-family: Arial, sans-serif; font-weight: bold; margin-bottom: 10px;">
# MAGIC   </h6>
# MAGIC   <img src="https://github.com/hngondoki/forecasting-flight-departure-delays/blob/main/images/5y_by_month_depdel.png?raw=true="Delays by Month" width="600"/>
# MAGIC   <div style="font-size: 90%; color: gray; margin-top: 5px;">
# MAGIC   </div>
# MAGIC </div>
# MAGIC
# MAGIC _Figure: A bar plot of flights by month shows low counts in delayed flights between September and November_
# MAGIC
# MAGIC Additional analysis highlighed sharp increase in flight counts in 2018 and 2019. A further hourly analysis in 6 hour blocks also indicated that you are twice more likely to get a delay at night (6pm-midnight) despite lower number of flights.Additionally we observe a sharp increase in total number of flights flights from 1700hours to 1800hrs.
# MAGIC
# MAGIC <div style="text-align: center">
# MAGIC   <h6 style="font-family: Arial, sans-serif; font-weight: bold; margin-bottom: 10px;">
# MAGIC   </h6>
# MAGIC   <img 
# MAGIC     src="https://github.com/hngondoki/forecasting-flight-departure-delays/blob/main/images/5y_trends_analysis.png?raw=true" 
# MAGIC     alt="Flights by Year and Time of Day" 
# MAGIC     width="1000"
# MAGIC   />
# MAGIC </div>
# MAGIC
# MAGIC _Figure: Number of flights shown by year and by time of day in 6 hour blocks_
# MAGIC
# MAGIC <div style="text-align: center; margin-top: 30px;">
# MAGIC   <h6 style="font-family: Arial, sans-serif; font-weight: bold; margin-bottom: 10px;">
# MAGIC     Delay by Hour
# MAGIC   </h6>
# MAGIC   <img 
# MAGIC     src="https://github.com/hngondoki/forecasting-flight-departure-delays/blob/main/images/5y_by_hour2.png?raw=true" 
# MAGIC     alt="Delay by Hour" 
# MAGIC     width="600"
# MAGIC   />
# MAGIC </div>
# MAGIC
# MAGIC _Figure: Hourly delay patterns illustrating increase in delays between 0500hours and 1700hours._
# MAGIC
# MAGIC **Graph Features**:
# MAGIC Two graph features were considered for use in the final model: betweenness centrality and degree centrality. 
# MAGIC
# MAGIC Betweenness centrality was used to identify "key connectors" or "transit hubs" that play a crucial role in the overall flow of traffic. DEN (Denver) emerged as the most critical connector in the network; many indirect routes go through here.If operations at these hubs are disrupted (weather, strikes, etc.), a large portion of the net work would be affected.
# MAGIC
# MAGIC <div style="text-align: center; margin-top: 30px;">
# MAGIC   <h6 style="font-family: Arial, sans-serif; font-weight: bold; margin-bottom: 10px;">
# MAGIC     Betweenness Centrality to identify transit hubs
# MAGIC   </h6>
# MAGIC   <img 
# MAGIC     src="https://github.com/hngondoki/forecasting-flight-departure-delays/blob/main/images/5y_graph_analysis.png?raw=true" 
# MAGIC     alt=" Between Centrality to identify transit hubs" 
# MAGIC     width="600"
# MAGIC   />
# MAGIC </div>
# MAGIC
# MAGIC _Figure showing DEN (Denver) is the most critical connetor in the network followed by ORD (Chicago O'Hare) and DFW (Dallas-Fort Worth)_
# MAGIC
# MAGIC Degree centrality was calculated as the number of incoming flights to a given airport. Degree centrality gives an indication of how busy an airport is. Busier airports have less flex in their runway availability and a more congested airspace, increasing the likelihood of a delay.
# MAGIC
# MAGIC <div style="text-align: center; margin-top: 30px;">
# MAGIC   <h6 style="font-family: Arial, sans-serif; font-weight: bold; margin-bottom: 10px;">
# MAGIC     Degree Centrality to identify transit hubs
# MAGIC   </h6>
# MAGIC   <img 
# MAGIC     src="https://github.com/ewadhwan/w261_final_project_images/blob/main/degree%20centrality%20of%20US%20airports.png?raw=true" 
# MAGIC     alt="Degree Centrality to identify busy airports" 
# MAGIC     width="600"
# MAGIC   />
# MAGIC
# MAGIC </div>
# MAGIC
# MAGIC _Figure showing ORD (Chicago O'Hare) is the busiest airport followed by ATL (Atlanta) and DFW (Dallas-Fort Worth)_
# MAGIC
# MAGIC ### Strategies for addressing data gaps
# MAGIC
# MAGIC 1. Data Balancing Strategy: The target variable, `DepDel15` in the newly joined dataset, is highly imbalanced, with approximately 80% of flights on time and only 20% delayed. This imbalance can reduce the model's  ability to detect meaningful delay patterns. To mitigate this, undersampling will be applied during training to balance both delayed and ontime flights. To avoids data duplication or sampling that could compromise temporal integrity delayed flights will be maintained whilst the ontime flights will be randomly sampled within a given time fold for the time-series cross validation.
# MAGIC
# MAGIC 2. Missing Value Strategy: Given high variance in numerical variables, three tailored imputation strategies are used to maintain statistical fidelity. For continuous variables like `CRSElapsedTime` or `TAXI_OUT`, median imputation is chosen to reduce the influence of outliers and skewness. For categorical variables like `Origin` or `Dest`, a new unknown class ("UNK") was used to fill null values. The creation of the "UNK" class provides a way of filling the null categorical values without smoothing over any underlying patterns that may cause a higher percentage of null values for a given class. Boolean variables, such as IS_WEEKEND were imputed with the majority class. The target variable, DEP_DEL15, has 130,110 missing values (1.8% of the dataset),which will be excluded to prevent injecting noise.
# MAGIC <div style="text-align: center; max-width: 400px; margin: auto;">
# MAGIC   <h6 style="font-family: Arial, sans-serif; font-weight: bold; margin-bottom: 10px;">
# MAGIC     Missing Values in Newly Joined Dataset
# MAGIC   </h6>
# MAGIC <img src="https://github.com/hngondoki/forecasting-flight-departure-delays/blob/main/images/5y_missing_value.png?raw=true"
# MAGIC style="width: 600; height: auto; border-radius: 8px; box-shadow: 0 0 8px rgba(0,0,0,0.15);"
# MAGIC   >
# MAGIC </div>
# MAGIC
# MAGIC _Figure: Missing values counts by column_
# MAGIC
# MAGIC 3. Outlier Handling Strategy: Box plot analysis revealed outliers across several variables. However, in the aviation domain, such values may reflect legitimate operational conditions—e.g., international flights or extreme weather events. For instance, DISTANCE ranges from 31.0 to 4983.0 miles, representing valid domestic and long-haul flights. Instead of removing these outliers, transformations will be used to reduce skewness without discarding informative signals. Specifically, logarithmic transformation will be used to compress scale for highly skewed variables, min-max normalization to bound continuous variables within 0-1, and robust scaling for resilience to outliers. Categorical and datetime variables will remain unscaled.
# MAGIC
# MAGIC Outlier values are shown in the graph in Appendix D
# MAGIC
# MAGIC 4. Feature Engineering and Modeling Readiness:The EDA in the Newly Joined Dataset revealed just about about 32 columns with missing values and less than 4 with high number(over 60%) of missing values. Univariate and bivariate analysis highlighted class imbalance and correlations relevant to model performance. Most numeric columns failed normality tests (e.g., Shapiro-Wilk), validating the choice of median-based imputations and transformation strategies.
# MAGIC
# MAGIC These EDA findings will inform a modeling pipeline that ensures robustness to imbalance, variance, and domain-specific variability. The approach emphasizes data integrity, statistical rigor, and scalability to support predictive accuracy in a real-world forecasting scenario.
# MAGIC
# MAGIC
# MAGIC ## Data dictionary of the raw features 
# MAGIC An in-depth analysis of the raw dataset was conducted to understand the domain-specific context and accurately categorize each variable by its functional type—namely continuous (numerical), categorical (nominal or ordinal), or text. This was done to inform downstream preprocessing decisions, such as encoding strategies, scaling methods, and feature selection. The data types of columns were systematically reviewed to identify any inconsistencies requiring type casting. A notable observation was that a significant portion of the weather-related features such as `HourlyPrecipitation` and `HourlyRelativeHumidity`, were represented as strings rather than float, thus will require to be type cast for use in the model. Additionally, certain variables—such as the target column `DepDel15` were originally stored as `double` despite representing binary outcomes, prompting conversion to boolean/integer format to ensure correct handling in classification. This early profiling ensured that each feature was treated appropriately based on both its semantic meaning and technical structure, laying a solid foundation for robust preprocessing and model training. The table with each column representation is as provided in Appendix B.
# MAGIC
# MAGIC
# MAGIC ## Dataset size (rows columns, train, test, validation)
# MAGIC The Newly Joined Dataset size is a total of 7,422,037 . When classified using the target variable `DEP_DEL15` the values are as follows:`DepDel15`with Label 0 (On time/ Early) (80.5%) : 25,576,004; 1 (Delayed) (18%): 5,693,541 and Null (Missing Values) (1.5%) : 477,296 with a **Total** of **31,746,841**
# MAGIC |DepDel15 labels|count|
# MAGIC |-----|--------|
# MAGIC |0|	25,576,004|
# MAGIC |1|	5,693,541|
# MAGIC |null|477,296|
# MAGIC
# MAGIC
# MAGIC The training data will comprise of the 2015 to 2017 data, which is representative of 6,042,754  records, validation data if necessary will comprise of 2018 data which culminates to 2,608,853. In some cases e.g Logistic regression, the validation datasets might be joined to form the training dataset. The test data will cover 2019 which 7,287,112 records. However, these will be further subjected to class imbalance handling strategies (undersampling) during data pre-processing.
# MAGIC
# MAGIC ## Summary statistics
# MAGIC
# MAGIC There are fewer number of missing records in the Newly Joined Dataset compared to the 1Y OTPW dataset. There are large ranges for the standard deviations for the non-categorical variables. Thus, using the missing values strategy will be applied as 1) Median for columns with numerical values 2)'UNK' for catgorical values and 3) removal of values within the target variable `DepDel15`
# MAGIC
# MAGIC Some interesting highlights from the Summary showed that scheduled departure time for flights have a median of 1325hrs while for schedule arrival it was at 1508hrs. While the fields are not continuous, it does highlight possible congestion during those hours when the airport is busy.
# MAGIC
# MAGIC The representation of the Summary Statitics  is provided in Appendix C
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC # IV) Join Strategy, Checkpointing, & Feature Engineering
# MAGIC ______
# MAGIC This section outlines the feature engineering pipeline built in PySpark on Databricks for predicting flight delays. The pipeline processes over 31 million U.S. domestic flights from 2015 to 2019, transforming raw aviation data into a rich, analysis-ready dataset containing 58 features. The work leverages enterprise-grade data engineering practices, including robust checkpointing, window functions, and optimized joins, to prepare a foundation for downstream machine learning. 
# MAGIC
# MAGIC The target variable is `DepDel15`, a binary label indicating whether a flight departed more than 15 minutes late. In aviation, a delay of 15 minutes or more is widely accepted as the standard threshold to classify a flight as delayed; this threshold is used dby the U.S. Department of Transportation (DOT) and the Bureau of Transportation Statistics (BTS). Features span temporal dynamics, airport congestion, weather, aircraft history, and operational trends which are all critical for modeling delay behavior in a realistic operational context.
# MAGIC
# MAGIC ## Data Scope
# MAGIC The dataset spans the calendar years 2015 to 2019, offering robust coverage of seasonal trends, holiday effects, and annual patterns. It includes only domestic flights to ensure consistency in regulations, operations, and data availability. With over 31 million records, the dataset provides both breadth and depth, making it well-suited for high-resolution modeling.
# MAGIC
# MAGIC ## Temporal Feature Engineering
# MAGIC Aviation operations are heavily influenced by time. The pipeline begins with standard calendar decomposition from `FlightDate`, and builds higher-order temporal features to capture operational behavior:
# MAGIC
# MAGIC - `Year`, `Quarter`, `Month`, `DayOfMonth`, `DayOfWeek`: basic calendar fields
# MAGIC - `IsWeekend`: flags Saturday and Sunday flights
# MAGIC - `IsBusinessHours`: marks flights scheduled between 6 AM and 10 PM — peak airport operating hours
# MAGIC - `SeasonQuarter`: maps each flight to a seasonal bucket based on month (e.g. winter, summer)
# MAGIC - `IsHolidayMonth`: flags flights during high-demand months (Nov, Dec, Jan)
# MAGIC
# MAGIC These features help model seasonal delays, business vs. leisure trends, and congestion during high-traffic hours.
# MAGIC
# MAGIC ## Aircraft History Features
# MAGIC Aircraft behavior is sequential, and past flight performance is a strong signal for predicting future delays. Using window functions over `Tail_Number`, the pipeline computes lag features that reflect prior activity:
# MAGIC
# MAGIC - `Prev_ArrDelay`: arrival delay from the aircraft's previous flight
# MAGIC - `Prev_TaxiIn` and `Prev_TaxiOut`: ground movement durations on the previous flight
# MAGIC - `TurnaroundTime`: time between last arrival and next scheduled departure (handles cross-midnight cases)
# MAGIC - `Is_First_Flight`: flag indicating whether this is the first flight of the aircraft in the dataset
# MAGIC
# MAGIC These features help assess readiness, accumulated delay, and whether the aircraft is likely under stress from tight schedules. These features are calculated using data from up to two hours before the scheduled flight departure times.
# MAGIC
# MAGIC ## Airport Congestion and Traffic Metrics
# MAGIC To understand real-time airport conditions, the pipeline calculates several features based on a 2-hour window before scheduled departure. These include:
# MAGIC
# MAGIC - `Num_airport_wide_delays`: counts delayed departures at the origin airport
# MAGIC - `Num_airport_wide_cancellations`: tracks cancellations during that window
# MAGIC - `Oncoming_flights`: counts flights planned to arrive at the origin airport (degree centrality), which may tie up gates or ground crew
# MAGIC
# MAGIC These indicators help quantify system stress and operational density and offers key predictors of departure delays.
# MAGIC
# MAGIC ## Historical performance Trends
# MAGIC Aircraft reliability is captured across three rolling windows. All of these features are calculated using data from up to two hours before the scheduled flight departure times:
# MAGIC
# MAGIC - Historical: full record of previous flights (+2 hours prior)
# MAGIC - Short-term: last 90 days 
# MAGIC - Recent: last 14 days 
# MAGIC
# MAGIC For each window, the pipeline computes on-time percentages for both arrivals and departures. These features help distinguish consistently reliable aircraft from those that have recently developed issues. A drop in short-term performance may reflect emerging maintenance problems or changing flight assignments.
# MAGIC
# MAGIC ## Weather Integration
# MAGIC Weather is one of the most complex yet crucial factors. The pipeline merges hourly weather data with flights by aligning both spatial location and temporal information. Notably, all these features are computed using data from up to two hours before the scheduled flight departure times—a process made possible through the use of helper columns, which significantly increased processing time.
# MAGIC
# MAGIC Airports are linked to their closest weather stations, and conditions are matched based on weather reported two hours before scheduled departure. This timing reflects how dispatch decisions are typically made. Weather readings are deduplicated and averaged when needed, and IATA/ICAO code mappings ensure consistent identifiers.
# MAGIC
# MAGIC The final weather features include wind speed and direction, visibility, temperature, humidity, precipitation, and atmospheric pressure. These are conditions provide a detailed snapshot of conditions that may impact flight operations.
# MAGIC
# MAGIC ## Join Strategy and Performance
# MAGIC Each major stage of the pipeline is checkpointed to distributed storage. This makes it easy to rerun or isolate portions of the pipeline during development, without recomputing everything from scratch. Joins are optimized using Spark’s broadcast strategy where applicable, and memory is managed by caching reusable intermediates and cleaning up temporary data.
# MAGIC
# MAGIC Temporal window functions are partitioned logically to avoid unnecessary shuffles, and row-count checks are used to validate join quality throughout the process.
# MAGIC
# MAGIC The end-to-end runtime from data loading to full feature computation—without utilizing any previously saved checkpoints—is a little over 1 hour.
# MAGIC
# MAGIC ## Data Quality & Null Handling
# MAGIC Data quality is carefully handled. For example, approximately 1.5% of `DepDel15` values are null. Lag-based aircraft features are expected to have nulls for the first flight of each aircraft and were preserved.
# MAGIC
# MAGIC Weather data inconsistencies are reduced using aggregation and distance-based station matching. Additional validation includes checking for row count mismatches and ensuring the integrity of key joins.
# MAGIC
# MAGIC ## Feature Design Pipeline & Checkpointing Stragety
# MAGIC This pipeline prioritizes operational realism by focusing on features that are both meaningful and computationally efficient. Aircraft reliability features take precedence due to their strong predictive power. Weather features are comprehensive, while congestion metrics effectively capture real-world airport dynamics.
# MAGIC
# MAGIC Temporal patterns contextualize operations across days, seasons, and peak periods. Together, these features capture short-term operational risks, long-term trends, and external stressors that influence flight delays.
# MAGIC
# MAGIC **Key highlights include:**  
# MAGIC - Consistent use of PySpark with structured transformations and windowing  
# MAGIC - Checkpointing to support reproducibility and iterative development  
# MAGIC - Optimizations for join order, partitioning, and caching  
# MAGIC - A modular, scalable design ready for production environments
# MAGIC
# MAGIC The engineered dataset strikes a balance between coverage, efficiency, and model readiness. With 58 well-structured features encompassing all major delay factors, this pipeline sets the stage for downstream machine learning models to accurately predict departure delays.
# MAGIC
# MAGIC To improve efficiency and reproducibility in our data processing workflow, our team implemented **checkpointing** to save and reload intermediate DataFrame states. Here’s how checkpointing was used:
# MAGIC
# MAGIC - **Checkpoint storage location:**  
# MAGIC   Checkpoints are saved at: `dbfs:/student-groups/Group_02_01/checkpoints`
# MAGIC
# MAGIC - **Saving checkpoints:**  
# MAGIC   The `save_checkpoint(df, checkpoint_name, description)` function saves a DataFrame:  
# MAGIC   - Data is stored in Parquet format  
# MAGIC   - Metadata such as row and column counts are printed for verification  
# MAGIC   - Optional descriptions document the purpose of each checkpoint
# MAGIC
# MAGIC - **Loading checkpoints:**  
# MAGIC   The `load_checkpoint(checkpoint_name)` function reloads saved DataFrames and prints details for confirmation
# MAGIC
# MAGIC - **Managing checkpoints:**  
# MAGIC   - Existence is verified with `checkpoint_exists(checkpoint_name)`  
# MAGIC   - Available checkpoints are listed via `list_checkpoints()`  
# MAGIC   - Unneeded checkpoints can be removed using `delete_checkpoint(checkpoint_name)`
# MAGIC
# MAGIC Checkpointing prevents costly recomputations, enables easy resumption from saved states, and facilitates progress tracking with descriptive names and notes. Persisting intermediate data between notebook runs ensures a smooth, efficient, and iterative development process.
# MAGIC
# MAGIC Next steps involve model development, training, and evaluation using this feature set—with a focus on identifying the most predictive variables and refining the pipeline as needed.
# MAGIC
# MAGIC ## Final Dataframe Overview
# MAGIC
# MAGIC Several features were dropped due to high collinearity (i.e., relative humidity and wet bulb temperature), repetitiveness (i.e. distance and distance group), or because they contained information only available after the two hour window prior to a flight's departure (i.e. the flight's arrival time). A subset of the features below were fed to the logistic regression model for training. They can be found in the "Data Post-Processing" Section.
# MAGIC
# MAGIC **Target Variable**
# MAGIC - **`DepDel15`** *(double)* – Indicates whether the flight was delayed by **15+ minutes** (`1 = Yes`, `0 = No`)  
# MAGIC   - This is the **primary target variable** for classification.  
# MAGIC   - It is *not* available at prediction time and should only be used during model training.
# MAGIC
# MAGIC **Temporal Features**
# MAGIC - **`FlightDate`** *(date)* – The actual flight date  
# MAGIC   - Enables **time-based joins**, aggregations, or trend analysis.  
# MAGIC   - Important for tracking events and ensuring **causal feature engineering**.
# MAGIC - **`Year`, `Quarter`, `Month`, `DayofMonth`, `DayOfWeek`** *(integer)* – Components of the flight date  
# MAGIC   - Helps capture **seasonal**, **weekly**, or **monthly** delay patterns.
# MAGIC - **`CRSDepTime`** *(integer)* – Scheduled departure time (e.g., `1415` = 2:15 PM)  
# MAGIC   - Can be used to derive **time of day** or group into periods.
# MAGIC - **`DepTimeBlk`** *(string)* – Scheduled departure block (e.g., `"1400-1459"`)  
# MAGIC   - Provides a **pre-bucketed** version of `CRSDepTime`.
# MAGIC - **`Hour`** *(integer)* – Extracted hour from scheduled departure  
# MAGIC   - Captures **hour-level time effects** (e.g., rush hours).
# MAGIC - **`IsWeekend`** *(integer)* – Flag for weekend flights (`1 = Yes`)  
# MAGIC   - Helps model **different demand patterns**.
# MAGIC - **`IsBusinessHours`** *(integer)* – Flag for business hours (`1 = Yes`)  
# MAGIC   - Business-hour flights may have different delay risk.
# MAGIC - **`SeasonQuarter`** *(string)* – Seasonal quarter label (e.g., `Q1:Winter`)  
# MAGIC   - Captures **weather and demand effects** related to the season.
# MAGIC - **`IsHolidayMonth`** *(integer)* – Indicates if flight is in a **holiday month** (`1 = Yes`)  
# MAGIC   - Helps model **holiday-related surges** in delays.
# MAGIC
# MAGIC **Flight-Specific Features**
# MAGIC - **`Reporting_Airline`** *(string)* – Carrier code  
# MAGIC   - Different airlines have different **delay behaviors**.
# MAGIC - **`Flight_Number_Reporting_Airline`** *(integer)* – Unique flight number  
# MAGIC   - Some flights are **consistently delayed**.
# MAGIC - **`Tail_Number`** *(string)* – Aircraft identifier  
# MAGIC   - Useful for capturing **aircraft-level reliability or issues**.
# MAGIC
# MAGIC **Airport and Route Features**
# MAGIC - **`Origin`, `OriginCityName`, `OriginState`**  
# MAGIC - **`Dest`, `DestCityName`, `DestState`**  
# MAGIC   - These fields provide geographic context to understand **route performance**.
# MAGIC - **`OriginAirportID`, `DestAirportID`** *(integer)* – Unique airport IDs  
# MAGIC - **`OriginWac`, `DestWac`** *(integer)* – FAA regional codes  
# MAGIC   - Useful for **regional trends and grouping**.
# MAGIC - **`Distance`, `DistanceGroup`** *(double, integer)* – Flight distance and its categorical group  
# MAGIC   - Longer flights may behave **differently in terms of delays**.
# MAGIC - **`CRSElapsedTime`** *(double)* – Scheduled duration of the flight  
# MAGIC   - Related to `Distance`, but also reflects **airspace complexity**.
# MAGIC
# MAGIC **Weather-Specific Features**
# MAGIC - **`HourlyWindSpeed`, `HourlyWindGustSpeed`** *(double)* – Wind intensity  
# MAGIC - **`HourlyVisibility`** *(double)* – Visibility in miles  
# MAGIC - **`HourlySkyConditions`** *(string)* – Cloud type and altitude  
# MAGIC - **`HourlyPrecipitation`** *(double)* – Rain/snow in inches  
# MAGIC - **`HourlyPresentWeatherType`** *(string)* – Weather condition (e.g., fog, rain)  
# MAGIC - **`HourlyDryBulbTemperature`, `HourlyWetBulbTemperature`, `HourlyDewPointTemperature`** *(double)* – Temperature measurements  
# MAGIC - **`HourlyRelativeHumidity`** *(double)* – Humidity percentage  
# MAGIC - **`HourlySeaLevelPressure`** *(double)* – Atmospheric pressure  
# MAGIC
# MAGIC **Lag-Based and Operational Features**
# MAGIC - **`Prev_TaxiIn`, `Prev_TaxiOut`** *(double)* – Taxi-in/out times from aircraft’s previous flight  
# MAGIC - **`Prev_ArrDelay`** *(double)* – Previous flight’s arrival delay  
# MAGIC - **`Prev_ArrTime`, `Prev_FlightDate`** – Previous leg’s arrival time/date  
# MAGIC   - Used in computing turnaround time.
# MAGIC - **`Turnaround_Time`** *(integer)* – Time between aircraft’s previous arrival and next scheduled departure  
# MAGIC   - Shorter times imply **higher risk of delay**.
# MAGIC   - `Turnaround_Time = CRSDepTimeₜ − ArrTimeₜ₋₁`
# MAGIC - **`Num_airport_wide_delays`, `Num_airport_wide_cancelations`** *(long)* – Count of recent delays/cancellations at departure airport  
# MAGIC   - Indicates **system-wide disruptions** at the origin.
# MAGIC - **`Oncoming_flights`** *(long)* – Number of inbound flights arriving shortly before scheduled departure  
# MAGIC   - Captures **gate congestion or airport busyness**.
# MAGIC   - Chosen over betweenness centrality because it is less computationally expensive and ranked airports similarly to betweenness centrality.
# MAGIC
# MAGIC **Historical & Recent Performance Features**
# MAGIC - **`historical_ontime_arrival_pct`**, **`historical_ontime_departure_pct`** *(double)*  
# MAGIC   - Long-term performance metrics for arrival and departure (all-time).
# MAGIC     - `historical_ontime_arrival_pct = (Total On-Time Arrivals / Total Arrivals) × 100`  
# MAGIC     - `historical_ontime_departure_pct = (Total On-Time Departures / Total Departures) × 100`
# MAGIC - **`recent_ontime_arrival_pct`**, **`recent_ontime_departure_pct`** *(double)*  
# MAGIC   - Rolling average over the **last 30 days**.
# MAGIC     - `recent_ontime_arrival_pct = (Σ Daily On-Time Arrival % over 30 days) / 30`  
# MAGIC     - `recent_ontime_departure_pct = (Σ Daily On-Time Departure % over 30 days) / 30`
# MAGIC - **`short_term_ontime_arrival_pct`**, **`short_term_ontime_departure_pct`** *(double)*  
# MAGIC   - Based on **last 7 days** of flight records.
# MAGIC     - `short_term_ontime_arrival_pct = (Σ Daily On-Time Arrival % over 7 days) / 7`  
# MAGIC     - `short_term_ontime_departure_pct = (Σ Daily On-Time Departure % over 7 days) / 7`
# MAGIC - **`OntimeArrivalPct`, `OntimeDeparturePct`** *(double)*  
# MAGIC   - Aggregated metrics capturing **timeliness** for this specific airport/route combination (all_time).
# MAGIC     - `OntimeArrivalPct = (On-Time Arrivals for Route / Total Arrivals for Route) × 100`  
# MAGIC     - `OntimeDeparturePct = (On-Time Departures for Route / Total Departures for Route) × 100`
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC # V) Modeling Pipelines
# MAGIC ____
# MAGIC This section outlines our end-to-end modeling pipeline, which includes post-processed data cleaning, feature refinement, and a multi-stage machine learning framework. Our approach begins with rigorous data preparation to eliminate noise, prevent data leakage, and balance the classification task. We then implement a progressive modeling strategy, starting with interpretable baseline models and advancing to more complex architectures such as neural networks and ensemble methods. Each stage is carefully designed to enhance predictive performance while maintaining transparency and reproducibility throughout the workflow.
# MAGIC
# MAGIC ## ML Pipeline Block Diagram
# MAGIC <img src="https://raw.githubusercontent.com/DrZubi/MIDS_261/main/p3_images/p3_ML_pipeline.png" width="100%"/>
# MAGIC
# MAGIC ## Data Post-processing
# MAGIC Our post data processing workflow consisted of several key steps designed to improve data quality and relevance for flight delay prediction:
# MAGIC
# MAGIC 1. **Eliminate columns with insufficient or unreliable data.** Several columns have no values (ie. some of the weather data) or are primarily null. Columns with 50% or more null values were dropped because they do not contain enough data to meaningfully impute null values and are not consistently informative.
# MAGIC 2. **Filter out cancelled flights.** Because cancelled flights are neither "on-time" or "delayed", they are irrelevant to the training data.
# MAGIC 3. **Drop any duplicate rows.** A duplicate row will have the same flight number and scheduled departure time.
# MAGIC 4. **Drop any rows missing key identifying information.** Drop any rows without a flight number, scheduled departure time, or DEP_DELAY15 status because there is not enough information to properly identify the flight.
# MAGIC 5. **Drop unused features.** Several features had high collinearity (ie relative humidity and wet bulb temperature), were repetitive (ie distance and distance group), or contained information that occured after the 2 hours before departure cutoff (ie OnTimeArrivalPct). These features were dropped from the data frame. A GBT Classifier model was also trained to help restrict the featureset to the most impactful. The final features fed to the logistic regression model were as follows:
# MAGIC - `Prev_TaxiIn`
# MAGIC - `Prev_ArrDelay`
# MAGIC - `Prev_ArrTime` 
# MAGIC - `Turnaround_Time`
# MAGIC - `short_term_ontime_arrival_pct` 
# MAGIC - `short_term_ontime_departure_pct`
# MAGIC - `Hour`
# MAGIC - `Oncoming_flights`
# MAGIC - `Month`  
# MAGIC
# MAGIC 6. **Split into train and test sets.** Data from the years 2015-2018 were used to form the training set while the 2019 data was held back to test the effectiveness of the model. The split occurs at this point in the pipeline so the information required for scaling and imputation is calculated only on the training set. This will prevent data leakage from the test to the training data.
# MAGIC 7. **Impute null variables.** Use the median for numerical variables and the "UNK" class for categorical variables. Median imputation was chosen for the numercial variables due to the large amounts of skew and outliers in the numerical veatures.  The "UNK" class was used to preserve any patterns that may result in more null values for a given class. 
# MAGIC 6. **Scale the data.** Use min-max scaling to scale numerical features. Min-max scaling was selected over standard scaling because of the non-normal distribution of the features. Categorical variables were not scaled.
# MAGIC 7. **Undersample the majority class in the training data.** As mentioned above, the "delayed" class only makes up ~20% of the dataset. To ensure the model would learn to detect true delays, the number of on-time flights was undersampled. Because the model training was done using blocked time-series cross validation, undersampling needed to occur within each year to ensure each training time block was balanced. This was done by splitting the training data into the same time blocks used for training. Within each block, a random sample of the on-time flights was retained to achieve an equal number of delayed and on-time flights.
# MAGIC
# MAGIC These steps reduced noise, minimized the risk of data leakage, and helped ensure that our subsequent feature engineering and modeling efforts were based on clean, high-quality data.
# MAGIC
# MAGIC ## Machine Learning & Evaluation
# MAGIC Our machine learning pipeline was designed as an iterative process, starting with simple, interpretable models and progressively building toward more complex and powerful solutions. This multi-stage approach allows for clear benchmarking and demonstrates the value added at each level of complexity. To allow for clear comparison between models, the features used in each experiment remained the same. Each experiment also sought to minimize binary cross entropy loss as the loss function: $$\mathcal{L}_{\text{BCE}} = - \frac{1}{N} \sum_{i=1}^N \left[ y_i \log(\hat{p}_i) + (1 - y_i) \log(1 - \hat{p}_i) \right]$$
# MAGIC
# MAGIC
# MAGIC
# MAGIC 1. Simple Baseline Model: We began with a standard logistic regression model to establish an initial performance benchmark. This provided a quick, interpretable reference point based on our feature-engineered dataset.
# MAGIC
# MAGIC 2. Enhanced Baseline Model: To account for the temporal nature of our data, we introduced a more robust pipeline featuring blocked time series cross-validation and a hyperparameter grid search. This ensured our logistic regression model was optimized for generalization over time.
# MAGIC
# MAGIC 3. Multi-Layer Perceptron (MLP): Recognizing that flight delays are driven by complex, non-linear interactions, we implemented a Multi-Layer Perceptron (MLP) neural network. This advanced model was designed to capture intricate patterns in the data that simpler linear models like logistic regression might miss.
# MAGIC
# MAGIC 4. Ensemble Model: To maximize predictive accuracy and robustness, we developed a hard-voting ensemble model. This approach combines the predictions from our enhanced logistic regression and MLP models, leveraging the diverse strengths of each to produce a final, more reliable forecast. The "wisdom of the crowd" principle behind this model helps to mitigate the individual weaknesses of its components.
# MAGIC
# MAGIC 5. Neural Network: Due to complexity of the relationships in the data, we explored the use of a tensorflow Neural Network(NN). NNs handle high-dimensional inputs better than many classical models because they can learn hierarchical feature representations (from raw or scaled inputs to higher-level patterns) and they are they’re flexible with mixed data.
# MAGIC
# MAGIC All stages of our pipeline were tracked using MLflow, ensuring transparency, reproducibility, and a clear, data-driven path for model comparison and selection. 
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC # VI) Results
# MAGIC ____
# MAGIC Experiment Matrix:
# MAGIC |Model Type|Baseline|Hyperparameters|Wall Time|Recall (Delayed)
# MAGIC |------|--------|-------|-------|-------|
# MAGIC |Logistic Regression (with Time-Series CV)|N/A|Regularization Parameter: 0.01, Elastic Net Parameter: 0.05|4 min|0.65
# MAGIC |Multi-layer Perceptron|Logistic Regression|Number of Nodes per Layer: [9,10,2], Iterations: 50, Step Size: 0.03|13 min|0.69
# MAGIC |Ensemble|Logistic Regression|N/A|N/A|0.73
# MAGIC |Neural Net|Logistic Regression|Number of Nodes per Layer: [256, 32], batch_size: 128, dropout rate: 0.2097, epochs: 20, learning rate: 0.0081, decision threshold: 0.3|1hr 27 min|0.72
# MAGIC
# MAGIC ##Logistic Regression
# MAGIC The results from training on the 2015-2018 dataset and subsequent blind testing on the 2019 data provide insights into the efficacy of logistitic regression, MLP, neural network, and ensemble models for flight delay prediction. We used a simple logistic regression baseline model in our initial experimentation with the one year dataset. On the blind test set representing Q4, the recall for the delayed class was 0.456, indicating that less than half of truly delayed flights were correctly identified. While the model demonstrated strong discrimination for the majority class (on time flights), these results highlight the challenges of modeling delays in a highly imbalanced dataset.
# MAGIC
# MAGIC Recognizing the importance of maximizing the detection of delayed flights, we introduced a blocked time series cross-validation framework paired with grid search hyperparameter tuning into our modeling pipeline. This approach maintained temporal integrity and allowed for validation by simulating the operational process of forecasting future quarters based only on historical data. 
# MAGIC <img src="https://raw.githubusercontent.com/trevorrovert/temp_repo/main/blockingtimeseriessplit.png" alt="Blocking Time-Series Cross-Validation Split" width="50%"/>
# MAGIC
# MAGIC The enhanced logistic regression baseline uses elastic-net regularization to reduce some of the complexity of the model. As such, grid search was used to find the optimum elastic net mixing parameter and regularization parameter. See grid search experiments below: 
# MAGIC <img src="https://github.com/ewadhwan/w261_final_project_images/blob/main/logistic%20regression%20cross%20validation.png?raw=true" alt="Logistic Regression Grid Search" width="50%"/>
# MAGIC
# MAGIC The hyperparameters with the highest mean F-1 score were then used to train the logistic regression on the full dataset. Training the enhanced logistic regression baseline model on 9 month blocks of years 2015-2018 and evaluating it on the holdout 2019 data yielded a recall for the delayed class of 0.65. The subsequent experiments will use this recall result as a benchmark for comparison. The wall time for the grid search and final model training was 4 minutes. A full summary of the performance metrics of the final logistic regression model on the 2019 dataset can be found below:
# MAGIC
# MAGIC <img src="https://github.com/ewadhwan/w261_final_project_images/blob/main/logistic%20regression%20results.png?raw=true" alt="Logistic Regression 2019 Data Evaluation" width="50%"/>
# MAGIC
# MAGIC ##Multilayer Perceptron (MLlib)
# MAGIC
# MAGIC Per the request of the stakeholder, this project also examined using a multi-layer perceptron model to predict flight delays. The hyperparameters (number of layers, number of training iterations, and step size) were chosen using time-blocked cross validation and grid search. The hyperparameters were elected by chosing the parameters that yielded the model with the highest F-1 score. Grid Search with time series cross validation results are shown below:
# MAGIC
# MAGIC <img src="https://github.com/ewadhwan/w261_final_project_images/blob/main/MLP%20grid%20search.png?raw=true" alt="MLP Hyperparameter Grid Search" width="50%"/>
# MAGIC
# MAGIC The highest scoring hyperparameters were then used to train a model on the whole 2015-2018 dataset. The evaluation of this model on the 2019 flight data showed a significant improvement in recall of the delayed class and F-1 score than the time-series cross validation experiments. This is likely because the training blocks used in the cross validation were always months 1-9 and the validation results were months 10-12. As a result, there are some seasonal patterns that may have been missed during the hyperparameter tuning that were captured in the final model tuning. The wall time for the grid search and final model training was 13 minutes. The recall of the delayed class on the 2019 data was 0.69, a small improvement over the logistic regression baseline. A summary of the final MLP model evaluated on the 2019 flight data is shown below:
# MAGIC
# MAGIC <img src="https://github.com/ewadhwan/w261_final_project_images/blob/main/MLP%20results.png?raw=true" alt="MLP 2019 Data Evaluation" width="50%"/>
# MAGIC
# MAGIC ##Ensemble Model
# MAGIC
# MAGIC The ensemble model combines the predictions of the above logistic regression model and MLP model into a simple voting scheme. The ensemble votes delayed (1) if either or both of the logistic regression and the MLP models returns the delayed class. Otherwise, it returns on-time (0). This simple combination resulted in an increase in recall of the delayed class from both individual models. As there was no training for this model, no wall time is reported. When evaluated on the 2019 flight data, the ensemble model yielded the following results:
# MAGIC
# MAGIC <img src="https://github.com/ewadhwan/w261_final_project_images/blob/main/ensemble%202019%20results.png?raw=true" alt="Ensemble 2019 Data Evaluation" width="50%"/>
# MAGIC
# MAGIC ##Tensorflow Neural Network
# MAGIC
# MAGIC Several experiments were conducted and hyperparameter tuning provided the the following parameters with the highest recall values batch_size: 128, dropout_rate: 0.2097
# MAGIC epochs: 20, learning_rate: 0.0081, with 2 hidden layers with values of 256 and 32 units. To further improve the recall for delayed flights which are underrepresented in the dataset, the decision threshold for classifying a flight as delayed was lowered from the default 0.5 to 0.3. This adjustment increased the model’s sensitivity to potential delays, allowing it to capture more true delay cases, albeit with some trade-off in precision. The following table shows the results on both the validation and test datasets on 1% samples of the training data. 1% of the data was used due to compute constraints.
# MAGIC |Metric|Validation results| Test results|
# MAGIC |------|--------|-------|
# MAGIC |Recall (Delayed Class)|0.89|0.715
# MAGIC |Accuracy|0.69|0.797|
# MAGIC |F1 Score (Delayed Class)|0.73|0.569|
# MAGIC |Precision (Delayed Class)|0.61|0.0.472|
# MAGIC |MAE|0.33|0.347|
# MAGIC |MSE|0.16|0.159|
# MAGIC
# MAGIC The Neural Network performed better albeit with some tradeofs. There was very high recall on validation (89%) which means the model was excellent at identifying delays However, on test, recall dropped to ~71%, which is still a good performance for generalizability becuase  the model catches most delays, though not as many as in training/validation. The precision metrics dropped from 61% to 47% in the test set, which means more predicted delays turned out to be false positives.This could be due to a different delay pattern in 2019 vs. training years, or because of the lower decision threshold (e.g., 0.3) that was tuned to favor recall.
# MAGIC
# MAGIC ##Gap Analysis
# MAGIC
# MAGIC While the models presented above made an improvement from the logistic regression baseline, there are some minor faults to be aware of, particularly in terms of data leakage. Data leakage in this context refers to when data from the future (in this case, data from after two hours prior to a flight's scheduled departure) is used to calculate features used by the model in prediction. All four of the models presented above experience some minor data leakage due to the post-processing steps. The imputation step fills null values with the median for numeric features and the mode for boolean features. While the medians and modes were calculated on the training dataset to protect against impacting the model with the testing data, these calculations used the entire scope of the training data. As a result, data from after the two-hours-before-departure cutoff was used in predicions for flights containing null values in their accompanying data. Similarly, the numerical features used by the models were scaled using min-max scaling. The min and max for each of the features were calculated from the entire training dataset, slightly biasing each datapoint with information after the time cutoff. There is also a tradeoff between data leakage and computing power. While the methods for scaling and imputation incur data leakage in the training set, they only involve one median and one min/max calculation for each field. However, imputing and scaling without data leakage would require a rolling calculation of medians and mins/maxes, increasing the time and resources required to process the input data. 
# MAGIC
# MAGIC While these sources of data leakage are ubiquitous, they are minor. Because scaling and imputation values were computed from the training set, the data leakage issue is restricted to the training data. The primary concern of data leakage is that it causes overfitting and falsely optimistic results at training time. However, the 2019 testing data shows that the models still yield acceptable recall values. 
# MAGIC
# MAGIC ##Discussion
# MAGIC
# MAGIC The results demonstrate clear performance gains when moving from the simple logistic regression baseline to more complex architectures and ensemble strategies. The baseline logistic regression, trained without time-series cross-validation, struggled with the highly imbalanced nature of the dataset, achieving a recall of only 0.456 for the delayed class. By introducing blocked time-series cross-validation and tuning elastic-net regularization parameters, the enhanced logistic regression improved recall to 0.65, providing a strong benchmark while maintaining interpretability and relatively low computational cost. This improvement highlights the importance of incorporating temporal validation techniques to mimic real-world forecasting conditions.
# MAGIC
# MAGIC The multilayer perceptron extended the gains seen with logistic regression, increasing recall to 0.69. This improvement can be attributed to the model’s ability to capture non-linear relationships among features, as well as its flexible architecture. The MLP’s performance suggests that flight delay patterns may involve complex interactions beyond what a linear model can capture. However, the model’s improvement over logistic regression was modest, and seasonal patterns likely influenced the results, as hyperparameter tuning used a fixed set of months (1–9) for training in each fold. This limitation points to the need for more seasonally aware validation strategies in future work.
# MAGIC
# MAGIC The ensemble model, which combined logistic regression and MLP predictions through a simple voting scheme, achieved the highest recall of 0.73 among the main comparative models. This result underscores the value of blending complementary models to leverage their individual strengths. While the ensemble does not introduce additional training cost, it benefits from the diversity in decision boundaries between the linear and non-linear approaches. Such a method is particularly appealing in operational contexts where incremental recall improvements translate into meaningful gains in identifying delayed flights.
# MAGIC
# MAGIC The TensorFlow neural network offered competitive performance, achieving a test recall of 0.715 despite being trained on only 1% of the data due to computational constraints. By tuning hyperparameters and lowering the decision threshold from 0.5 to 0.3, the model was optimized for recall at the expense of precision, which dropped to 47% on the test set. This trade off may be acceptable in applications where missing a true delay is more costly than issuing false delay detections. The sharp drop in recall from validation (89%) to test (71%) indicates some overfitting and possible sensitivity to temporal shifts in delay patterns between training and 2019 data. Overall, while all models improved upon the logistic regression baseline, the ensemble approach offered the best balance of recall improvement and operational feasibility.
# MAGIC
# MAGIC Results Summary:
# MAGIC |Model|Recall (Delayed)|Training Wall Time|Key Strengths|Key Limitations|
# MAGIC |------|--------|-------|-------|-------|
# MAGIC |Logistic Regression (Enhanced)|0.65|4 min|Fast, interpretable, good baseline with time-series CV|Limited ability to capture non-linear patterns|
# MAGIC |Multi-Layer Perceptron (MLlib)|0.69|13 min|Captures non-linear relationships, modest recall gain|Seasonal bias in CV, slightly longer training|
# MAGIC |Ensemble (LogReg + MLP)|0.73|N/A|Highest recall, no extra training cost|Slightly lower precision, interpretability reduced|
# MAGIC |TensorFlow Neural Network|0.715|1 hr 27 min (1% data)|Very high recall on validation, flexible architecture|Precision trade-off, overfitting risk, high compute needs|
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC # VII) Conclusion
# MAGIC _____
# MAGIC This project aimed to develop a predictive pipeline for U.S. domestic flight departure delays of 15 minutes or more, using a engineered dataset that makes use of enterprise level technology on data available up to two hours before scheduled departure. Accurate forecasting of delays is critical for aviation stakeholders, enabling proactive resource allocation, minimizing disruptions, and improving passenger experience.
# MAGIC
# MAGIC We hypothesized that machine learning models trained on carefully engineered features that capture temporal patterns, airport congestion, aircraft reliability, and weather conditions could effectively predict delays, even amid class imbalance and data quality challenges. To test this, we built a scalable PySpark pipeline processing over 31 million flights, created 58 predictive features, and implemented both baseline and advanced models with time-aware cross-validation and imbalancing strategies.
# MAGIC
# MAGIC Key contributions include a robust data preprocessing framework, feature engineering of aircraft usage and congestion patterns, and evaluation across multiple models. The multilayer perceptron performed best, particularly in identifying true delays; this was a critical success in this imbalanced classification task. Our results affirm the importance of domain-specific features and recall-focused evaluation in operationally relevant ML systems.
# MAGIC
# MAGIC Several key surprises and lessons emerged throughout the project. One unexpected finding was that weather features were not strong predictors of departure delays in our models. Our exploratory data analysis revealed that weather data was often sparse, noisy, and less correlated with delay outcomes than features related to airport congestion, day-of-week patterns, and aircraft turnaround performance. This challenges common assumptions in the domain and highlights the importance of rigorous feature validation over intuition. 
# MAGIC
# MAGIC Another critical realization was the importance of checkpointing and memory-efficient coding practices in PySpark. As we scaled up to processing over 31 million flight records with increasingly complex transformations, we encountered out-of-memory errors and long execution times, particularly during cross-validation and model training. Implementing strategic checkpoints, minimizing wide shuffles, and optimizing joins became essential to maintaining stability and performance in the distributed environment.
# MAGIC
# MAGIC For future iterations, investing in more scalable pipeline design, early feature pruning, and monitoring resource usage more proactively will be vital to improving both development speed and model throughput. These insights reinforce the dual importance of domain-aware feature engineering and infrastructure-aware data processing in building robust, production-ready machine learning systems.
# MAGIC
# MAGIC This work lays the foundation for real-time flight delay prediction. Future efforts could extend to include live weather integration, air traffic data incorporation, and deployment as a real-time decision support tool for airline operations centers.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Appendix
# MAGIC _____

# COMMAND ----------

# MAGIC %md
# MAGIC ### A) Reference to Code Notebooks
# MAGIC
# MAGIC - Checkpoint Functions: https://dbc-fae72cab-cf59.cloud.databricks.com/editor/notebooks/2084015877078680?o=4021782157704243
# MAGIC - EDA: https://dbc-fae72cab-cf59.cloud.databricks.com/editor/notebooks/3831723723989602?o=4021782157704243
# MAGIC - Data Preprocessing: https://dbc-fae72cab-cf59.cloud.databricks.com/editor/notebooks/893645516381367?o=4021782157704243
# MAGIC - Feature Engineering: https://dbc-fae72cab-cf59.cloud.databricks.com/editor/notebooks/2084015877078661?o=4021782157704243
# MAGIC - Data Post-processing & Pipeline: https://dbc-fae72cab-cf59.cloud.databricks.com/editor/notebooks/3831723723990379?o=4021782157704243
# MAGIC - ML Model Training: 
# MAGIC   - Baseline, MLP, NN: https://dbc-fae72cab-cf59.cloud.databricks.com/editor/files/3631386026269581?o=4021782157704243
# MAGIC   - Ensemble Model: https://dbc-fae72cab-cf59.cloud.databricks.com/editor/notebooks/3600546862782042?o=4021782157704243
# MAGIC - Github repo for storing images: 
# MAGIC   - https://github.com/DrZubi/MIDS_261/tree/main
# MAGIC   - https://github.com/hngondoki/forecasting-flight-departure-delays/tree/main/images
# MAGIC   - https://github.com/ronghuang0604/MIDS_261_ML_at_Scale/tree/main
# MAGIC   - https://github.com/ewadhwan/w261_final_project_images.git
# MAGIC   

# COMMAND ----------

# MAGIC %md
# MAGIC ### B) Data dictionary of the raw features 
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

# COMMAND ----------

# MAGIC %md
# MAGIC ### C) Newly Joined Dataset Summary Statistcis
# MAGIC |index|null_counts|percentage|mean|stddev|min|max|median|
# MAGIC |:----|:----|:----|:----|:----|:----|:----|:----|
# MAGIC |Prev_FlightDate|7895|0.02|null|null|null|null|null|
# MAGIC |OntimeArrivalPct|24694|0.08|82.25|5.07|0.0|100.0|82.44|
# MAGIC |OntimeDeparturePct|24694|0.08|80.3|6.89|0.0|100.0|80.36|
# MAGIC |historical_ontime_arrival_pct|24694|0.08|82.25|5.07|0.0|100.0|82.44|
# MAGIC |historical_ontime_departure_pct|24694|0.08|80.3|6.89|0.0|100.0|80.36|
# MAGIC |short_term_ontime_arrival_pct|29223|0.09|82.2|6.41|0.0|100.0|82.73|
# MAGIC |short_term_ontime_departure_pct|29223|0.09|80.32|8.09|0.0|100.0|81.03|
# MAGIC |Tail_Number|72487|0.23|8806.54|1.96|8805.0|8809.0|8805|
# MAGIC |recent_ontime_arrival_pct|97865|0.31|82.32|10.08|0.0|100.0|83.64|
# MAGIC |recent_ontime_departure_pct|97865|0.31|80.49|11.61|0.0|100.0|82.05|
# MAGIC |DEP_TIME|472320|1.49|1334.21|503.29|1.0|2400.0|1325|
# MAGIC |DepDel15|477296|1.5|0.18|0.39|0.0|1.0|0|
# MAGIC |TAXI_OUT|486417|1.53|16.83|9.49|0.0|227.0|15|
# MAGIC |Prev_TaxiOut|494171|1.56|16.83|9.49|0.0|227.0|15|
# MAGIC |ARR_TIME|501922|1.58|1468.9|536.36|1.0|2400.0|1508|
# MAGIC |TAXI_IN|501924|1.58|7.56|5.93|0.0|414.0|6|
# MAGIC |Prev_ArrTime|509669|1.61|1468.83|536.32|1.0|2400.0|1505|
# MAGIC |Prev_TaxiIn|509671|1.61|7.56|5.93|0.0|414.0|6|
# MAGIC |ARR_DELAY|570640|1.8|4.62|45.59|-238.0|2695.0|-5|
# MAGIC |Prev_ArrDelay|578365|1.82|4.61|45.59|-238.0|2695.0|-5|
# MAGIC |Turnaround_Time|879857|2.77|405.97|693.11|-2001.0|4758.0|105|
# MAGIC |HourlyDryBulbTemperature|1241972|3.91|56.96|21.64|-56.0|140.0|59.67|
# MAGIC |HourlyRelativeHumidity|1300673|4.1|68.58|22.76|1.0|100.0|72|
# MAGIC |HourlyDewPointTemperature|1304469|4.11|44.59|20.35|-89.5|96.0|46|
# MAGIC |HourlyWindSpeed|1309347|4.12|7.8|6.29|0.0|1122.5|7|
# MAGIC |HourlyVisibility|1536567|4.84|8.97|2.82|0.0|95.0|10|
# MAGIC |HourlyWetBulbTemperature|2374516|7.48|50.72|18.65|-44.0|96.0|53|
# MAGIC |HourlySkyConditions|3271855|10.31|30.86|21.75|0.0|74.0|26|
# MAGIC |HourlySeaLevelPressure|19106405|60.18|30|0.24|28.37|31.25|30|
# MAGIC |HourlyPrecipitation|23552602|74.19|0.01|0.07|0.0|5.76|0|
# MAGIC |HourlyWindGustSpeed|24778097|78.05|22.94|6.2|11.0|94.0|21.67|
# MAGIC |HourlyPresentWeatherType|26683438|84.05|null|null|null|null|null|
# MAGIC
# MAGIC
# MAGIC ### Shapiro Wilk Test Results for Newly Joined Dataset
# MAGIC |Column|statistic|p_value|
# MAGIC |:----|:----|:----|
# MAGIC |CANCELLED|1|1|
# MAGIC |cancel_flag|1|1|
# MAGIC |DEP_TIME|0.9890479375080403|2.9026848067250615e-19|
# MAGIC |CRSDepTime|0.9886539995269192|1.2540719845221062e-19|
# MAGIC |Prev_ArrTime|0.9885942298123761|1.1061544301250833e-19|
# MAGIC |HourlyDryBulbTemperature|0.9849427323515518|1.112412341119789e-22|
# MAGIC |Hour|0.9838179338521951|1.7039328217930747e-23|
# MAGIC |HourlySeaLevelPressure|0.9821351457851466|1.2231404536062378e-24|
# MAGIC |HourlyDewPointTemperature|0.977139086052452|1.2876694631283415e-27|
# MAGIC |HourlyWetBulbTemperature|0.9757989146054689|2.490318962582302e-28|
# MAGIC |HourlyRelativeHumidity|0.9650118116922295|3.519888020052843e-33|
# MAGIC |CRS_ARR_TIME|0.960804746079183|9.362191625155041e-35|
# MAGIC |DayofMonth|0.9593935021910212|2.9706448396421865e-35|
# MAGIC |HourlyWindSpeed|0.9578151463040001|8.535943322323247e-36|
# MAGIC |recent_ontime_arrival_pct|0.9554533713271902|1.4127095732058116e-36|
# MAGIC |recent_ontime_departure_pct|0.954783247176138|8.597921308422552e-37|
# MAGIC |ARR_TIME|0.9498787402545389|2.6852615249316e-38|
# MAGIC |short_term_ontime_departure_pct|0.9496899896153104|2.3628772935995205e-38|
# MAGIC |Month|0.9434064572302894|4.10231921275725e-40|
# MAGIC |short_term_ontime_arrival_pct|0.94073403873852|8.158852595874001e-41|
# MAGIC |OriginAirportID|0.9362990937286779|6.337824948714536e-42|
# MAGIC |DestAirportID|0.934294028561119|2.0923067352270548e-42|
# MAGIC |DayOfWeek|0.9200702559100247|1.6259065664064448e-45|
# MAGIC |Flight_Number_Reporting_Airline|0.9193970684119595|1.1897589787248372e-45|
# MAGIC |OntimeDeparturePct|0.9164492007353906|3.1058862347522657e-46|
# MAGIC |historical_ontime_departure_pct|0.9164492007353906|3.1058862347522657e-46|
# MAGIC |HourlyWindGustSpeed|0.9138419568738245|9.776030060832226e-47|
# MAGIC |DestWac|0.9126833143479318|5.9026560552617355e-47|
# MAGIC |Oncoming_flights|0.9011038621946983|5.044960760099666e-49|
# MAGIC |OriginWac|0.8981119668795595|1.5886133638285989e-49|
# MAGIC |OntimeArrivalPct|0.8975495237607151|1.2824376680564218e-49|
# MAGIC |historical_ontime_arrival_pct|0.8975495237607151|1.2824376680564218e-49|
# MAGIC |Year|0.8835570905113399|8.32264257507857e-52|
# MAGIC |CRSElapsedTime|0.8811722312995192|3.709548172184155e-52|
# MAGIC |Distance|0.8751627471835948|5.131134521735883e-53|
# MAGIC |DistanceGroup|0.8724334399094235|2.144475601880746e-53|
# MAGIC |Quarter|0.8591715474590791|3.809405701622243e-55|
# MAGIC |Prev_TaxiOut|0.7724793080530816|4.759185356994541e-64|
# MAGIC |TAXI_OUT|0.7455327122304156|3.1726203615209865e-66|
# MAGIC |Num_airport_wide_delays|0.7180646749795354|2.9909022230143725e-68|
# MAGIC |Prev_TaxiIn|0.6938332358905311|6.663975816782392e-70|
# MAGIC |TAXI_IN|0.6643900135906269|9.122690622427282e-72|
# MAGIC |HourlyVisibility|0.5752742269794862|1.1629469361201497e-76|
# MAGIC |IsWeekend|0.5739846260987003|1.0033908229834585e-76|
# MAGIC |ARR_DELAY|0.5118295400697428|1.242208543047728e-79|
# MAGIC |Prev_ArrDelay|0.506833045646012|7.491867177993162e-80|
# MAGIC |DepDel15|0.5058354988617366|6.776039975126069e-80|
# MAGIC |delay_flag|0.5058354988617366|6.776039975126069e-80|
# MAGIC |Turnaround_Time|0.4946486484934731|2.2233065543356185e-80|
# MAGIC |IsHolidayMonth|0.4942355970832374|2.1345532441827824e-80|
# MAGIC |Num_airport_wide_cancelations|0.29856390012489586|1.22195030037885e-87|
# MAGIC |HourlyPrecipitation|0.20285085141283932|1.471783284533358e-90|
# MAGIC |IsBusinessHours|0.0998744645413695|2.2384747588425854e-93|
# MAGIC
# MAGIC
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## D) Outlier Graphs
# MAGIC <div style="text-align: center; max-width: 400px; margin: auto;">
# MAGIC   <h6 style="font-family: Arial, sans-serif; font-weight: bold; margin-bottom: 10px;">
# MAGIC   </h6>
# MAGIC   <img 
# MAGIC     src="https://github.com/hngondoki/forecasting-flight-departure-delays/blob/main/images/box_plot_new_dataset.png?raw=true" 
# MAGIC     alt="Newly Joined Dataset Boxplots" 
# MAGIC     style="width: 600; height: 800; border-radius: 8px; box-shadow: 0 0 8px rgba(0,0,0,0.15);"
# MAGIC   >
# MAGIC </div>

# COMMAND ----------

