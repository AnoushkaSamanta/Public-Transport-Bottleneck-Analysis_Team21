# Comprehensive Project Documentation
# Public Transport Delay Propagation and Bottleneck Prediction

## 1. Project Overview
**Objective:** To build an end-to-end data mining, machine learning, and deep learning pipeline to analyze public transport delays, identify systemic bottlenecks, and predict future congestion patterns.

**Technologies Used:**
- **Programming Language:** R
- **Libraries/Packages:** `dplyr`, `tidyr`, `lubridate`, `ggplot2`, `caret`, `randomForest`, `e1071` (SVM), `keras`/`tensorflow` (LSTM), `cluster`, `dbscan`, `FactoMineR`, `igraph` (Network Analysis).
- **Visualization:** R (ggplot2, plotly) and Power BI.

---

## 2. Project Pipeline & Methodology (Exact Code Interpretations)

### Phase 1: Data Collection & Profiling (`01_load_and_profile.R`)
- **Data Source:** Raw public transport dataset (`public_transport_delays.csv`).
- **Profiling:** Generates summary statistics and profiles missing value frequencies, saved to `reports/data_profile.csv` and `reports/missing_value_summary.csv`.

### Phase 2: Data Cleaning (`02_data_cleaning.R`)
- **Missing Value Handling (Exact Strategies):**
  - **Dropped Rows:** Records missing the primary key (`trip_id`), missing `date`, or missing *both* `actual_departure_delay_min` and `actual_arrival_delay_min` are completely dropped. 
  - **Median Imputation:** Used strictly for purely numerical variables (`temperature_C`, `humidity_percent`, `wind_speed_kmh`, `precipitation_mm`, `event_attendance_est`, `traffic_congestion_index`, and instances where only one of the delay minutes is missing).
  - **Mode Imputation:** Used for all categorical, binary, and unparsed string columns (`transport_type`, `weather_condition`, `event_type`, `season`, `route_id`, `origin_station`, `destination_station`, `holiday`, `peak_hour`, `delayed`, `weekday`, `time`, `scheduled_departure`, `scheduled_arrival`).
- **Duplicates:** Full row duplicates are removed using `dplyr::distinct()`. For duplicate `trip_id`s, the first occurrence is kept.
- **Outlier Handling (IQR Winsorization):** Extreme outliers are explicitly handled using the Interquartile Range (IQR) method—NOT Z-scores. Values falling below `Q1 - 1.5 * IQR` or above `Q3 + 1.5 * IQR` in columns like delay times, temperature, and attendance are capped at those bounds (Winsorized). Tracked in `reports/outlier_detection_log.csv`.
- **Logical Validation:** Detects overnight cases (arrival < departure) and flags circular trips (`origin_station` == `destination_station`).

### Phase 3: Data Transformation & Feature Engineering (`03_transformation.R`)
- **Temporal Vectors:** Extracted using `lubridate` (converting `date` + `time` into `POSIXct` datetime). Extracted features include `hour`, `day_name`, `is_weekend` (0 vs 1), `month`, and `time_of_day_bucket` (Night, Morning, Afternoon, Evening).
- **Categorical Encoding (Label Encoding):** The script does NOT use dummy/one-hot encoding. Instead, it uses strictly mapped label encoding for categoricals:
  - *Transport:* Bus = 1, Metro = 2, Train = 3, Tram = 4
  - *Weather:* Clear = 1, Cloudy = 2, Fog = 3, Rain = 4, Snow = 5, Storm = 6
  - *Event:* None = 0, Concert = 1, Festival = 2, Parade = 3, Protest = 4, Sports = 5
  - *Season:* Winter = 1, Spring = 2, Summer = 3, Autumn = 4
- **Interaction Features:** Created combined metrics like `weather_severity_index` (weighted metric of wind, precipitation, and storms), `feels_like_impact` (temperature × wind chill formula), and `event_impact_score` (attendance × congestion).
- **Dual-Strategy Scaling:** Both standardizations are performed on 15 numeric features:
  - **Z-Score Normalization (`_z` suffix):** Calculated by `(X - Mean) / Standard Deviation`. These variables are used heavily during PCA and clustering (K-Means/DBSCAN) spatial calculations.
  - **Min-Max Scaling (`_mm` suffix):** Calculated by `(X - Min) / (Max - Min)` clamping values to [0,1]. These variables are directly prepared for Deep Learning (LSTM algorithms).

### Phase 4: Exploratory Data Analysis (EDA) (`04_eda.R`)
- Extensive univariate and bivariate analysis. Visualizations of delay distributions across different routes, stations, and times of day.
- Output metrics are exported to support PowerBI dashboards (`reports/powerbi_hourly_patterns.csv`).

### Phase 5: Dimensionality Reduction (`05_pca.R`)
- **Principal Component Analysis (PCA):** Applied to highly correlated numeric (Z-scored) variables to reduce multicollinearity and the feature workspace while plotting variance.
- **Outputs:** `pca_transformed_data.csv`, `pca_model.rds`

### Phase 6: Clustering & Bottleneck Identification (`06_clustering.R`)
- **K-Means Clustering:** Groups stations or routes using the standardized `_z` vectors.
- **DBSCAN:** Density-based clustering to map severe irregular density (anomalies) across stations.

### Phase 7: Predictive Classification (`07_classification.R`)
- **Goal:** Predict whether a future route will be delayed. Evaluated as a Binary Factor ("No" vs "Yes").
- **Train-Test Evaluation Setup:** Stratified 80/20 train/test split. A check is run for target class imbalance. If the minority class is < 35%, `caret` invokes an UP-sampling strategy to balance the subsets. Trained with a **5-fold Cross Validation**.
- **Features Extracted:** Uses exclusively pre-trip known variables (no actual delay metrics).
- **Models Built:**
  1. **Logistic Regression (`glm` binomial):** Serves as an interpretable baseline model.
  2. **Random Forest (`method = 'rf'`):** Default set to `ntree = 100`. Captures complex interactions across encoded labels.
  3. **SVM Radial (`svmRadial`):** Tuned `tuneLength = 5` to map non-linear boundary classes.
- **Outputs:** Evaluation based on Accuracy, Precision, Recall, and AUC-ROC.

### Phase 8: Deep Learning Forecasting (`08_lstm.R`)
- **Goal:** Sequential time-series prediction of upcoming delays.
- **Algorithm:** Long Short-Term Memory (LSTM) created in Keras/TensorFlow. Ingests the Min-Max (`_mm`) scaled features formatted into 3D tensors: `[samples, timesteps, features]`.

### Phase 9: Delay Propagation & Network Analysis (`09_propagation.R`)
- **Analysis:** Employs the `igraph` package to identify route "nodes" (stations) and "edges" (trips) tracking how delays exponentially cascade from one terminal to the next.- **Outputs:** `interactive_delay_network.html` and `station_network_metrics.csv`.

### Phase 10: Reporting & BI Integration (`10_reporting.R`)
- Compiles all metrics, model comparisons, and cluster summaries into uniform CSVs optimized for ingestion into Power BI.
- Output files like `powerbi_overall_summary.csv` and `model_comparison.csv` serve as the semantic layer for dashboards.

---

## 3. Key Conclusions & Outputs

1. **Systemic Bottlenecks:** Clustering (K-Means/DBSCAN) effectively segregates high-risk zones. The profiles reveal that specific hub stations and peak hours universally contribute to the most severe bottlenecks.
2. **Predictive Accuracy:** Ensemble methods (Random Forest) generally outperform linear baseline models in identifying categorical delay events due to complex spatial-temporal feature interactions.
3. **Forecasting Power:** The LSTM neural network proves effective at anticipating delay spikes by learning from recent sequence windows, enabling proactive crowd-control or vehicle-dispatch configurations.
4. **Network Cascades:** The propagation analysis highlights the "domino effect"—identifying the exact nodes (stations) where localized interventions resolve network-wide cascading delays.

## 4. How to Run the Pipeline
The entire stack is orchestrated by `scripts/run_all.R`, which sequentially executes modules `00` through `10`. Ensure all raw data is placed in the `data/` directory and run:
`source("scripts/run_all.R")`
