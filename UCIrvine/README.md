# Individual Household Electric Power Consumption Dataset

## 📊 Dataset Overview

This dataset contains measurements of electric power consumption in a single household over multiple years, with a sampling rate of one measurement per minute.

**Source:** UCI Machine Learning Repository  
https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption

**Sampling rate:** 1 minute  
**Time span (observed in dataset):** January 2007 — September 2010  

---

## 📈 Dataset Statistics

| Property | Value |
|---|---|
| Number of rows | 2,075,260 |
| Number of features | 9 |
| File size | 128 MB |
| Missing values | 155,874 |
| Missing value representation | `?` |
| Data frequency | 1 measurement per minute |

---

## 📁 Structure

```
household_power_consumption.csv
```

Each row represents one minute of electrical measurements.

---

## 🧩 Feature Description

### Data Schema

```
Power Consumption Record
│
├── Date → measurement date (dd/mm/yyyy)
├── Time → measurement time (hh:mm:ss)
│
├── Global_active_power (kW)
│   └── Total active power consumed by the household
│
├── Global_reactive_power (kW)
│   └── Reactive power consumed by the household
│
├── Voltage (V)
│   └── Average voltage per minute
│
├── Global_intensity (A)
│   └── Average current intensity
│
├── Sub_metering_1 (Wh)
│   └── Kitchen appliances (dishwasher, oven, microwave)
│
├── Sub_metering_2 (Wh)
│   └── Laundry appliances (washing machine, dryer, refrigerator)
│
└── Sub_metering_3 (Wh)
    └── Electric water heater and air conditioning
```

---

## 📐 Feature Units

| Feature | Unit | Description |
|---|---|---|
| Date | — | Measurement date |
| Time | — | Measurement time |
| Global_active_power | kilowatt (kW) | Total active power consumption |
| Global_reactive_power | kilowatt (kW) | Reactive power |
| Voltage | volt (V) | Average voltage |
| Global_intensity | ampere (A) | Current intensity |
| Sub_metering_1 | watt-hour (Wh) | Kitchen energy consumption |
| Sub_metering_2 | watt-hour (Wh) | Laundry energy consumption |
| Sub_metering_3 | watt-hour (Wh) | Heating / AC energy consumption |

---

## 🔍 Data Quality Analysis

### Missing Values

- Missing values are represented as `?`
- Total missing values: **155,874**
- Some rows contain missing values across multiple features.
- Missing values must be handled before modeling.

Example:

```
21/12/2006,11:23:00,?,?,?,?,?,?,
21/12/2006,11:24:00,?,?,?,?,?,?,
```

---

### Date Range

Observed values:

- **Start:** 01/01/2007
- **End:** 09/09/2010

---

### Global Active Power Statistics (Exploratory)

| Metric | Value |
|---|---|
| Approximate mean | 0.671 kW |
| Missing values present | Yes |

Note: Statistics should be recomputed after cleaning missing values.

---

## ⚠️ Data Characteristics

- Time series dataset with minute-level resolution
- Multivariate energy consumption measurements
- Contains missing values
- Sub-metering features represent specific household energy usage areas
- Suitable for time series analysis and anomaly detection

---

## 🎯 Potential Use Cases

- Energy consumption forecasting
- Power usage anomaly detection
- Load pattern analysis
- Smart energy monitoring
- Time series modeling
- Predictive maintenance
- Household behavior modeling

---

## 🧹 Recommended Preprocessing Steps

Before using the dataset:

1. Handle missing values (`?`)
2. Convert Date + Time into a timestamp
3. Convert data types to numeric
4. Normalize or scale features
5. Aggregate or resample if needed

---

## 🚀 Project Context

This dataset is used for anomaly detection in household electrical consumption patterns.

