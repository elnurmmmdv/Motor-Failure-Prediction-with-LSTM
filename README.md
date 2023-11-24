# Motor Failure Prediction

## Overview

This project aims to predict motor failure in the next cycle using machine learning techniques. The maximum cycle of the motor is the last cycle before it breaks down. The model is designed to forecast the remaining cycles until a motor breaks down based on sensor readings and settings.

## Table of Contents

- [Introduction](#introduction)
- [Data Description](#data-description)
- [Installation](#installation)
- [Usage](#usage)
- [Data Preprocessing](#data-preprocessing)
- [Model](#model)
- [Examples](#examples)
- [Results](#results)


## Introduction

Predictive maintenance is crucial in various industries to minimize downtime and optimize resource utilization. This project offers a solution for predicting motor failure, allowing for proactive maintenance.

## Data Description

The dataset includes the following features:

- `id`: Motor identifier
- `cycle`: Operation cycle
- `p00-p20`: Sensor readings taken during operation
- `s0`, `s1`: Settings changed at the end of each cycle

## Installation

To use this project, follow these steps:

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/motor-failure-prediction.git
2. Install dependencies::

   ```bash
   pip install -r requirements.txt

## Usage
To use the model, follow the examples provided in the [Examples](#examples) section.

## Data Preprocessing
### Sorting Data
The dataset is initially sorted by id and cycle to maintain ordering.
```python
data = data.sort_values(by=['id', 'cycle'])
```

### Removing Columns with Less Than 2 Distinct Values
```python
filtered_data = data.loc[:, data.nunique() >= 2].copy()
```

### Creating New Features to Find Change Between Current and Previous State of Parameters
```python
filtered_data = pd.concat(
    [
        filtered_data,
        filtered_data.groupby('id').diff().iloc[:, 1:].rename(
            {col: col + '_change' for col in filtered_data.columns[2:]}
        )
    ], axis=1
)
```

### Handling Null Values
```python
filtered_data = filtered_data.dropna()
```
### Creating Time_to_Failure Column
``` python

filtered_data['time_to_failure'] = (filtered_data.groupby('id')['cycle'].transform('max') - filtered_data['cycle'])
```
### Model
The project includes a Python class Model located in the model.py file. This class provides functionality for building, compiling, and training an LSTM-based model for motor failure prediction. The model supports two types: many-to-many (m_to_m) and many-to-one (m_to_1). The sequence type can be set to either sliding window (sliding_window) or padding (padding).

### Examples
The notebook provides examples of how to use the Model class for different scenarios, including m_to_m with sliding window, m_to_m with padding, and m_to_1 with sliding window.

Model 1: m_to_m with Sliding Window
```python
model_1 = Model(model_type='m_to_m')
model_1.prepare_data(train_data, test_data, sequence_type='sliding_window')
model_1.train_model(epochs=50, batch_size=64)
model_1.plot_training_history()
```

Model 2: m_to_m with Padding
```python
model_2 = Model(model_type='m_to_m')
model_2.prepare_data(train_data, test_data, sequence_type='padding')
model_2.train_model(batch_size=2)
model_2.plot_training_history()
```

Model 3: m_to_1 with Sliding Window
```python
model_3 = Model(model_type='m_to_1')
model_3.prepare_data(train_data, test_data, sequence_type='sliding_window')
model_3.train_model(batch_size=256)
model_3.plot_training_history()
```

### Results
The training history and performance metrics are visualized using matplotlib, showing the loss and mean absolute error (MAE) for both training and testing sets.
