# Motor Failure Prediction

## Overview

This project aims to predict motor failure in the next cycle using machine learning techniques. The maximum cycle of the motor is the last cycle before it breaks down. The model is designed to forecast the remaining cycles until a motor breaks down based on sensor readings and settings.

## Table of Contents

- [Introduction](#introduction)
- [Data Description](#data-description)
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

## Usage

To use this project, follow these steps:

1. Clone the repository:

   ```bash
   git clone https://github.com/elnurmmmdv/motor-failure-prediction.git
2. Install dependencies::

   ```bash
   pip install -r requirements.txt

3. Run the Jupyter Notebook:

   ```bash
   jupyter notebook
   
4. Open and run the provided notebook (motor_failure_prediction.ipynb).


## Data Preprocessing

The preprocessing steps include sorting the data, filtering columns with less than 2 distinct values, creating new features to find the change between the current and previous state of parameters, handling null values, creating the time_to_failure label, splitting the data into training and testing sets based on motor IDs, and feature scaling.

## Model
The project includes a Python class Model located in the model.py file. This class provides functionality for building, compiling, and training an LSTM-based model for motor failure prediction. The model supports two types: many-to-many (m_to_m) and many-to-one (m_to_1). The sequence type can be set to either sliding window (sliding_window) or padding (padding).

### Examples
The notebook provides examples of how to use the Model class for different scenarios, including m_to_m with sliding window, m_to_m with padding, and m_to_1 with sliding window.

### Results
The training history and performance metrics are visualized using matplotlib, showing the loss and mean absolute error (MAE) for both training and testing sets.
