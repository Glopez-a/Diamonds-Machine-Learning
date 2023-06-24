# Diamond Price Prediction

This repository contains a machine learning model for predicting diamond prices. The project includes exploratory data analysis (EDA), data preprocessing, feature engineering, label encoding, and utilizes the GradientBoost model with grid search for optimal hyperparameter tuning.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [EDA](#EDA)
- [Project Structure](#project-structure)


## Introduction

The aim of this project is to develop a machine learning model that accurately predicts the price of diamonds based on their various features. By leveraging the power of GradientBoost algorithm and performing grid search for hyperparameter tuning, we strive to achieve the best possible performance.

The project consists of several stages, including exploratory data analysis to gain insights into the dataset, data preprocessing to handle missing values and normalize features, feature engineering to create new meaningful features, label encoding to convert categorical variables into numerical representations, and training the GradientBoost model using grid search for hyperparameter optimization.

## Dataset

The dataset used for this project is the Diamond Dataset, which includes various attributes of diamonds such as carat weight, cut quality, color, clarity, dimensions, and price. The dataset is not included in this repository, but it can be obtained from a reliable source such as Kaggle or a diamond data provider.

## EDA

Before building the machine learning model, it is essential to gain insights into the diamond dataset through exploratory data analysis (EDA). The EDA helps us understand the relationships between the features and the target variable, identify patterns, and uncover any anomalies or outliers in the data.

To perform the EDA, we use the `pandas`, `matplotlib`, and `seaborn` libraries in Python. Here's a summary of the EDA steps and visualizations:

- Loading the Dataset:
  - The diamond dataset is loaded from a .db file named 'diamonds_train.db' located in the 'data/' directory.
  - The first few rows of the dataset are displayed to get a glimpse of the data.

- Summary Statistics:
  - The `describe()` function is used to compute summary statistics such as count, mean, standard deviation, minimum, and maximum values for each numerical column in the dataset.
  - These statistics provide an overview of the distribution and range of values in the dataset.

- Correlation Matrix:
  - The correlation matrix is calculated using the `corr()` function from `pandas`.
  - A heatmap is created using `seaborn` to visualize the correlations between different features.
  - This visualization helps identify strong positive or negative correlations between features, which can indicate important relationships in the dataset.
    [EDA Heatmap](images/corr.png)

- Scatter Plot: Carat vs. Price:
  - A scatter plot is generated using `seaborn` to visualize the relationship between carat (diamond weight) and price.
  - The scatter plot helps understand the general trend and any potential outliers in the data.
  - It provides insights into how the diamond price varies with carat weight.

- Box Plot: Cut vs. Price:
  - A box plot is created using `seaborn` to analyze the relationship between the cut quality of the diamonds and their prices.
  - The box plot displays the distribution of prices for different diamond cuts.
  - It helps identify any significant price differences based on the cut quality.

The EDA process provides valuable insights into the dataset, enabling us to make informed decisions during preprocessing, feature engineering, and model selection. Feel free to explore and modify the EDA code to suit your specific requirements and gain a deeper understanding of the diamond dataset.

## Project Structure

The project structure is organized as follows:

- `data/`: Directory to store the diamond dataset.
- `src/`: Directory containing the source code for the project.
- `data_preprocessing.py`: Module for data preprocessing tasks.
- `eda.ipynb`: Jupyter Notebook for exploratory data analysis.
- `feature_engineering.py`: Module for feature engineering.
- `label_encoding.py`: Module for label encoding of categorical variables.
- `main.py`: Main script for running the project.
- `model.py`: Module containing the GradientBoost model and grid search.
- `utils.py`: Utility functions used throughout the project.
- `README.md`: Readme file providing information about the project.
- `requirements.txt`: File listing the required dependencies.


