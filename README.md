# Laptop Price Prediction

## Problem Statement :
* Designed a web app that predicts the price of the laptop given the configurations.
* Scraped the laptops data from smartpix.com using selenium and BeautifulSoup package
* Use various ML/DL approach to find best model.
* Deploy model using flask library on cloud.

## Web Scraping :
```bash
Data is scraped from smartpix.com website using selenium & BeautifulSoup. We try to extract the 
different features of the laptops such as:
- model name
- price
- score
- processor
- no of cores
- storage
- ram
- display
- os
- warranty
```

## Dataset :
```bash
We have scraped the data from smartpix.com.
The raw dataset contains 11 features and 1020 entries.`price` is the dependent feature 
and ramaining are the independent features.
```

## Attribute Information :
```bash
- model - brand & model name of the laptop.
- price - price of the laptop in rupee.
- score - score/ratings of each model.
- processor - types of the processor e.g. intel , ryzen wrt their generations.
- no_of_core - number of cores each processor have e.g.Dual Core,Quad core, Hexa core, Octa Core etc
- storage - type of storage e.g Hard disk, SSD in GB and TB
- ram - RAM of the particular model in GB.
- display - screen resolution of the laptop in pixels and touch type.
- os - Operating system type e.g Windows,Mac,Ubuntu,Android
- warranty - Warranty offered by companies in years and months.
```

## Data Assesment & Cleaning
```bash
- Types of Assessment
There are 2 types of assessment styles
1. Manual - Looking through the data manually in google sheets
2. Programmatic - By using pandas functions such as info(), describe() or sample()

Scraped dataset have various quality issues and tidiness issues.Please refer 
data_accessing_and_cleaning.ipynb for more details.

After dataset cleaning we are able to extract more information from raw dataset.Initially we have 11 features which further breakdowns to 17 features and 1016 entries.

```

## Attribute Information after data cleaning:
```bash
- brand_names - 'category' - brand names eg. Apple,HP etc.
- model name - 'object' - model name of the laptop eg. Inspiron, Lattitude.
- price - 'int' price of the laptop in rupee.
- score - 'float' score/ratings of each model.
- processor brand - 'object' types of the processor e.g. Intel , AMD etc.
- processor type - 'object' - types of processor i3,i5,ryzen etc.
- processor gen - 'float' - generation of processor eg. 3rd,5th,7th etc.
- type of core - 'object' -number of cores each processor have e.g.Dual Core,Quad core, Hexa core, Octa Core etc.
- no of threads - 'float' - no of therads.
- storage type - 'object' -type of storage e.g Hard disk, SSD
- storage capcity - 'float' - storage capacity in GB's
- Ram - RAM of the particular model in GB.
- ram type - 'object' - ram type eg.DDR4,LPDDR5 etc
- screen size - 'float' screen size in inches.
- screen resolution - 'object' - screen resolution in pixels.
- OS - 'object' - Operating system type e.g Windows,Mac,Ubuntu,Android
- warranty - 'object' Warranty offered by companies in years.
``` 

## EDA :
```bash
- `Univariate analysis` : Analyze each variable individually to understand their distribution and characteristics.

- `Bivariate analysis` : Analyze the relationship between two variables to understand the impact of one variable on the other.

- `Multivariate analysis` : Analyze the relationship between multiple variables to identify patterns and trends in the data.

- `Data visualization` : Use visualizations like histograms, scatter plots,bar charts and box plots to understand the data.
```

## Attribute Information after EDA:
```bash
We have now 15 features with 1016 records.
- brand_names - 'object' - brand names eg. Apple,HP etc.
- price - 'int' price of the laptop in rupee.
- score - 'float' score/ratings of each model.
- processor brand - 'object' types of the processor e.g. Intel , AMD etc.
- processor type - 'object' - types of processor i3,i5,ryzen etc.
- processor gen - 'float' - generation of processor eg. 3rd,5th,7th etc.
- type of core - 'float' -number of cores each processor have e.g.2,4,6,8 etc.
- no of threads - 'float' - no of therads.
- storage type - 'object' -type of storage e.g Hard disk, SSD
- storage capcity - 'float' - storage capacity in GB's.
- Ram - 'int' - RAM of the particular model in GB.
- ram type - 'object' - ram type eg.DDR4,LPDDR5 etc
- ppi - 'float' - pixel per inches
- OS - 'object' - Operating system type e.g Windows,Mac,Ubuntu,Android
- warranty - 'float' Warranty offered by companies in years.
``` 

## Pre-processing:
```bash
- `Missing value imputation` : 
    1. Complete Case Analysis-Drop the row 
    #This method is used when data is missed at random(MCAR)
    2. Univariate Imputation -Numerical
        2.1 Mean-Median Imputation 
        # This method is used when data is missing at random(MCAR)
            2.1.1 Using Pandas
            2.1.2 Using SciKit learn-Simple imputer (strategy = mean/median)
        2.2 Arbitary Value Imputation 
        #This method is used when data is not missing at random
            2.2.1 Using Pandas
            2.2.2 Using SciKit learn-Simple imputer(strategy = constant)
        2.3. End of distribution Imputation
            2.3.1 Fill value by mean + 3*sigma or mean- 3*sigma if data is normally distributed
            2.3.2 Fill value by Q1-1.5*IQR or  Q3+1.5*IQR id distribution is skewed
    3. Univariate Imputation -Categorical
        3.1 Frequent value imputation (mode)
        3.2 Missing catergory imputation (strategy='constant',fill_value='Missing')
    4. Random Value Imputation
    5. Missing Value Indicator
    6. KNN imputer
    7. MICE-Multivariate Imputation By Chained Equations algorithm
# Since we have limited data we cannot drop the rows.
# data is missing at random hence we are not using Arbitary value imputation,Random Value Imputation

# We have experimented with below imputation method
    - KNN imputation
    - Sci Kit learn Simple Imputer(strategy=mean/median/constant)
    - MICE

- `Encoding`: 1. Ordinal encoding: Assign an ordered integer value to each category, reflecting the relative order or importance of the categories.
              2. One-hot encoding: Represent each unique category as a binary variable, with a value of 1 indicating the presence of the category 
                                  and a value 0 indicating its absence.
              3. Using Pandas -getdummies(one hot encoding)

- `Outliers handling` : 1. Z score method -Identify outliers as values that are a specified number of standard deviations away from the mean.
                                            # when data is normally distributed
                        2. Trimming
                        3. Capping
                        4. IQR (Interquartile Range) method: Identify outliers as values that are below the first quartile - 1.5 times
                                              the interquartile range  or above the third quartile + 1.5 times the interquartile range.
                        5. winsorization -Replacing the outliers with a specified value, such as the minimum or maximum value in the dataset, 
                                          or with the value at a specified percentile.
# Most of the features are not normally distributed, Z score method, trimming, Capping not useful.
# We are using IQR method for outlier handling.

- `Scaling`: 1. Min-Max Scaling: Scale the values of the variables between 0 and 1.
             2. Standardization(Z-score normalization): Scale the values of the variables to have a mean of zero and a standard deviation of one.

             3. Max Absolute Scaling: Scale the values of the variables between -1 and 1.

             4. Robust Scaling: Scale the values of the variables using the median and interquartile range to make the scaling robust to outliers.

             5. Normalization: Scale the values of the variables to have a sum of one.

# Since most of the independent variables are not normally distributed we cannot use Standardscaler
# because most of the feature has outliers. So Minmax will scale data according to Max values which is outlier.
# Robust Scaler removes the median and scales the data according to the quantile range (defaults to IQR: Interquartile Range).The IQR is the range between the 1st quartile (25th quantile) and the 3rd quartile (75th quantile).

- `Dimenssionality Reduction`- Reducing the dimensionality of the data using techniques like PCA or feature selection.

- Pipeline: Pipeline class in scikit-learn is a useful tool for creating a series of steps for transforming and applying machine learning algorithms to a dataset. It simplifies the process of testing and tuning various components of a machine learning system, and provides a convenient and consistent way of applying the same steps to both training and test data.

```

## Model building and evaluation
```bash
- We have experimented using various missing value imputation methods on below ML algorithms.
- 1. Linear Regression
- 2. Lasso Regression
- 3. Ridge Regression
- 4. DT Regressor
- 5. Random Forest Regressor
- 6. Gradient Boosting Regressor
- 7. Adaboost Regressor
- 8. XGB Regressor
- 9. K-Neighbors Regressor
- 10. SVR
- 11. PCA

- We have defined two functions 
    1. evaluate_reg    
        - This function takes in true values and predicted values and returns r2_score ,RMSE
    2. evaluate_models
        - This function takes in X and y and models dictionary as input.
        - It splits the data into Train Test split
        - Iterates through the given model dictionary and evaluates the metrics
        - Returns: Dataframe which contains report of all models metrics with root mean sqaured error.

- Result:
- Experiment 1: KNN imputer
    1. Gradient Boosting - 0.857
    2. Random Forest     - 0.852(overfitting)
    3. XGB               - 0.837(overfitting)

- Experiment 2: Simple Imputer(strategy=median)
    1. XGB               - 0.877(overfitting)
    2. Gradient Boosting - 0.858
    3. Random Forest     - 0.842(overfitting)

- Experiment 3: MICE
    1. Gradient Boosting - 0.854
    2. Random Forest     - 0.806(overfitting)
    3. Ridge Regression  - 0.783

- Experiment 4: Simple Imputer(strategy=constant)
    1. Gradient Boosting - 0.864
    2. Random Forest     - 0.833(overfitting)
    3. XGB               - 0.832(overfitting)

- Experiment 5: Simple Imputer(strategy=mean)
    1. XGB               - 0.861(overfitting)
    2. Gradient Boosting - 0.860
    3. Random Forest     - 0.857(overfitting)
    

- Experiment 6: PCA(stratrgy = median)
    1. XGB               - 0.877(overfitting)
    2. Gradient Boosting - 0.858
    2. Random Forest     - 0.855(overfitting)
    
- In all experiment gradient boosting performing well giving best accuarcy 0.864 in experiment 5.

```
## Hyperparameter tuning
```bash
- Result:
- Experiment 1: KNN imputer
    1. Gradient Boosting - 0.913
    2. Random Forest     - 0.893
    3. XGB               - 0.867

- Experiment 2: Simple Imputer(strategy=median)
    1. Gradient Boosting - 0.884
    2. XGB               - 0.848 
    3. Random Forest     - 0.844

- Experiment 3: MICE
    1. Gradient Boosting - 0.888
    2. Random Forest     - 0.876
    3. Ridge Regression  - 0.80

- Experiment 4: Simple Imputer(strategy=constant)
    1. Random Forest     - 0.889
    2. Gradient Boosting - 0.875
    3. XGB               - 0.838

- Experiment 5: Simple Imputer(strategy=mean)
    1. Random Forest     - 0.899
    2. Gradient Boosting - 0.890 
    3. XGB               - 0.839

- Experiment 6: PCA(stratrgy = median)
    1. Gradient Boosting - 0.884
    2. XGB               - 0.848
    3. Random Forest     - 0.844

- After hyperparameter tuning overfitting issue get resolved.


```
### We get best accuracy 91.3 using Gradient Boosting 

