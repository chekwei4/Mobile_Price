# Mobile Price Prediction
Mobile price prediction project which aims to examine various feature selection methods and train model to observe accuracy improvement.

# Introduction
Kaggle: https://www.kaggle.com/iabhishekofficial/mobile-price-classification

Bob has started his own mobile company. He wants to give tough fight to big companies like Apple,Samsung etc.

He does not know how to estimate price of mobiles his company creates. In this competitive mobile phone market you cannot simply assume things. To solve this problem he collects sales data of mobile phones of various companies.

Bob wants to find out some relation between features of a mobile phone(eg:- RAM,Internal Memory etc) and its selling price. But he is not so good at Machine Learning. So he needs your help to solve this problem.

In this problem you do not have to predict actual price but a price range indicating how high the price is.

# Objective
To examine different feature selection techniques and observe how they will improve classification accuracy. 

In feature selection, the main aim is to select the features which are highly dependent on the response.

Model used throughout the project will be RandomForestClassier with default hyperparameters only.

# Variance Threshold
All features are non-unary

# Correlation Coefficient

If two variables are correlated, we can predict one from the other. Therefore, if two features are correlated, the model only really needs one of them, as the second one does not add additional information

Per Pearson Correlation check, there's no correlation above 0.85. 

<p align="center">
    <img width=500, height=400, src="./images/cor.png">
</p>

# Chi Square Test (Categorical Features)

Credits: https://towardsdatascience.com/chi-square-test-for-feature-selection-in-machine-learning-206b1f0b8223

Null Hypothesis (H0): Two variables are independent.

Alternate Hypothesis (H1): Two variables are not independent.

If p-value ≥0.05, failed to reject null hypothesis as there is no relationship between target variable and categorical features.

if p_value <0.05, rejects null hypothesis as there will be some relationship between target variable and categorical features

It is observed that all the 6 categorical features have large p_values, hence we fail to reject the null hypothesis: all of them are indepence of the target class, and we will unlikely select them for model training. 

Looking at chi-square statistic scores below in graphical:

<p align="center">
    <img width=500, height=400, src="./images/chi_stats.png">
</p>

Higher the Chi-Square value, the feature is more dependent on the response and it may be selected for model training.

touch_screen looks most desirable to be selected
three_g is worst.

# Summary

