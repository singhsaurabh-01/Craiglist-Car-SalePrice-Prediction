#!/usr/bin/env python
# coding: utf-8

# # Predicting the Pre-owned car price

# 1. Problem Statement
# * How well we predict car price
# 
# 
# 2. Data
# * The data is downloaded from Keggle - Craiglist car dataset(https://www.kaggle.com/alessiocozzi/craiglist-car-dataset)
# 
# 
# 3. Evaluation
# * The evaluation metric for this project will be all possible regression Metric accross different Regression models. The goal for Regression model is to minimize the error.
# 
# 
# 4. Features
# 
# 

# In[1]:


# Import all relevent packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn

df_vehicle = pd.read_csv("vehicles.csv")


# ### Data undestanding
df_vehicle.columns
df_vehicle.title_status.value_counts()
#len(df_vehicle)
#df_vehicle.odometer.mean()
#df_vehicle.columns
#df_vehicle.dtypes
#df_vehicle.drive.value_counts()
#df_vehicle.isna().sum()
df_vehicle.describe()

# ### Data exploration

df = df_vehicle.copy()

df.drop(df["odometer"].idxmax(), inplace=True)

#Remove outlier
df.drop(df[df["odometer"] >= 400000].index, inplace=True)

plt.figure(figsize=(15,6))
plt.scatter(df.paint_color.astype(str),df.odometer);

# Group by region based on size column
df.groupby(["region","title_status"], dropna=False)["title_status"].count()

df.paint_color.value_counts()

plt.figure(figsize=(12,6))
plt.hist(df.paint_color, color="black", stacked=1)

df.head().T

df.isna().sum()

df[df.condition.isnull() == False].isna().sum()

df.state.unique()

df_clean = df_vehicle.copy()

df_clean.dtypes
len(df_clean)
#df_clean.isna().sum()


# #### --------------------------------------------------------------------------------------------------------------------------------------------------------------
# 
# #### Issue here is the Null values in the categorical features
#     * Condition
#     * Cylinder
#     * Title_status
# 
# #### Also the variables which may be indepoendent but can influence the price of car
#     * Paint_color
#     * Drive
#     
# #### To do that we will be building classification model foreach of the column seperatly and to do that we will try below algorithms
#     * Random Forest Classification
#     * K nearest Neighour
#    > Logistic Regression - We will not use this since it cannot be applied directly on to multiclass classification
# #### -------------------------------------------------------------------------------------------------------------------------------------------------------------
# 
# ### Before that let's find what feature is imporant to us and accordingly we will create classification model

#Deleteting first set of unnecessary columns from Data
df_clean.drop(["size","county","id","url","region_url", "vin","image_url","description","state","region"], axis=1, inplace=True)

# Finding outlier and deleteting the from dataset
for header, data in df_clean.items():
    if pd.api.types.is_numeric_dtype(data) and (header!="long" or header != "lat"):
        q3 = df_clean[header].quantile(q=.75)
        q1 = df_clean[header].quantile(.25)

        iqr = q3 - q1

        no_outlier = ((q1-1.5 * iqr) < df_clean[header]) & (df_clean[header] < (q3+1.5 * iqr))
        outlier = ((q1-1.5 * iqr) > df_clean[header]) | (df_clean[header] > (q3+1.5 * iqr))
        df_clean.drop(df_clean[((q1-1.5 * iqr) > df_clean[header]) | (df_clean[header] > (q3+1.5 * iqr))].index, inplace=True)

#Fill null odometer reading with Median values of odometer
df_clean.odometer.fillna(df_clean["odometer"].median(), inplace=True)

df_clean.dropna(inplace=True)


df_clean.isna().sum()

# Clean 'model' column and just take car name in place of every moidel number

model_name = pd.DataFrame({'Model_name' : df_clean.model.str.partition(sep=' ',expand=True)[0]})
df_clean = pd.concat([df_clean, model_name], axis=1)
df_clean.drop("model", axis=1, inplace=True)


# ## Converting Categorical values into numerical values by using category.codes


for header, content in df_clean.items():
    if pd.api.types.is_object_dtype(content):
        df_clean[header] = df_clean[header].astype('category')
        df_clean[header] = pd.Categorical(content).codes
    elif pd.api.types.is_categorical_dtype(content):
        df_clean[header] = pd.Categorical(content).codes      


# In[315]:


# from sklearn.preprocessing import OneHotEncoder
# from sklearn.compose import ColumnTransformer

# x = df_clean.drop("price", axis=1)
# y = df_clean["price"]

# one_hot = OneHotEncoder()
# categorical_features = ['manufacturer', 'model', 'condition', 'cylinders', 'fuel', 'title_status', 'transmission',
#         'drive', 'type', 'paint_color']

# transformer = ColumnTransformer([("one_hot", one_hot, categorical_features)], remainder="passthrough")

# transformed_x = transformer.fit_transform(x)

# pd.DataFrame(transformed_x,).head()
#----------------------------------------------------------------------------------------
# x_dummies = pd.get_dummies(x_df_no_na[['region', 'manufacturer', 'Model_name', 'condition', 'cylinders', 'fuel', 'title_status', 'transmission',
#         'drive', 'type', 'paint_color', 'state']])

# x_temp = x_df_no_na.drop(['region', 'manufacturer', 'Model_name', 'condition', 'cylinders', 'fuel', 'title_status', 'transmission',
#         'drive', 'type', 'paint_color', 'state'], axis=1)


# ## Feature importance calculation
# 
#     > We will use Recusrsive feature elimination(RFE) to find the feature importance of df_clean dataframe

np.random.seed(42)
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor
from sklearn import svm
from sklearn.model_selection import train_test_split

x = df_clean.drop("price", axis=1)
y = df_clean["price"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = .30)

# # Add all different model I want to work with.
# models = {'SVM': svm.SVR(),
#          'RandomForestRegressor' : RandomForestRegressor()}


np.random.seed(42)
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor
from sklearn import svm
#from sklearn.linear_model import r
from sklearn.model_selection import train_test_split

x = df_clean.drop("price", axis=1)
y = df_clean["price"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = .30)

# Add all different model I want to work with.
models = {'RandomForestRegressor' : RandomForestRegressor()}


def fit_and_score(x_train, y_train, x_test, y_test):
    """
    We will fit different model here using RFE and print the each step of RFE with score
    """
    model_score = {}
    #model_step = {}
    #for i in range(5, len(x_train.columns)):
    for model_name, model in models.items():
        estimator = RFE(model, n_features_to_select=10, step=1, verbose=1)
        estimator.fit(x_train, y_train)
        model_score[model_name] = model.score(x_test, y_test)
        #model_step[model_name] = i
    return model_score
#, model_step

# selector = RFE(RandomForestRegressor(), n_features_to_select=10, step=1, verbose=1)

# selector = selector.fit(x_train, y_train)

#Call the fitandscore method
model_score = fit_and_score(x_train, y_train, x_test, y_test)

selector.score(x_train, y_train)

selector.support_

y_preds = selector.predict(x_test)

df_clean.model.head(10)

# Import evaluation metrics and other packages:

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import mean_gamma_deviance
from sklearn.metrics import median_absolute_error
from sklearn.metrics import mean_poisson_deviance
from sklearn.metrics import mean_gamma_deviance
from sklearn.metrics import max_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

# Clean 'model' column to reduce dummies/encoding values

#model_name = pd.DataFrame({'Model_name' : df_no_na.model.str.partition(sep=' ',expand=True)[0]})
#df_no_na = pd.concat([df_no_na, model_name], axis=1)
#df_no_na.drop("model", axis=1, inplace=True)
#
## Before fcrating the model convert the categorical values to integer values
#np.random.seed(42)
#x_df_no_na = df_no_na.drop("price", axis=1)
#y_df_no_na = df_no_na["price"]

#from sklearn.preprocessing import OneHotEncoder
#from sklearn.compose import ColumnTransformer

# one_hot = OneHotEncoder()
# categorical_features = ['region', 'manufacturer', 'Model_name', 'condition', 'cylinders', 'fuel', 'title_status', 'transmission',
#         'drive', 'type', 'paint_color', 'state']

# transformer = ColumnTransformer([("one_hot", one_hot, categorical_features)], remainder="passthrough")

# transformed_x = transformer.fit_transform(x_df_no_na)

# pd.DataFrame(transformed_x).head()
#----------------------------------------------------------------------------------------
#x_dummies = pd.get_dummies(x_df_no_na[['region', 'manufacturer', 'Model_name', 'condition', 'cylinders', 'fuel', 'title_status', 'transmission',
#        'drive', 'type', 'paint_color', 'state']])
#
#x_temp = x_df_no_na.drop(['region', 'manufacturer', 'Model_name', 'condition', 'cylinders', 'fuel', 'title_status', 'transmission',
#        'drive', 'type', 'paint_color', 'state'], axis=1)
#
#
## In[60]:
#
#
#transformed_x=pd.concat([x_temp,x_dummies],axis=1)
#x_train, x_test, y_train, y_test = train_test_split(transformed_x, y_df_no_na, test_size = .25)
#
#model = RandomForestRegressor(max_samples=10000,n_jobs=-1, n_estimators=20)
#model.fit(x_train, y_train)
#model.score(x_train, y_train)
#
#
#y_preds = model.predict(x_test)


# In[62]:


# Create evaluation metrics fuction

def evaluate(y_test, y_preds):
    """
    All metrics to evaluate Regression model
    """
    print("The R2 value is : {}".format(r2_score(y_test, y_preds)))
    
    print("Mean Absolute error value is : {}".format(mean_absolute_error(y_test, y_preds)))
    
    print("Mean Squared Error value is : {}".format(mean_squared_error(y_test, y_preds)))
    
    print("Root Mean Squared error value is : {}".format(mean_squared_error(y_test, y_preds, squared=False)))
    
    print("Max Error value is : {}".format(max_error(y_test, y_preds)))
    
    #print("Mean gamma Deviance value is {}".format(mean_gamma_deviance(y_test, y_preds)))
    
    #print("Mean Poisson Deviance value is {}".format(mean_poisson_deviance(y_test, y_preds)))

    print("Median absolute error value is : {}".format(median_absolute_error(y_test, y_preds)))
    
    print("mean Squared log error value is : {}".format(mean_squared_log_error(y_test, y_preds)))

