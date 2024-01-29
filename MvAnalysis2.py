# Course Code: IT8104
# Course Name: Data Analysis
# Assignment: 1 - Scenario-based analytical report using data analysis tools and techniques
# 
# File: MVAnalysis2.py
# Author: Gautam Sarup
# Date: Sept 5, 2023
# 

# from http.cookies import CookieError
# from imp import cache_from_source
# from re import X
from importlib.metadata import packages_distributions
from statistics import LinearRegression
from typing import Self
from enum import Enum
from os import remove
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xlwings

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn import linear_model
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error, mean_absolute_error

class Model(Enum):
    LINEAR = 0
    LINEAR_B = 1
    DECISION_TREE = 2
    RIDGE = 3

# load the raw data into a data frame
def load_data():
    return pd.read_csv("Data\\train_raw.csv")

def load_preprocessed_data():
    return pd.read_csv("Data\\train_preprocessed.csv")

# remove duplicate rows. duplicate rows have the same value for:
# user id, product id, product category 1, product category 2, product category 3 and purchase
#
# this is a precautionary step because we have only one dataset to process and we know it does not
# contain duplicates.
#
def remove_duplicates(raw_df):
    return raw_df.drop_duplicates(
        subset = ["User_ID", "Product_ID", "Product_Category_1", "Product_Category_2", "Product_Category_3", "Purchase"],
        keep = 'last')

# group the data by user ID and Product_Category_1
def group_data(cooked_df):
    operations = {
            "Age":"first",
            "Gender":"first",
            "Occupation":"first",
            "City_Category":"first",
            "Stay_In_Current_City_Years":"first",
            "Marital_Status":"first",
            "Purchase":"sum",
        }
    
    return cooked_df.groupby(["User_ID", "Product_Category_1"]).aggregate(operations)

# encode gender F -> 1, M -> 0
#
def encode_gender(cooked_df):
    cooked_df.replace(['M', 'F'], [0, 1], inplace=True)
    return cooked_df

# encode residence years 4+ -> 4
#
def encode_residence_years(cooked_df):
    cooked_df.replace(["4+"], 4, inplace=True)
    return cooked_df

def encode_column(cooked_df, col_name):
    encoder = OneHotEncoder(handle_unknown='ignore')
    encoded_df = pd.DataFrame(encoder.fit_transform(cooked_df[[col_name]]).toarray())
    
    encoded_df.columns = encoder.get_feature_names_out([col_name])
    pos = cooked_df.columns.get_loc(col_name)
    for col in encoded_df.columns:
        cooked_df[[col]] = encoded_df[[col]].to_numpy()
        
    cooked_df.drop([col_name], axis = 1, inplace = True)
    
    return cooked_df

def auto_cat_encode(cooked_df):
    encode_column(cooked_df, "City_Category")
    encode_column(cooked_df, "Age")
    encode_column(cooked_df, "Occupation")

    return cooked_df

def encode(cooked_df):
    cooked_df = encode_gender(encode_residence_years(cooked_df))
    cooked_df = auto_cat_encode(cooked_df)
    
    return cooked_df

def preview_cooked_data(cooked_df):
    debug = False
    if debug:
        xlwings.view(cooked_df)

def remove_outliers(cooked_df):
    # sns.boxplot(cooked_df['Purchase'])
    # cutoff = input("Enter cutoff value: ")
    # cutoff = int(cutoff)
    cutoff=20000
    
    deletable_df = cooked_df[cooked_df["Purchase"] > cutoff].index
    cooked_df.drop(index = deletable_df, inplace = True)

def export_cooked_data(cooked_df):
    export = True
    if export:
        cooked_df.to_csv("Data\\train_preprocessed.csv")

def Predict(df, model_type):
        X = df[["Product_Category_1", "Gender", "Stay_In_Current_City_Years", "Marital_Status",
                "City_Category_A", "City_Category_B", "City_Category_C",
                 "Age_0-17", "Age_18-25", "Age_26-35", "Age_36-45", "Age_46-50", "Age_51-55", "Age_55+",
                 "Occupation_0", "Occupation_1", "Occupation_2", "Occupation_3", "Occupation_4",
                 "Occupation_5", "Occupation_6", "Occupation_7", "Occupation_8", "Occupation_9",
                 "Occupation_10", "Occupation_11", "Occupation_12", "Occupation_13", "Occupation_14",
                 "Occupation_15", "Occupation_16", "Occupation_17", "Occupation_18", "Occupation_19",
                 "Occupation_20"]]
        y = df["Purchase"]
 
        match model_type:
            case Model.LINEAR:
                model_id = "Linear"
                regr = linear_model.LinearRegression(fit_intercept = True, copy_X = True)
                regr.fit(X.values, y.values)
            case Model.LINEAR_B:
                model_id ="linear_b"
                LinearRegression().fit(X.value, y.values)
            case Model.DECISION_TREE:
                model_id = "Preferred model: Decision tree regressor"
                regr = DecisionTreeRegressor(max_depth = 35)
                regr.fit(X.values, y.values)
            case Model.RIDGE:
                model_id = "Ridge"
                regr = linear_model.Ridge(fit_intercept = True, copy_X = True)
                regr.fit(X.values, y.values)
            case _:
                print("unknown model")
                
        known_purchases = []
        predicted_purchases = []
        
        rows = len(df.index)

        print("Running model: " + model_id)
        
        for idx in df.index:
            if idx == 0:
                continue
            
            user_id = df["User_ID"][idx]
            prod_cat_1 = df["Product_Category_1"][idx]
            gender = df["Gender"][idx]
            stay = df["Stay_In_Current_City_Years"][idx]
            marital_status = df["Marital_Status"][idx]
            city_cat_a = df["City_Category_A"][idx]
            city_cat_b = df["City_Category_B"][idx]
            city_cat_c = df["City_Category_C"][idx]
            age_0 = df["Age_0-17"][idx]
            age_18 = df["Age_18-25"][idx]
            age_26 = df["Age_26-35"][idx]
            age_36 = df["Age_36-45"][idx]
            age_46 = df["Age_46-50"][idx]
            age_51 = df["Age_51-55"][idx]
            age_55 = df["Age_55+"][idx]
            occupation_0 = df["Occupation_0"][idx]
            occupation_1 = df["Occupation_1"][idx]
            occupation_2 = df["Occupation_2"][idx]
            occupation_3 = df["Occupation_3"][idx]
            occupation_4 = df["Occupation_4"][idx]
            occupation_5 = df["Occupation_5"][idx]
            occupation_6 = df["Occupation_6"][idx]
            occupation_7 = df["Occupation_7"][idx]
            occupation_8 = df["Occupation_8"][idx]
            occupation_9 = df["Occupation_9"][idx]
            occupation_10 = df["Occupation_10"][idx]
            occupation_11 = df["Occupation_11"][idx]
            occupation_12 = df["Occupation_12"][idx]
            occupation_13 = df["Occupation_13"][idx]
            occupation_14 = df["Occupation_14"][idx]
            occupation_15 = df["Occupation_15"][idx]
            occupation_16 = df["Occupation_16"][idx]
            occupation_17 = df["Occupation_17"][idx]
            occupation_18 = df["Occupation_18"][idx]
            occupation_19 = df["Occupation_19"][idx]
            occupation_20 = df["Occupation_20"][idx]
            
            known_purchase = df["Purchase"][idx]
            
            predicted_purchase = regr.predict([[
                prod_cat_1, gender, stay, marital_status, city_cat_a, city_cat_b, city_cat_c,
                age_0, age_18, age_26, age_36, age_46, age_51, age_55,
                occupation_0, occupation_1, occupation_2, occupation_3, occupation_4, occupation_5,
                occupation_6, occupation_7, occupation_8, occupation_9, occupation_10, occupation_11,
                occupation_12, occupation_13, occupation_14, occupation_15, occupation_16, occupation_17,
                occupation_18, occupation_19, occupation_20
             ]])
            
            known_purchases.append(known_purchase)
            predicted_purchases.append(predicted_purchase[0])

        return known_purchases, predicted_purchases

def calc_metrics(known_purchases, predicted_purchases):
    r2 = r2_score(known_purchases, predicted_purchases)
    mse = mean_squared_error(known_purchases, predicted_purchases)
    mae = mean_absolute_error(known_purchases, predicted_purchases)
    
    return r2, mse, mae

def __main__():
    raw_df = load_data()
    print("Raw dataset rows: " + str(len(raw_df)))

    cooked_df = remove_duplicates(raw_df)
    print ("Cooked dataset with duplicates removed: " + str(len(cooked_df)))
    
    remove_outliers(cooked_df)
    
    cooked_df = group_data(cooked_df)
    print ("Cooked dataset grouped by User and Product: " + str(len(cooked_df)))    
    
    cooked_df = encode(cooked_df)
    print("Data encoded")
    
    preview_cooked_data(cooked_df)
    export_cooked_data(cooked_df)
    
    # done preprocessing. load the final data.
    preprocessed_data = load_preprocessed_data()
    
    print("Running model: Decision Tree")
    known_purchases, predicted_purchases = Predict(preprocessed_data, Model.DECISION_TREE)
    r2, mse, mae = calc_metrics(known_purchases, predicted_purchases)
    print("The R2 for this model is : " + str(r2))
    print("The Mean Squared Error for this model is : " + str(mse))
    print("The Mean Absolute Error for this model is : " + str(mae))
    print("")
    
    print("Running model: Linear")
    known_purchases, predicted_purchases = Predict(preprocessed_data, Model.LINEAR)
    r2, mse, mae = calc_metrics(known_purchases, predicted_purchases)
    print("The R2 for this model is : " + str(r2))
    print("The Mean Squared Error for this model is : " + str(mse))
    print("The Mean Absolute Error for this model is : " + str(mae))
    print("")
    
    print("Running model: LinearB")
    known_purchases, predicted_purchases = Predict(preprocessed_data, Model.LINEAR_B)
    r2, mse, mae = calc_metrics(known_purchases, predicted_purchases)
    print("The R2 for this model is : " + str(r2))
    print("The Mean Squared Error for this model is : " + str(mse))
    print("The Mean Absolute Error for this model is : " + str(mae))
    print("")
    
    print("Running model: Ridge")
    known_purchases, predicted_purchases = Predict(preprocessed_data, Model.RIDGE)
    r2, mse, mae = calc_metrics(known_purchases, predicted_purchases)
    print("The R2 for this model is : " + str(r2))
    print("The Mean Squared Error for this model is : " + str(mse))
    print("The Mean Absolute Error for this model is : " + str(mae))
    print("")

__main__()