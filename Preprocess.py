# Course Code: IT8104
# Course Name: Data Analysis
# Assignment: 1 - Scenario-based analytical report using data analysis tools and techniques
# 
# File: Preprocess.py
# Author: Gautam Sarup
# Date: 09/26/2023
# 

import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# load the raw data into a data frame
#
raw_df = pd.read_csv("Data\\train_raw.csv")
print("Raw dataset rows: " + str(len(raw_df)))

# remove duplicate rows. duplicate rows have the same value for:
# user id, product id, product category 1, product category 2, product category 3 and purchase
#
# this is a precautionary step because we have only one dataset to process and we know it does not
# contain duplicates.
#
cooked_df = raw_df.drop_duplicates(
    subset = ["User_ID", "Product_ID", "Product_Category_1", "Product_Category_2", "Product_Category_3", "Purchase"],
    keep = 'last')
print ("Cooked dataset with duplicates removed: " + str(len(cooked_df)) + " rows")

# drop the raw dataframe because all further processing will be done only on the cooked data.
#
del [[raw_df]]

# This code is for the Excel preprocessed data in which all categorical variables are already mapped.
#
# # group the data so that for each user ID and Product Category[1..3] there is a single row.
# operations = {
#             "User_ID":"first",
#             "Age_Ordinal":"first",
#             "Gender_Ordinal":"first",
#             "Occupation":"first",
#             "City_Category_Ordinal":"first",
#             "Marital_Status":"first",
#             "Product_Category_1":"first",
#             "Purchase":"sum",
#         }
# cooked_df = cooked_df.groupby(["User_ID", "Product_Category_1"]).aggregate(operations)

# print("Here are the top 5 groupd rows...")
# print(cooked_df.head(5))

operations = {
        "User_ID":"first",
        "Product_ID":"first",
        "Age":"first",
        "Occupation":"first",
        "City_Category":"first",
        "Marital_Status":"first",
        "Purchase":"sum"
    }

cooked_df = cooked_df.groupby(["User_ID", "Product_ID"]).aggregate(operations)

do_save = input("Save data frame as cooked_train.csv (Y/y): ")
if do_save == 'Y' or do_save == 'y':
    cooked_df.to_csv("Data\\cooked_train.csv", header="True", index="False")