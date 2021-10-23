import pandas as pd
import matplotlib.pylab as plt

filename = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/Data%20files/auto.csv"

headers = ["symboling","normalized-losses","make","fuel-type","aspiration", "num-of-doors","body-style",
         "drive-wheels","engine-location","wheel-base", "length","width","height","curb-weight","engine-type",
         "num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",
         "peak-rpm","city-mpg","highway-mpg","price"]

df = pd.read_csv(filename, names = headers)
df.head()

import numpy as np

df.replace("?", np.nan, inplace = True)
df.head(5)

missing_data = df.isnull()
missing_data.head(5)

#inspec nan value in dataframe
for column in missing_data.columns:
    print(column)
    print (missing_data[column].value_counts())
    print("")
#replace nan value in numerical columns
avg_norm_loss = df["normalized-losses"].astype("float").mean(axis=0)
df["normalized-losses"].replace(np.nan, avg_norm_loss, inplace=True)

avg_bore=df['bore'].astype('float').mean(axis=0)
df["bore"].replace(np.nan, avg_bore, inplace=True)

avg_stroke=df['stroke'].astype('float').mean(axis=0)
df['stroke'].replace(np.nan, avg_stroke,inplace=True)

avg_horsepower = df['horsepower'].astype('float').mean(axis=0)
df['horsepower'].replace(np.nan, avg_horsepower, inplace=True)

avg_peakrpm=df['peak-rpm'].astype('float').mean(axis=0)
df['peak-rpm'].replace(np.nan, avg_peakrpm, inplace=True)


#replace nan value in categorical columns:

df['num-of-doors'].value_counts()
df["num-of-doors"].replace(np.nan, "four", inplace=True)

df.dropna(subset=["price"], axis=0, inplace=True)

# reset index, because we droped two rows
df.reset_index(drop=True, inplace=True)


#formating data type:

df.dtypes

df[["bore", "stroke"]] = df[["bore", "stroke"]].astype("float")
df[["normalized-losses"]] = df[["normalized-losses"]].astype("int")
df[["price"]] = df[["price"]].astype("float")
df['peak-rpm'] = df['peak-rpm'].astype("float")

df.dtypes


# data_stanard
# change mpg to 100km/h
df['city-L/100km'] = 235/df["city-mpg"]
df['highway-mpg'] = 235/df['highway-mpg']
df.rename({'highway-mpg':'highway-L/100km'},axis=1, inplace=True)
df.drop('city-mpg', axis=1,inplace=True)

#using mean_normalization
df['length'] = df['length']/df['length'].max()
df['width'] = df['width']/df['width'].max()
df['height'] = df['height']/df['height'].max()


# one-hot encoding for categorical veriables:

dummy_variable_1 = pd.get_dummies(df["fuel-type"])
dummy_variable_1.rename(columns={'gas':'fuel-type-gas', 'diesel':'fuel-type-diesel'}, inplace=True)
df = pd.concat([df, dummy_variable_1], axis=1)
df.drop("fuel-type", axis = 1, inplace=True)

dummy2 = pd.get_dummies(df.aspiration)
dummy2.rename({'std':'aspiration-std','turbo':'aspiration-turbo'},axis=1, inplace=True)
df = pd.concat([df,dummy2],axis=1)
#df.drop('aspiration', axis=1, inplace=True)

dummy3 = pd.get_dummies(df['drive-wheels'])
dummy3.rename({'rwd':'drive-rwd','fwd':'drive-fwd','4wd':'drive-4wd'},axis=1, inplace=True)
df = pd.concat([df,dummy3],axis=1)
#df.drop('drive-wheels', axis=1, inplace=True)

# load cleaned_data to folder
df.to_csv('clean_df2.csv')
