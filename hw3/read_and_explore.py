# Building an ML Pipeline, CAPP 30254
# 
# Scripts to load, explore, clean, and split the data
# 
# Ratul Esrar


import os
import csv
import random
import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')
from sklearn.preprocessing import Imputer
from sklearn.cross_validation import train_test_split


def read_data(f_name):
    '''
    Using os.path.splitext(file_name)[-1].lower(), find the extension of f_name and then read into pandas dataframe.
    Found here for reference: 
        http://stackoverflow.com/questions/5899497/checking-file-extension

    Input: 
        f_name (str)

    Output: 
        df (pandas dataframe)
    '''
    ext = os.path.splitext(f_name)[-1].lower()  
    if ext == ".csv":
        df = pd.read_csv(f_name)
    elif ext == ".xls" or ext == ".xlsx":
        df = pd.read_excel(f_name)
        
    return df


def check_missing_vals(df, cols_to_exclude = []):
    '''
    Check the proportion of missing values from columns in dataframe. Adapted from lab 4: https://github.com/dssg/MLforPublicPolicy/blob/master/labs/lab4-solutions.ipynb.

    Inputs:
        df (pandas dataframe)
        cols_to_exclude (list)

    Output:
        print of the percent of values missing for each column
    '''
    for col in df.columns:
        if col in cols_to_exclude:
            continue
        elif df[col].count() < len(df):
            missing_perc = ((len(df) - df[col].count()) / float(len(df))) * 100.0
            print('%.1f%% missing from: Column %s' % (missing_perc, col))


def describe_data(df, cols_to_exclude = []):
    '''
    Takes a df and prints descriptive stats for columns unless specified to be excluded. 
    For example, the summary stats for PersonID or zipcode are not meaningful in this display.

    Inputs:
        df (pandas dataframe)
        cols_to_exclude (list)
    
    Output:
        tables
        col_list (list)
    '''
    col_list = []
    for col in df:
        col_list.append(col)
        if str(col) in cols_to_exclude:
            pass
        else:
            print(df[str(col)].describe().to_frame(), '\n')
            
    return col_list


def make_box_hist_plots(df, col_name):
    '''
    Make histogram and boxplot of a given column in the data frame.

    Inputs:
        df (pandas dataframe)
        col_name (str)

    Ouput:
        plot (matlplotlib object)
    '''
    colors = ["red", "blue", "green", "yellow", "purple", "black"]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    df.boxplot(column = col_name, ax = ax1, vert = False)
    ax1.set_title(col_name + " Boxplot")
    ax1.set_yticklabels([col_name], rotation = 90)
    
    df[col_name].plot(kind = "hist", alpha = 0.2, bins = 20, color = random.choice(colors), ax = ax2)
    ax2.set_title(col_name)
    
    plt.tight_layout()

    
def plot_df_summaries(df, col_list, cols_to_exclude = []):
    '''
    Iterate through the dataframe for specified columns and make box plots and histogram for them.
    
    Inputs:
        df (pandas dataframe)
        col_list (list of columns to describe)
        cols_to_exclude (list)
    
    Output:
        make_box_hist_plots (matplotlib objects)
    '''
    for col in col_list:
        if str(col) in cols_to_exclude:
            pass
        else:
            make_box_hist_plots(df, col)


def make_correlation_matrix(df):
    '''
    Takes a dataframe and plots the correlation matrix between all the features.

    Input: 
        df (pandas dataframe)

    Output:
        plot of correlation matrix
    '''
    fig, ax = plt.subplots(figsize = (10, 8))
    cor = df.corr()
    sn.heatmap(cor, mask = np.zeros_like(cor, dtype = np.bool), cmap = sn.diverging_palette(220, 10, as_cmap = True),
                square = True, ax = ax)
    ax.set_title('Data Correlation Matrix')
    plt.tight_layout()


def discretize(df, col, sep_val):
    '''
    Takes a column that is a categorical variable and turns it into a binary value.

    Inputs:
        df (pandas dataframe)
        col (string)
        sep_val (int)

    Output:
        df_copy
    '''
    df_copy = df.copy(deep = True)
    df_copy[col] = (df_copy[col] >= sep_val).astype(int)
    
    return df_copy


def binarize_cats(df, col_list, drop = True):
    '''
    Takes a df and a list of columns to binarize. Choose to drop the not-modified column by default. Adapted from lab4.

    Inputs:
        df (pandas dataframe)
        col_list (list) 
        drop (bool) 
    
    Output:
        df (modified dataframe)
    '''
    for col in col_list:
        binary_cols = pd.get_dummies(df[col], col)
        df = pd.merge(df, binary_cols, left_index = True, right_index = True, how = 'inner')
    if drop:
        df.drop(cat_cols, inplace = True, axis = 1)
    
    return df


def split_data(df, y_col, size = 0.2):
    '''
    Take a dataframe and split into training and testing sets.

    Inputs:
        df (pandas dataframe)
        y_col (string)
        size (float)

    Output:
        X_train, X_test, y_train, y_test
    '''
    X = df.drop(y_col, axis = 1)
    y = df[y_col]
    
    return train_test_split(X, y, test_size = size, random_state = 3)


def imputer(df, method = 'median'):
    '''
    Function to complete missing values by imputation on the median by default. Other strategies can be used as well.

    Input:
        df (pandas dataframe)

    Output:
        df (modified dataframe) 
    '''
    imputer = Imputer(missing_values = 'NaN', strategy = method, axis = 0)
    imputer = imputer.fit(df)
    data = imputer.transform(df)
    
    return pd.DataFrame(data).reset_index(drop = True)

