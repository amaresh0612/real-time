
import pandas as pd
import numpy as np
from sklearn import preprocessing
## Memory optimization
# Original code from https://www.kaggle.com/gemartin/load-data-reduce-memory-usage by @gemartin
# Modified to support timestamp type, categorical type
# Modified to add option to use float16
from pandas.api.types import is_datetime64_any_dtype as is_datetime
from pandas.api.types import is_categorical_dtype

le = preprocessing.LabelEncoder()
def reduce_mem_usage(df, use_float16=False):
    """
    Iterate through all the columns of a dataframe and modify the data type to reduce memory usage.        
    """
    
    start_mem = df.memory_usage().sum() / 1024**2
    print("Memory usage of dataframe is {:.2f} MB".format(start_mem))
    
    for col in df.columns:
        if is_datetime(df[col]) or is_categorical_dtype(df[col]):
            continue
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if use_float16 and c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype("category")
    end_mem = df.memory_usage().sum() / 1024**2
    print("Memory usage after optimization is: {:.2f} MB".format(end_mem))
    print("Decreased by {:.1f}%".format(100 * (start_mem - end_mem) / start_mem))
    
    return df

def convToDateTime(df,cols):
    columns= df.columns
    for i,v in enumerate(columns):
        for j,k in enumerate(cols):
            if(v==k and format is not None):
                df[v] = pd.to_datetime(df[v],infer_datetime_format=True)
            elif(v==k):
                df[v] = pd.to_datetime(df[v])
    return df


def convertTOCategorical(df):
    cdf = df.select_dtypes(exclude=["number"]).columns
    ndf = df.select_dtypes(include=["number"]).columns
    if(len(cdf)>0):
        for col in cdf:
            df[col] = df[col].astype('category')
    return df,cdf,ndf

def handleMissingValues(df):
    print("\n*** Handling Missing values by column ***")
    mvcp = pd.DataFrame(round((100 * df.isnull().sum() / len(df)),2))
    mvcp['Columns']= mvcp.index
    mvcp.reset_index(drop=True, inplace=True)
    print(mvcp)
    mvcp = mvcp.rename(columns = {0 : 'Missing'})
    mvcp = mvcp[mvcp['Missing']>50]
    print ("\n***Your selected dataframe has " + str(df.shape[1]) + " columns.***\n"+"*** " + "*** Columns having missing values greater than 50% : " +str(mvcp.shape[0]) +".***")
    colsToRemove=mvcp.Columns
    if(len(colsToRemove)>0):
        print("\n***Removing Following columns from Data Frame: " + colsToRemove.values + " ***")
        df.drop(colsToRemove,axis=1,inplace=True)
        print("\n***Remaining Columns ***")
        print(df.columns)
    print("\n*******------******")
    print("\n***remaining Missing values by Column ***")
    mvcp = pd.DataFrame(round((100 * df.isnull().sum() / len(df)),2))
    mvcp['Columns']= mvcp.index
    mvcp.reset_index(drop=True, inplace=True)
    mvcp = mvcp.rename(columns = {0 : 'Missing'})
    mvcp = mvcp[mvcp['Missing']>0]
    if(len(mvcp)>0):
        print(mvcp.sort_values(by=['Missing'], ascending=False))
        print("\n*******------******")
        print("\n*** Handling Missing values by Rows ***")
        mvrp = pd.DataFrame(round((100 * df.isnull().sum(axis=1) / len(df)),2))
        mvrp['rownum']=mvrp.index
        mvrp = mvrp.rename(columns = {0: 'Missing'})
        mvrp = mvrp[mvrp['Missing']>75]
        mvrp.reset_index(drop=True, inplace=True)
        rowsToRemove = mvrp['rownum']
        if(len(rowsToRemove)>0):
            print ("\n***Your selected dataframe has " + str(df.shape[0]) + " Rows.***\n"+"*** " + " Rows have missing values greater than 75% : " +str(mvrp.shape[0]) +" .***")
            print("\nRemoving Following Rows from Data Frame: ")
            for row in rowsToRemove:
                print(df.iloc[[row]])
                df.drop(row,axis=0,inplace=True)
                print("\n*******------******")
                print("\n***remaining Missing values by Row ***")
        mvrp = pd.DataFrame(round((100 * df.isnull().sum(axis=1) / len(df)),2))
        mvrp['rownum']=mvrp.index
        mvrp = mvrp.rename(columns = {0: 'Missing'})
        mvrp = mvrp[mvrp['Missing']>0]
        mvrp.reset_index(drop=True, inplace=True)
        if(len(mvrp)>0):
            print(mvrp.sort_values(by=['Missing'], ascending=False))
        else:
            print("\nHurray!!!, No rows with missing values left")
    else:
        print("\nHurray!!!, No missing values left")
    print("\n*******------******")
    return df


def imputeMissingValues(df):
    cdf = df.select_dtypes(exclude=["number"]).columns
    ndf = df.select_dtypes(include=["number"]).columns
    if(len(ndf)>0):
        for col in ndf:
            df[col].fillna(df[col].mean(), inplace=True)
    if(len(cdf)>0):
        for col in cdf:
            df[col]=df[col].fillna(df[col].mode().iloc[0])
        df[df.select_dtypes(exclude=["number"]).columns] = df[df.select_dtypes(exclude=["number"]).columns].apply(lambda col: le.fit_transform(col))
    return df
            
    
def handleCategoricalValues(df,columns):
    if(len(columns)>0):
        for col in columns:
            df[col]=le.fit_transform(df[col])
    return df
        
  
