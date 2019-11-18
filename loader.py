import pandas as pd
import numpy as np
import os
import gc
import random
from utilities import reduce_mem_usage,convToDateTime,convertTOCategorical,handleMissingValues,imputeMissingValues,handleCategoricalValues
import time

def loader_fn():
# get Current Working Directory
      path = os.getcwd()
      print(path)
      
      train_path = "data/train.csv"
      test_path = "data/test.csv"
      weather_train_path = "data/weather_train.csv"
      weather_test_path = "data/weather_test.csv"
      building_path= "data/building_metadata.csv"
      
      columns_datetime=[]
      columns_categorical=[]
      target_variable=["Loan_Status"]
      columns_categorical_enc=[]
      
      
      print("Read Train Data")
      start = time.process_time()
      train_df=pd.read_csv(train_path)
      train_y = train_df[target_variable]
      train_df = train_df.drop(columns=target_variable, axis=1)
      print(time.process_time() - start)
      print(train_df.head(3))
      
      print("Read Test Data")
      start = time.process_time()
      test_df=pd.read_csv(test_path)
      print(time.process_time() - start)
      print(test_df.head(3))
      
      train_objs_num = len(train_df)
      
      all_df = pd.concat(objs=[train_df, test_df], axis=0)
      del [[train_df,test_df]]
      gc.collect()
      print(all_df.head(3))
      
      print("reduce_mem_usage")
      start = time.process_time()
      all_df=reduce_mem_usage(all_df)
      print(time.process_time() - start)
      
      print("convToDateTime")
      start = time.process_time()
      all_df=convToDateTime(all_df,columns_datetime)
      print(time.process_time() - start)
      
      print("convertTOCategorical")
      start = time.process_time()
      all_df,cat_cols,num_cols=convertTOCategorical(all_df)
      print(time.process_time() - start)
      
      print("handleMissingValues")
      start = time.process_time()
      all_df=handleMissingValues(all_df)
      print(time.process_time() - start)
      
      
      print("imputeMissingValues")
      start = time.process_time()
      all_df=imputeMissingValues(all_df)
      print(time.process_time() - start)
      
      print("handleCategoricalValues")
      start = time.process_time()
      all_df=handleCategoricalValues(all_df,cat_cols)
      print(time.process_time() - start)
      
      print(all_df.head(3))
      print(all_df.isnull().sum())
      
      train_X = all_df[:train_objs_num]
      test_X = all_df[train_objs_num:]
      train_df= pd.concat([train_X,train_y],axis=1)
      train_X.to_csv('data/train_X.csv',index=False)
      test_X.to_csv('data/test_X.csv',index=False)
      all_df.to_csv('data/all_cleaned_X.csv',index=False)
      train_y.to_csv('data/train_y.csv',index=False)
      del [[train_y,all_df,train_X,test_X]]
      gc.collect()
