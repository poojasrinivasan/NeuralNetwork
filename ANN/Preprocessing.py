import pandas as pd
import numpy as np
import sys
def check_Numeric(attributecolumn):
   numeric_columns=[]
   for data in attributecolumn:
     try:
         val = float(data)
     except ValueError:
         return False
   return True

inputdatapath=sys.argv[1]
preprocceseddatapath=sys.argv[2]
inputdata=pd.read_csv(inputdatapath,header=None)
inputdata=inputdata.replace(' ?',np.nan).dropna()
#print(inputdata)
Count_Row=inputdata.shape[0] #gives number of row count
Count_Col=inputdata.shape[1] #gives number of col count
Column_headers=list(inputdata)
#seperating features and class values in a dataframe
instances=inputdata.iloc[:,0:Count_Col-1]
labels=inputdata.iloc[:,[Count_Col-1]]
#identifying distinct values in features and classes
distinct_raw_features=[]
distinct_class_features=[]
for col in instances:
    distinct_raw_features.append(instances[col].unique())
for col in labels:
    distinct_class_features.append(labels[col].unique())
#categorizing each column as numeric or categorical
numeric_columns=[]
categorical_columns=[]
index = 0
df=pd.DataFrame()
for column in distinct_raw_features:
    flag= check_Numeric(column)
    if flag==True:
        numeric_columns.append(index)
        index+=1

    else:
        categorical_columns.append(index)
        index += 1

#encode categorical value to numeric value for all non-numeric columns
stacked = instances[categorical_columns].stack()
vals = stacked.drop_duplicates().values
b = [x for x in stacked.drop_duplicates().rank(method='dense')]
d1 = dict(zip(b, vals))
instances[categorical_columns] = instances[categorical_columns].stack().rank(method='dense').unstack()
df1=instances
#normalizing feature data
df1 = (df1 - df1.mean())/df1.std()
#encode categorical value to numeric value for class labels
labels=labels.stack().rank(method='dense').unstack()
Count_Col=df1.shape[1]#gives number of col count
lables_list=list(labels.values.flatten())
df1[Count_Col] = lables_list
df1.to_csv(preprocceseddatapath,header=False,index=False)





