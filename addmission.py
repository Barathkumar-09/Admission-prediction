#import libraries 

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from jupyterthemes import jtplot
jtplot.style(theme='monokai', context='notebook', ticks=True, grid=False)


# read the csv file 
admission_df=pd.read_csv('Admission_Predict.csv')
# Let's drop the serial no
admission_df.drop('Serial No.',axis =1,implace =true)
admission_df

task 3:

# checking the null 
admission_df.isnull().sum()
# Check the dataframe information
admission_df.info()
# Statistical summary of the dataframe
adminssion_df.describe()
# Grouping by University
df_university = admission_df.group(by = 'University Rating' )



task4
•	admission_df.hist(bins = 30,figsize =(20,20),color ='r')
•	sns.pairplot(admission_df)
•	corr_matrix=admission_df.corr()
plt.figure(figsize=(12,12))
sns.heatmap(corr_matrix,annot =True)
plt.show()
