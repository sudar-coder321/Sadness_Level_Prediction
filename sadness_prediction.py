import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from scipy.stats import ttest_1samp
from scipy import stats
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import chi2

np.random.seed(6)

import warnings
warnings.filterwarnings('ignore')

sadness_df = pd.read_csv('Sadness.csv')


print("df is \n",sadness_df)

sadness_df['Total_people_surveyed'] = sadness_df['All_or_most_of_the_time'] + sadness_df['Some_of_the_time']

print("Changed df is \n",sadness_df)

print("Number of columns in dataset",len(list(sadness_df.columns)))

columns_list = list(sadness_df.columns)

print("The columns in the Dataset are \n")

for col in columns_list:
    print(col)



sadness_numerical_df = sadness_df.select_dtypes(include = 'number', exclude = None)

print("Numerical Data in the Dataset is \n",sadness_numerical_df)

n_of_total = sum(list(sadness_df['Total_people_surveyed']))

print("The total number of people surveyed is ",n_of_total)

tset, pval = ttest_1samp(sadness_numerical_df, 30)

print("Inducing significance level on the Probability Values in the data")

print("Type of pval ",type(pval))

print("Probability values p-values",pval)

significance_level = 0.01

if pval.any() < significance_level:    # alpha value (significance value) is 0.01 or 1%
   print(" we are rejecting null hypothesis")
else:
  print("we are accepting null hypothesis")

print("\n")

print(" ----------------------------------------------------------------------------------- \n")

print("The Degree of sadness has been calculated for each of the population as X/total(column) \n, where X is the count of current column and total(column) is sum of all the values in the column \n\n")

n_for_allormost = sum(list(sadness_df['All_or_most_of_the_time']))

n_for_some = sum(list(sadness_df['Some_of_the_time']))

print("Total number of people surveyed as sad for all or most of the time ",n_for_allormost)

print("Total number of people surveyed as sad for some of the time ",n_for_some)

print("Preprocessing the counts for people who are sad for all or most of the time ... \n\n")

sadness_df['Degree_of_sadness_allormost'] = sadness_df['All_or_most_of_the_time']/n_for_allormost

sadness_df['Degree_of_sadness_some'] = sadness_df['Some_of_the_time']/n_for_some

sadness_df['Total_Degree_of_sadness'] = sadness_df['Total_people_surveyed']/n_of_total

print("Changed DataFrame after applying Degree of Sadness \n")

print(sadness_df)

sadness_df_cor = sadness_df.corr(method = 'kendall')

ax = sns.heatmap(sadness_df_cor, annot=True)

print(ax)

plt.show()

print("Hence we can conclude that the correlation graph is different for each of the features involved")
