# %% PRIMARILY NOTES FROM A DATACAMP COURSE
import pandas as pd

# a couple of simple dataframes to play around with and test these functions
dfDict1 = {'name': ['Bonnie', 'Betty', 'Chase', 'Nancy'],
           'height': [32, 40, 61, 15],
           'weight': [10, 20, 50, 22],
           'color': ['spotted', 'black', 'white', 'brown']}
dfDict2 = {'name': ['Bonnie', 'Terrance', 'Chase', 'Blue', 'Toby'],
           'height': [32, 18, 61, 44, 27],
           'weight': [10, 17, 50, 31, 19],
           'has_owner': [True, False, False, True, True]}

dogs1 = pd.DataFrame(dfDict1)
dogs2 = pd.DataFrame(dfDict2)
print(dogs2)

# %% Typical Joining Functions
#df.merge()             #default is inner
    #noteworthy parameters: on='col_name' (or right_on='right_col_name', left_on='left_col_name')),
                        #how='left/right/inner/outer',suffixes=('_suff1','_suff2')
#pd.merge_ordered()     #PANDAS CALL, NOT DATAFRAME METHOD #default is outer
#pd.merge_asof()     #PANDAS CALL, NOT DATAFRAME METHOD #default is left
#pd.concat()            #PANDAS CALL, NOT DATAFRAME METHOD
    #noteworthy parameters: join='inner', keys=['key','list','of','concats'], sort=True,
                        #ignore_index=True/False (must be False with keys)
#df.append() #simplified concat method always has join='outer'
    #noteworthy parameters: keys = ['as','above'], sort=True/False

# %% Typical Sorting Functions
#df.groupby() #yeilds a groupby object and not a dataframe?
    #noteworthy parameters:
#df.pivot_table()
#df.sort_values()
#df.sort_index()
#df.isin()
#df.isna() (or df.isnull())
#pd.melt()          #PANDAS CALL, NOT DATAFRAME METHOD

# %% Typical Math Functions
#df.describe() #gives count, mean, std, min, 25%, 50%, 75%, and max
#df.mean()
#df.median()
#df.sum()
#df.max()
#df.corr()
#df.std()

# %% General ways to apply functions
#df.agg({'col_name':'function'})
    #noteworthy parameters:
#df.apply()

#example: df.apply(lambda x: x.replace(char,''))

# %% ---LABEL ENCODING VS ONE HOT ENCODING---
# from sklearn.preprocessing import LabelEncoder

# label encoding provides numerical assigments to values, but these are (incorrectly) implicityly
# assumed as having significance.
# for example, label '0' is less than label '3', but this is meaningless if 0=bird and 3=dog

# one hot encoding provides duplicate information of a single row
dogs_comb = pd.merge(dogs1,dogs2, how='outer')
ohe = pd.get_dummies(dogs_comb['name'], prefix='ohe_name')

# A target encoding is any kind of encoding that replaces a feature's categories with some number 
# derived from the target. see https://www.kaggle.com/ryanholbrook/target-encoding

# %% pandas crosstab can help you visualize clustering
ct = pd.crosstab(dogs_comb['height'], dogs_comb['name'])
print(ct)

# %% Sets
y = set([1, 1, 2, 3, 3, 3, 4]) # creates a set of only the unique values
print(y)
print(y=={1,2,3,4}) # curly braces can also be used to define a set
z = {1,1,2,3,3,3,4}
print(z)

# %% Iterators
w = 'python'
w_iterator = iter(w) # creates an iterator object
print(w_iterator)
print(next(w_iterator))
print(next(w_iterator))
print(next(w_iterator))
print(next(iter(w))) # note that recalling iter creates a new object each time
print(next(iter(w)))
# %%
