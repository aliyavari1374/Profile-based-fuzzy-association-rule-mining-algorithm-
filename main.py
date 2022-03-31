# This is Modify version from Apriori algorithm by Ali Yavari For Profile-based Fuzzy Association Rule Mining Algorithm
# Original Code Written by, Sebastian Raschka <sebastianraschka.com> and Available in Github:   http://rasbt.github.io/mlxtend/user_guide/frequent_patterns/generate_new_combinations/

import pandas as pd
from mlxtend.frequent_patterns import association_rules
import apriori
import time
from Profile_basedFuzzyPartitioning import *
import os 
import pickle

os.chdir('datasets/')
dataset = "cleveland.csv"
df = pd.read_csv(dataset)  


#df = pd.read_csv('FirstScanTable._ZAlizadehSaniV2T.csv') 
#df = pd.read_csv('FirstScanTable_ClevelandV2T.csv')
#df = pd.read_csv('FirstScanTable_StalogV2T.csv')


start = time.time()
FirstScan = Profile_basedFuzzyPartitioning(df,dataset.split('.')[0])
end = time.time() - start 

types = FirstScan.iloc[0,:] # For Banset
types = types.reset_index(drop=True)    


FirstScan = FirstScan.drop(FirstScan.index[0])# Delete and separate banset from dataframe
FirstScan = FirstScan.reset_index(drop=True)    
FirstScan = FirstScan.astype(float)

    
start1 = time.time()
items1 = apriori.apriori(FirstScan,types, min_support=0.03, use_colnames=True)    
end1 = time.time() - start1



# times = dict()
# generated_items = dict()
# sups = [0.03,0.05,0.07,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
# for i in range(0,len(sups)):
#     start1 = time.time()
#     items1 = apriori.apriori(FirstScan,types, min_support=sups[i], use_colnames=True)    
#     end1 = time.time() - start1
#     times[sups[i]] = end1
#     generated_items[sups[i]] = len(items1)   
#     print("run ", str(i), "is finished")



rules = association_rules(items1, metric="confidence", min_threshold=0.9)
#rules[(rules['lift'] >= 1) & (rules['confidence'] >= 1)]
#rules = rules[rules['confidence'] >= 0.9]

# Filter targets rules
filter_ = (rules["consequents"]==((frozenset({"CAD_yes"})) or (frozenset({"CAD_no"}))))
proned_rules = rules[filter_==True]



# with open('generated_items.pkl', 'wb') as handle:
#     pickle.dump(generated_items, handle, protocol=pickle.HIGHEST_PROTOCOL)

# with open('times.pkl', 'wb') as handle:
#     pickle.dump(times, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
# with open('sups_my.pkl', 'rb') as handle:
#     x = pickle.load(handle)
    
# with open('sups_prev.pkl', 'rb') as handle:
#     y = pickle.load(handle)