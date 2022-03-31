import pandas as pd
import numpy as np
import time
from create_variables import *

def assign_profile(record):
    age = record['Age']
    sex = record['Sex']
    if age=='normal' and sex=='male':
        return 1
    elif age=='normal' and sex=='female':
        return 2
    elif age=='high' and sex=='male':
        return 3
    else:
        return 4


def trapmf(x, mf):
    a = mf[0]
    b = mf[1]
    c = mf[2]
    d = mf[3]
    if x < a or x > d:
        return 0
    elif a<=x<=b:
        return (x-a) / (b-a) 
    elif b<x<c:
        return 1
    elif c<=x<=d:
        return (d-x)/(d-c)
    

def Profile_basedFuzzyPartitioning(df,dataset_name):
    # Load All numerical Variables and their membership function
    lookup = create_variables(dataset_name)
    
    # type for banset
    types = list()
    
    # Assign Profile
    pis =  [assign_profile(df.iloc[i]) for i in range (0,df.shape[0])]
    number_of_profiles = sorted(set(pis))
    df['P#'] = pis
    
    # Separate numerical and categiral features
    cat_df = df.select_dtypes(include=['object'])
    num_df = df.select_dtypes(exclude=['object'])
    
    # create FistScan
    FirstScan = dict()
    
    # For numerical features
    for col in num_df.columns:
        if col =='P#':
            break
        x = num_df[col]
        resp = lookup[lookup['Name'].str.lower() == col.lower()]
        
        # for single profile numerical features
        if resp.shape[0]==1:
            ling_val_names = list(resp['LingValNames'])                            
            ling_val_values = list(resp['LingValValues'])        
            K =  int(resp['K'])
            i = 0
            for i in range(0,K):
                name = ling_val_names[0][i]
                name = col +'_' + name 
                mf = ling_val_values[0][i]                        
                mvs = [trapmf(xi,mf) for xi in x]    
                FirstScan[name] = mvs
                types.append(col)
        
        # for multi profile numerical features
        elif resp.shape[0]>1:
                    
            ling_val_names = list(resp['LingValNames'][resp.index[0]])
            K =  int(resp['K'][resp.index[0]])
            lnames = list()
            for name in ling_val_names:
                types.append(col)
                name = col +'_' + name 
                lnames.append(name)
                
            part_FirstScan = pd.DataFrame(columns=lnames)
            temp_FirstScan = part_FirstScan.copy()
            for p in number_of_profiles:
                num_df_temp = num_df[(num_df['P#'] == p)]
                x = num_df_temp[col]
    
                for index in resp.index:
                    if p in resp['Profiles'][index]:
                        ling_val_values = list(resp['LingValValues'][index] )
                        break
                            
                i = 0    
                for i in range(0,K):
                    name = ling_val_names[i]
                    name = col +'_' + name
                    mf = ling_val_values[i]                        
                    mvs = [trapmf(xi,mf) for xi in x]
                    temp_FirstScan[name] = mvs 
                    
                temp_FirstScan.index = x.index
                part_FirstScan = part_FirstScan.append(temp_FirstScan)            
                temp_FirstScan=[]
                temp_FirstScan=pd.DataFrame(columns=lnames)
            part_FirstScan.sort_index(inplace=True)
            for col2 in part_FirstScan.columns:
                FirstScan[col2] = list(part_FirstScan[col2])
    
    # For Categorical features
    for col in cat_df.columns:
        x = cat_df[col]
        ling_val_values = list(set(x))
        K = len(ling_val_values)
        for i in range(0,K):        
            types.append(col)
            name = col +'_' + ling_val_values[i]
            mvs = (x == ling_val_values[i])       
            FirstScan[name] = list(mvs)
            
    FirstScan = pd.DataFrame(FirstScan)
    FirstScan.loc[-1] = types
    FirstScan.index = FirstScan.index + 1  # shifting index
    FirstScan.sort_index(inplace=True) 
    return FirstScan
