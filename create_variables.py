import pandas as pd
import numpy as np

# Create lookup table for numerical features
def create_variables(dataset_name):
    
    if dataset_name.lower() in list(['Cleveland', 'cleveland', 'Stalog', 'stalog']):
        BP = ('trestbps',3,['low','normal','high'],[[-1e-16, 0, 67.5, 90],[67.5, 90, 127.5, 140],[127.5, 140, np.PINF, np.PINF]],[1,2,3,4])
        CHOL = ('chol', 3, ['low','normal','high'],[[-1e-16, 0, 149.5, 200],[149.5, 200, 229.1515, 239],[229.1515, 239, np.PINF, np.PINF]],[1,2,3,4])
        THALACH = ('thalach', 3,['low','normal','high'] ,[[-1e-16, 0, 65.7777, 88],[65.7777, 88, 133.5959, 149],[133.5959, 149, np.PINF, np.PINF]],[1,2,3,4])
        OLDPEAK = ('oldpeak',2,['no','yes'], [[-1e-16, 0, 0.7474, 1],[0.7474, 1, np.PINF, np.PINF]],[1,2,3,4])
        numerical_features = [BP, CHOL, THALACH, OLDPEAK]
        
        
    elif dataset_name.lower() in list (['ZAlizadehsani', 'zalizadehsani','test']):
        # columns=['Name', 'K', 'LingValNames', 'LingValValues','Profiles'])
        FBS = ('FBS',3,['low','normal','high'],[[-1e-16,0,45,60],[45,60,89.25,99],[89.25,99,np.PINF,np.PINF]],[1,2,3,4])
        CR1 = ('Cr',3,['low','normal','high'],[[-1e-16,0,0.5625,0.75],[0.5625,0.75,1.0875,1.2],[1.0875,1.2,np.PINF,np.PINF]],[1,3])
        CR2 = ('Cr',3,['low','normal','high'],[[-1e-16,0,0.4875,0.65],[0.4875,0.65,0.9125,1],[0.9125,1,np.PINF,np.PINF]],[2,4])
        HB1 = ('HB',3,['low','normal','high'],[[-1e-16, 0, 10.125, 13.5],[10.125, 13.5, 16.5, 17.5],[16.5, 17.5, np.PINF, np.PINF]],[1,3])
        HB2 = ('HB',3,['low','normal','high'],[[-1e-16, 0, 9, 12],[9, 12, 15, 16],[15, 16, np.PINF, np.PINF]],[2,4])
        LDL = ('LDL', 2, ['normal','high'], [[-1e-16, 0, 97.5, 130],[97.5, 130, np.PINF, np.PINF]],[1,2,3,4])
        HDL = ('HDL', 2, ['low','normal'], [[-1e-16, 0, 30, 40],[30, 40, np.PINF, np.PINF]],[1,2,3,4])
        WBC = ('WBC',3,['low','normal','high'],[[-1e-16, 0, 3000, 4000],[3000, 4000, 8500, 10000],[8500, 10000, np.PINF, np.PINF]],[1,2,3,4])
        BUN = ('BUN',3,['low','normal','high'],[[-1e-16, 0, 6, 8],[6, 8, 17.75, 21],[17.75, 21, np.PINF, np.PINF]],[1,2,3,4])    
        K = ('K',3,['low','normal','high'],[[-1e-16, 0, 2.55, 3.4],[2.55, 3.4, 4.825, 5.3],[4.825, 5.3, np.PINF, np.PINF]],[1,2,3,4])
        NA = ('NA',3,['low','normal','high'],[[-1e-16, 0, 102.75, 137],[102.75, 137, 144.5, 147],[144.5, 147, np.PINF, np.PINF]],[1,2,3,4])
        PLT = ('PLT',3,['low','normal','high'],[[-1e-16, 0, 112.5, 150],[112.5, 150, 366.75, 399],[366.75, 399, np.PINF, np.PINF]],[1,2,3,4])
        BP = ('BP',3,['low','normal','high'],[[-1e-16, 0, 67.5, 90],[67.5, 90, 127.5, 140],[127.5, 140, np.PINF, np.PINF]],[1,2,3,4])
        PR = ('PR',3,['low','normal','high'],[[-1e-16, 0, 45, 60],[45, 60, 90, 100],[90, 100, np.PINF, np.PINF]],[1,2,3,4])
        TG = ('TG',2,['normal','high'],[[-1e-16, 0, 150, 200],[150, 200, np.PINF, np.PINF]],[1,2,3,4])
        Neut = ('Neut',3,['low','normal','high'],[[-1e-16, 0, 34.5, 46],[34.5, 46, 70, 78],[70, 78, np.PINF, np.PINF]],[1,2,3,4])
        Lymph = ('Lymph',3,['low','normal','high'],[[-1e-16, 0, 13.5, 18],[13.5, 18, 43.5, 52],[43.5, 52, np.PINF, np.PINF]],[1,2,3,4])
        EF = ('EF',2,['low','normal'],[[-1e-16, 0, 37.5, 50],[37.5, 50, np.PINF, np.PINF]],[1,2,3,4])
        numerical_features = [FBS,CR1,CR2,HB1,HB2,LDL,HDL,WBC,BUN,K,NA,PLT,BP,PR,TG,Neut,Lymph,EF]
        
    df = pd.DataFrame(numerical_features, columns=['Name', 'K', 'LingValNames', 'LingValValues','Profiles'])
    return df
