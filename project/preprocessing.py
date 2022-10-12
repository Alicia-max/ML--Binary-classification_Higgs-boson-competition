import numpy as np

def standarize (x) :
    
    centered_data = x - np.mean(x, axis=0)
    std_data = centered_data / np.std(centered_data, axis=0)
    
    return std_data



def missing_values (x_train, x_test, th=0.9) : 
    
    #Check missing value according the features 
    remove_col=[]
    
    for j in range(x_train.shape[1]) : 
        
        feat=x_train[:,j]
        
        #check % of missing value per col
        miss_perc = len(feat[feat==-999.0])/len(feat)
        
        #replace missing value by the median of non missing value
        if (miss_perc<th): 
            med=np.median(feat[feat!=-999.0])
            x_train[:,j] = np.where(feat==-999.0, med, x_train[:,j])
            x_test[:,j] = np.where(x_test[:,j]==-999.0, med , x_test[:,j])
        else : 
            remov_col.append(col)
                
    x_train = np.delete(x_train, remove_col, axis = 0)
    x_test = np.delete(x_test, remove_col, axis = 0)
  
        
    return x_train, x_test
        
def outlier(x):
    
        
        
def remove_features (x, features) : 
    