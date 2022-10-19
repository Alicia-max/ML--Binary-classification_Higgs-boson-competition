import numpy as np

'''
Black listed columns ID - Name

4 - DER_deltaeta_jet_jet
5 - DER_mass_jet_jet
6 - DER_prodeta_jet_jet

15 - PRI_tau_phi
18 - PRI_lep_phi
20 - PRI_met_phi

22 - NOT SURE YET (Nbs of jet) --> REMOVE

23 - PRI_jet_leading_pt
24 - PRI_jet_leading_eta
25 - PRI_jet_leading_phi
26 - PRI_jet_subleading_pt
27 - PRI_jet_subleading_eta
28 - PRI_jet_subleading_phi

29 - NOT SURE YET TOO (Total pt) --> Maybe keep this? --> The scalar sum of the transverse momentum of all the jets of the events
'''
def preprocess_data(x,y, test = False):
    '''
    Preprocessing the data.
    Should be applied to the train and test sets separately
    '''
    black_listed_columns  = [4, 5, 6, 15, 18, 20, 22, 23, 24, 25, 26, 27, 28]
   
    ## First removing the black-listed columns the columns
    x = _remove_columns(x, black_listed_columns)
    
    ## Filling the missing values with the median of each sets
    x = _fill_missing_values(x)
    
    ## Remove outliers on the training set only
    if test==False :
        x, y = _remove_outlier(x,y, 1.5, 5)  

    ## Normalizing the data
    x = _standardize(x)
    
    ##Offset adding
    x=_add_offset(x)

    return x,y

def _standardize(x):
    '''
    Standarize the data according the mean and the standard deviation
    and take as an input the features matrix X
    
    '''
    
    centered_data = x - np.mean(x, axis=0)
    std_data = centered_data / np.std(centered_data, axis=0)
    return std_data

def _remove_columns(x, columns):
    '''
    Remove the given column of X 
    Input : 
        - x : the features matrix X
        - columns : index of column to be removed   
    Output :
        - x : the modified features matrix X
    '''
    
    total_nb_columns = x.shape[1]
    kept_columns =  np.delete(np.arange(total_nb_columns), columns)
    return x[:, kept_columns]

def _fill_missing_values(x, threshold = .8):
    '''
    Verify for each features of X the level of missing information 
    and replace the missing values (by the median) or discard the feature according to the threshold.
    Input : 
        - x : the features matrix X
        - threshold : it defines if a column is uninformative or not 
    Output :
        - x : the modified features matrix X
    '''
    
    discarded_cols = []
    
    #Check missing value according the features 
    for j in range(x.shape[1]) : 
        feat = x[:,j]
        #check % of missing value per col
        miss_perc = len(feat[feat==-999.0])/len(feat)
        if miss_perc > threshold:
            discarded_cols.append(j)
        else:
            # I observed that mean was a bit better, but median is more robust
            mean = np.median(feat[feat!=-999.0])
            x[:,j] = np.where(feat == -999.0, mean, x[:,j])
    
    x = _remove_columns(x, discarded_cols)
    return x

def _remove_outlier(x, y, cst =1.5, level=10):
    '''
    Delete rows with outliers, that for example may be due to experimental error.
    Input : 
        - x : the features matrix X
        - cste : define how far away from the quantile should be a point to be a outliers
        - level : define the used quantiles
    Output :
        - x : the modified features matrix X
    '''

    col_out=[]
   
    #find outliers in each column 
    for col in range(x.shape[1]) : 
        outliers_idx=(_find_outliers(x[:,col],threshold, level))
        for it, outlier in enumerate(outliers_idx) : 
            col_out.append(outlier)
        
    if(len(col_out)>0) :
        
        idx_outliers = np.array(list(set(col_out)))
        x = np.delete(x, idx_outliers, axis=0)
        y = np.delete(y,idx_outliers , axis=0)
        
    else : 
        print("There's no outliers")
        
    return np.array(x), np.array(y)

def _find_outliers(x, cste=1.5, level=10): 
     '''
    Scearch the index containing outliers using the interquartile range (IQR)
    
    Input : 
        - a list of values : x
        - level : defined the used quantiles
        - cste : define how far away from the quantile should be a point to be a outliers
        
    Output : 
        - outliers_index : indexes of the list x corresponding to the outliers
    '''
    sorted(x)
    
    #Compute the quantiles & IQR
    q1, q3= np.percentile(np.array(x),[level,100-level])
    IQR = (q3-q1)
    
    #Compute the boundaries
    lower_bound = q1 -(threshold * IQR) 
    upper_bound = q3 +(threshold * IQR) 
    
    #Recover index of values out of the boundaries
    outliers_index=[]
    for idx, val in enumerate(x):
        if (val<=lower_bound or val >=upper_bound): 
            outliers_index.append(idx)
            
    return outliers_index

def _add_offset(x): 
    '''
    Add a column of 1 to the feature matrix x
    '''
    return (np.c_[np.ones(x.shape[0]), x])
    
if __name__ == "__main__":
    # For testing purposes
    x = np.random.rand(100,3)
    print(x)
    print(_standardize(x))
    print(_remove_columns(x, [1]))
    ## OKAY IT WORKS LEZGOOOOOOOOOO