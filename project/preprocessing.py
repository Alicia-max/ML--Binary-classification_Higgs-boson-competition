import numpy as np

'''
Black listed columns ID - Name

4 - DER_deltaeta_jet_jet
5 - DER_mass_jet_jet
6 - DER_prodeta_jet_jet

15 - PRI_tau_phi NOT SURE YET
18 - PRI_lep_phi NOT SURE YET
20 - PRI_met_phi NOT SURE YET

22 - PRI_jet_num

23 - PRI_jet_leading_pt
24 - PRI_jet_leading_eta
25 - PRI_jet_leading_phi
26 - PRI_jet_subleading_pt
27 - PRI_jet_subleading_eta
28 - PRI_jet_subleading_phi


'''
# -*------------------------- Features Engeneering ---------------------------------*-

def add_offset(x): 
    '''
    Add a column of 1 to the feature matrix x
    '''
    return (np.c_[np.ones(x.shape[0]), x])


def standardize(x):
    '''
    Standarize the data according the mean and the standard deviation
    and take as an input the features matrix X
    '''
    centered_data = x - np.mean(x, axis=0)
    std=np.std(centered_data, axis=0)
    
    std_data = centered_data[:,std>0] /std[std>0]
    
    std_0 = std[std <=0]
    
    if(len(std_0)>0) :
        raise ValueError("DIVISION BY 0 : Find features with a 0 standard deviation")
 
    
    return std_data

def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    poly = np.ones((len(x),0))
    
   
    for deg in range(1, degree+1):
      
        poly = np.c_[poly, np.power(x, deg)]   
  
    return poly

# -*-------------------------  Data preprocessing ---------------------------------*-

def preprocess_data(X_train, X_test, y_train):
    '''
    Preprocessing the data.
    Should be applied to the train and test sets separately
    '''
    black_listed_columns  = [16,18, 22, 23, 24, 25, 26, 27, 28]
   
    ## First removing the black-listed columns the columns
    X_train, X_test = _remove_columns(X_train, X_test,  black_listed_columns)
   
    
    ## Filling the missing values with the median of each sets
    X_train , X_test = _fill_missing_values(X_train, X_test)
    
    ## Remove outliers on the training set only
    X_train, y_train = _remove_outlier(X_train,y_train, 1.5, 5)  

    return X_train, X_test, y_train

# -*-------------------------  Methods - preprocessing ---------------------------------*-

def _remove_columns(X_train, X_test, columns):
    '''
    Remove the given column of X 
    Input : 
        - x : the features matrix X
        - columns : index of column to be removed   
    Output :
        - x : the modified features matrix X
    '''
    total_nb_columns = X_train.shape[1]
    kept_columns =  np.delete(np.arange(total_nb_columns), columns)
    return X_train[:, kept_columns], X_test[:, kept_columns]

def _fill_missing_values(X_train,X_test, threshold = .8):
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
    for j in range(X_train.shape[1]) : 
        feat = X_train[:,j]
        #check % of missing value per col
        miss_perc = len(feat[feat==-999.0])/len(feat)
        if miss_perc > threshold:
            discarded_cols.append(j)
        else:
            mean = np.median(feat[feat!=-999.0])
            X_train[:,j] = np.where(feat == -999.0, mean, X_train[:,j])
            X_test[:,j] = np.where(X_test[:,j]==-999.0, mean, X_test[:,j])
    
    X_train, X_test = _remove_columns(X_train, X_test, discarded_cols)

    
    return X_train, X_test

def _remove_outlier(x, y, threshold = 1.5, level=5):
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

def _find_outliers(x, threshold=1.5, level=5):
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




if __name__ == "__main__":
    # For testing purposes
    x = np.random.rand(100,3)
    print(x)
    print(_standardize(x))
    print(_remove_columns(x, [1]))
    ## OKAY IT WORKS LEZGOOOOOOOOOO