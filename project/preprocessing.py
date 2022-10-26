import numpy as np
'''
Columns ID - Name

0 - DER_mass_MMC
1 - DER_mass_transverse_met_lep
2 - DER_mass_vis
3 - DER_pt_h

4 - DER_deltaeta_jet_jet
5 - DER_mass_jet_jet
6 - DER_prodeta_jet_jet

7 - DER_delta_tau_lep
8 - DER_pt_tot
9 - DER_sum_pt
10 - DER_pt_ratio
11 - DER_met_phi_centrality
12 - DER_lep_eta_centrality

13 - PRI_tau_eta
14 - PRI_tau_phi

15 - PRI_tau_phi NOT SURE YET
16 - PRI_lep_eta
17 - PRI_lep_phi
18 - PRI_lep_phi NOT SURE YET
19 - 
20 - PRI_met_phi NOT SURE YET
21 - 
22 - PRI_jet_num

23 - PRI_jet_leading_pt
24 - PRI_jet_leading_eta
25 - PRI_jet_leading_phi
26 - PRI_jet_subleading_pt
27 - PRI_jet_subleading_eta
28 - PRI_jet_subleading_phi
29 - PRI_jet_all_pt
'''
# -*------------------------- Features Engeneering ---------------------------------*-

def add_offset(x): 
    '''
    Add a column of 1 to the feature matrix x
    '''
    return (np.c_[np.ones(x.shape[0]), x])


def standardize(x, mean_x=None, std_x =None):
    '''
    Standarize the data according the mean and the standard deviation
    and take as an input the features matrix X
    '''
    
    if (mean_x is None): 
        mean_x = np.mean(x, axis=0)
    if (std_x is None) : 
        std_x = np.std(x, axis=0)
        
    centered_data = x - mean_x
    std_data =centered_data[:,std_x>0] /std_x[std_x>0]

    if(len(std_x[std_x <=0])>0) :
        raise ValueError("DIVISION BY 0 : Find features with a 0 standard deviation")
 
    
    return std_data, mean_x, std_x

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


def preprocess_data_new(X_train, X_test, y_train, sampling_strategy=None):
    '''
    Testing preprocessing by subdividing into jet numbers
    '''
    jet_num_train = X_train[:, 22]
    jet_num_test = X_test[:, 22]

    # Still do not know if we should divide into 3 or 4 groups.
    def divide_into_subgroups(x, jet_num):
        x_groups = {
            'group_0': x[jet_num == 0.],
            'group_1': x[jet_num == 1.],
            'group_2': x[jet_num == 2.],
            'group_3': x[jet_num == 3.]
        }
        masks = {
            'group_0': jet_num == 0.,
            'group_1': jet_num == 1.,
            'group_2': jet_num == 2.,
            'group_3': jet_num == 3.
        }
        return x_groups, masks
    X_train_groups , _= divide_into_subgroups(X_train, jet_num_train)
    y_train_groups, _ = divide_into_subgroups(y_train, jet_num_train)
    X_test_groups, masks = divide_into_subgroups(X_test, jet_num_test)
    # TODO
    black_listed_columns = {
        'group_0': [4, 5, 6, 22, 23, 24, 25, 26, 27, 28, 29],
        'group_1': [4, 5, 6, 22, 26, 27, 28],
        'group_2': [22],
        'group_3': [22]
    }
    for group in X_train_groups.keys():
        ## First removing the black-listed columns the columns
        X_train_groups[group], X_test_groups[group] = _remove_columns(
             X_train_groups[group],
             X_test_groups[group],
             black_listed_columns[group]
        ) 
    
        ## Filling the missing values with the median of each sets
        X_train_groups[group] , X_test_groups[group] = _fill_missing_values(
            X_train_groups[group],
            X_test_groups[group]
        )
        
        ## Remove outliers on the training set only
        X_train_groups[group], y_train_groups[group] = _remove_outlier(
            X_train_groups[group],
            y_train_groups[group],
            threshold = 1.5,
            level = 5
        )
        ## Over/Under/None sampling, according to the chosen method
        ## Only for the training set, obviously
        if sampling_strategy == 'over':
            X_train_groups[group], y_train_groups[group] = _random_over_sampling(
                X_train_groups[group],
                y_train_groups[group],
                seed = 1
            )
        elif sampling_strategy == 'under':
            X_train_groups[group], y_train_groups[group] = _random_under_sampling(
                X_train_groups[group],
                y_train_groups[group],
                seed = 1
            )

    ## Should return dictonaries with group as keys
    return X_train_groups, X_test_groups, y_train_groups, masks

# -*-------------------------  Methods - preprocessing ---------------------------------*-
def _random_over_sampling(x, y, seed):
    '''
    Over sampling for an unbalanced dataset
    '''
    diff = len(y[y == -1.]) - len(y[y == 1.])
    sampling_nb = np.abs(diff)
    sampling_class = np.sign(diff)
    indices = np.array([i for i, y_ in enumerate(y) if y_ == sampling_class])

    # Setting seed for reproducability
    np.random.seed(seed)
    np.random.shuffle(indices)

    x = np.concatenate((x, x[indices[:sampling_nb], :]))
    y = np.concatenate((y, y[indices[:sampling_nb]]))
    # If the sampling is not 50/50 yet, in the case a class has more than twice the nb of elements in the other class
    if sampling_nb > len(indices):
        return _random_over_sampling(x, y, seed)
    return x, y

def _random_under_sampling(x, y, seed):
    '''
    Under sampling for an unbalanced dataset
    '''
    diff = len(y[y == 1.]) - len(y[y == -1.])
    sampling_nb = np.abs(diff)
    sampling_class = np.sign(diff)
    indices = np.array([i for i, y_ in enumerate(y) if y_ == sampling_class])
    
    # Setting seed for reproducability
    np.random.seed(seed)
    np.random.shuffle(indices)
    
    kept_indices = np.delete(np.arange(len(y)), indices[:sampling_nb])
    return x[kept_indices, :], y[kept_indices]
 

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
            med = np.median(feat[feat!=-999.0])
            X_train[:,j] = np.where(feat == -999.0, med, X_train[:,j])
            X_test[:,j] = np.where(X_test[:,j]==-999.0, med, X_test[:,j])
    
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
    print(standardize(x))
    print(_remove_columns(x, [1]))
    ## OKAY IT WORKS LEZGOOOOOOOOOO