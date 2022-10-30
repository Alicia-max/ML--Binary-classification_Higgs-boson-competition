import numpy as np

# -*------------------------- Features Engeneering ---------------------------------*-

def add_offset(x): 
    '''
    Add a column of 1 to the feature matrix x
    '''
    return (np.c_[np.ones(x.shape[0]), x])


def standardize(x, mean_x=None, std_x =None):
    '''
    Standarize the data according the mean and the standard deviation
    and take as an input the Data matrix x
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
    """
    Polynomial basis functions for input data x, for j=2 up to j=degree.
    """
    out = x
    for deg in range(2, degree+1):
        out = np.c_[out, np.power(x, deg)]   
    return out

def cross_terms(x):
    """
    TODO
    """
    out = np.ones((len(x),0))
    for i in range(x.shape[1]):
        for j in range(i+1, x.shape[1]):
            out = np.c_[out, x[:,i]*x[:,j]]
    return out

# -*-------------------------  Data preprocessing ---------------------------------*-

def preprocess_data(X_train, X_test, y_train, Jet_Features=None):
    '''
    Preprocessing the data without considering the Jet feature.
    Input : 
        - X_train : Train Data Matrix
        - X_test : Test Data Matrix
        - y_train : Expected value vector for the train set 
        - Jet_Features : index of features relate to the Jet_Number
    '''
    if(Jet_Features is not None) : 
        # Removing the Jet related features
        X_train, X_test = _remove_columns(X_train, X_test, Jet_Features)
   
    
    # Filling the missing values with the median of each sets
    X_train , X_test = _fill_missing_values(X_train, X_test)
    
    # Remove outliers on the training set only
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
        'group_0': [4, 5, 6, 12, 22, 23, 24, 25, 26, 27, 28, 29],
        'group_1': [4, 5, 6, 12, 22, 26, 27, 28],
        'group_2': [22],
        'group_3': [22]
    }
    angle_columns = {
        'group_0': [15, 18, 20],
        'group_1': [15, 18, 20, 25],
        'group_2': [15, 18, 20, 25],
        'group_3': [15, 18, 20, 25]
    }
    for group in X_train_groups.keys():
        # Even before, add cos and sin of angles at the end (and then remove the angles)
        X_train_groups[group], X_test_groups[group] = _add_cos_sin_angles(
            X_train_groups[group],
            X_test_groups[group],
            angle_columns[group]
        )
        ## First removing the black-listed columns the columns
        X_train_groups[group], X_test_groups[group] = _remove_columns(
             X_train_groups[group],
             X_test_groups[group],
             black_listed_columns[group] + angle_columns[group]
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
        
    ## Should return dictonaries with group as keys
    return X_train_groups, X_test_groups, y_train_groups, masks

# -*-------------------------  Methods - preprocessing ---------------------------------*-

def _add_cos_sin_angles(x, x_test, columns):
    for column in columns:
        x = np.c_[x, np.cos(x[:, column]), np.sin(x[:, column])]
        x_test = np.c_[x_test, np.cos(x_test[:, column]), np.sin(x_test[:, column])]
    print('Angles added')
    return x, x_test

def _remove_columns(X_train, X_test, columns):
    '''
    Remove the given column of X 
    
    Input : 
        - x : Data Matrix
        - columns : index of column to be removed   
    '''
    total_nb_columns = X_train.shape[1]
    kept_columns =  np.delete(np.arange(total_nb_columns), columns)
    return X_train[:, kept_columns], X_test[:, kept_columns]

def _fill_missing_values(X_train,X_test, threshold = .8):
    '''
    Verify for each features of X the level of missing information 
    and replace the missing values (by the median) or discard the feature according to the threshold.
    
    Input : 
        - x : Data Matrix
        - threshold : it defines if a column is uninformative or not 
    '''
    discarded_cols = []
    
    # Iteration on each feature
    for j in range(X_train.shape[1]) : 
        feat = X_train[:,j]
        
        # Check % of missing value per colonne
        miss_perc = len(feat[feat==-999.0])/len(feat)
        if miss_perc > threshold:
            discarded_cols.append(j)
            print('Feature number', j, 'is removed')
            
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
        - x : Data Matrix 
        - cste : Define how far away from the quantile should be a point to be a outliers
        - level : Define the used quantiles
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
    Search for the index containing the outliers using the interquartile range (IQR)
    
    Input : 
        - x : list of values
        - level : Defined the used quantiles
        - cste : Define how far away from the quantile should be a point to be a outliers
        
    Output : 
        - outliers_index : Indexes of the list x corresponding to the outliers
    '''
    sorted(x)
    
    # Compute the quantiles & IQR
    q1, q3= np.percentile(np.array(x),[level,100-level])
    IQR = (q3-q1)
    
    # Compute the boundaries
    lower_bound = q1 -(threshold * IQR) 
    upper_bound = q3 +(threshold * IQR) 
    
    # Recover index of values out of the boundaries
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
    print(x[0,:])
    print(build_poly(x, 2)[0,:])
    ## OKAY IT WORKS LEZGOOOOOOOOOO