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


def preprocess_data(x):
    '''
    Preprocessing the data.
    Should be applied to the train and test sets separately
    '''
    black_listed_columns  = [4, 5, 6, 15, 18, 20, 22, 23, 24, 25, 26, 27, 28]
    # Should we include 29 ?
    ## First removing the black-listed columns the columns
    x = _remove_columns(x, black_listed_columns)
    ## Filling the missing values with the median of each sets
    x = _fill_missing_values(x)
    ## Should we do the outliers here ?

    ## Polynomial expansion
    x = _polynomial_expansion(x)

    # I was going to try this but my conda failed to activate at the end of the day :(
    # Will fix that try this sooooon :)

    ## Normalizing the data
    x_normalized = _standardize(x)

    return x_normalized



def _standardize(x):
    centered_data = x - np.mean(x, axis=0)
    std_data = centered_data / np.std(centered_data, axis=0)
    return std_data

def _remove_columns(x, columns):
    total_nb_columns = x.shape[1]
    kept_columns =  np.delete(np.arange(total_nb_columns), columns)
    return x[:, kept_columns]

def _fill_missing_values(x, threshold = .8):
    discarded_cols = []
    #Check missing value according the features 
    for j in range(x.shape[1]) : 
        
        feat = x[:,j]
        #check % of missing value per col
        miss_perc = len(feat[feat==-999.0])/len(feat)
        if miss_perc > threshold:
            discarded_cols.append(j)
        else:
            # I observed that mean was a bit better
            mean = np.mean(feat[feat!=-999.0])
            x[:,j] = np.where(feat == -999.0, mean, x[:,j])
    
    x = _remove_columns(x, discarded_cols)
    return x

def _polynomial_expansion(x, degree = 3):
    nb_points = x.shape[0]
    expanded_x = [x]
    for i in range(2, degree + 1):
        expanded_x.append(x**i)
    return np.array(expanded_x).reshape(nb_points, -1)

def outlier(x):
    pass

if __name__ == "__main__":
    # For testing purposes
    x = np.random.rand(100,3)
    print(x)
    print(_standardize(x))
    print(_remove_columns(x, [1]))
    ## OKAY IT WORKS LEZGOOOOOOOOOO