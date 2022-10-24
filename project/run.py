from helpers import *
from implementations import *
from preprocessing import *
#from cross_val import *

def run(method, params):
    PATH_TRAIN= '../data/train.csv'
    PATH_TEST = '../data/test.csv'

    # Loading data
    target_train, features_train, id_train = load_csv_data(PATH_TRAIN, sub_sample=False)
    _, features_test, id_test = load_csv_data(PATH_TEST, sub_sample=False)

    
    ## Preprocess data
    preprocessed_features_train,preprocessed_features_test,preprocessed_y = preprocess_data(features_train,features_test, target_train)

    
    degree = params['degree']
    del params['degree']
    
    ##Poly
    tx_tr = build_poly(preprocessed_features_train, degree)
    tx_te = build_poly(preprocessed_features_test, degree)
    
    ##standarization
    tx_tr_std= standardize(tx_tr)
    tx_te_std= standardize(tx_te)
    
    #Offset 
    offset=params['offset']
    del params['offset']
    
    if(offset):
        tx_tr_std =  add_offset(tx_tr_std)
        tx_te_std =  add_offset(tx_te_std)
    
    W, loss = method(preprocessed_y, tx_tr_std, **params)
    test_prediction = predict(tx_te_std, W)
    
    create_csv_submission(id_test, np.sign(test_prediction), 'submission_test.csv')


if __name__ == "__main__":
    run()