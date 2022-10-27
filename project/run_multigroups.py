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
    preprocessed_features_train,preprocessed_features_test,preprocessed_y, test_masks = preprocess_data_new(
        features_train, 
        features_test,
        target_train, 
        sampling_strategy = None
    )

    degree = params['degree']
    del params['degree']

    fourier = params['fourier']
    del params['fourier']

    #Offset 
    offset=params['offset']
    del params['offset']

    test_prediction = np.zeros(shape=(features_test.shape[0],))
    groups = ['group_0', 'group_1', 'group_2', 'group_3']
    for group in groups:
        print(group)
        ##Poly
        tx_tr = build_poly(preprocessed_features_train[group], degree)
        tx_te = build_poly(preprocessed_features_test[group], degree)

        tx_tr = fourier_encoding(tx_tr, fourier)
        tx_te = fourier_encoding(tx_te, fourier)

        ##standarization
        tx_tr_std, mean, std= standardize(tx_tr)
        tx_te_std, _, _= standardize(tx_te, mean, std)
        
        if(offset):
            tx_tr_std =  add_offset(tx_tr_std)
            tx_te_std =  add_offset(tx_te_std)
        
        W, loss = method(preprocessed_y[group], tx_tr_std, **params)
        test_prediction[test_masks[group]] = predict(tx_te_std, W)

    create_csv_submission(id_test, np.sign(test_prediction), 'submission_nosampling_leasqsuares_fourier1.csv')


if __name__ == "__main__":
    params = {
        'offset': True,
        'degree': 8,
        'fourier': 1
    }
    run(least_squares, params)