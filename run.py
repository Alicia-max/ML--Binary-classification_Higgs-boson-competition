from project.helpers import *
from project.implementations import *
from project.preprocessing import *

def run():
    datapath_train = './data/train.csv'
    datapath_test = './data/test.csv'

    # Loading data
    target_train, features_train, id_train = load_csv_data(datapath_train, sub_sample=True)
    _, features_test, id_test = load_csv_data(datapath_test, sub_sample=True)

    print('Size of train features', features_train.shape)
    print('Size of test features', features_test.shape)

    ## Preprocess data
    preprocessed_features_train = preprocess_data(features_train, target_train)
    preprocessed_features_test = preprocess_data(features_test)

    # ridge regression test
    W, loss = ridge_regression(target_train, preprocessed_features_train, 0.1)
    train_prediction = preprocessed_features_train.dot(W)
    print(train_prediction)
    
    accuracy = np.sum((np.sign(train_prediction) == target_train)) * 100 / len(train_prediction)
    print('train accuracy ', accuracy)
    # Test submission writing
    test_prediction = preprocessed_features_test.dot(W)
    create_csv_submission(id_test, np.sign(test_prediction), 'submission_test.csv')


if __name__ == "__main__":
    set_seed(1)
    run()