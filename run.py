from project.helpers import *
from project.implementations import *

def run():
    datapath_train = './data/train.csv'
    datapath_test = './data/test.csv'

    # Sub sampling for testing purposes
    target_train, features_train, id_train = load_csv_data(datapath_train, sub_sample=True)
    _, features_test, id_test = load_csv_data(datapath_test, sub_sample=True)

    print('Size of train features', features_train.shape)
    print('Size of test features', features_test.shape)

    # Simple least square for starters
    W, loss = least_squares(target_train, features_train)

    # Test submission writing
    test_prediction = features_test.dot(W)
    create_csv_submission(id_test, np.sign(test_prediction), 'submission_test.csv')


if __name__ == "__main__":
    set_seed(1)
    run()