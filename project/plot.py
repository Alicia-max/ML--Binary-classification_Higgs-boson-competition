import numpy as np
import matplotlib.pyplot as plt

def cross_validation_visualization(param, acc_tr, acc_te, name_param):
    """visualization the curves of acc_tr and acc_te."""
   
    plt.semilogx(param, acc_tr, marker=".", color='b', label='train error')
    plt.semilogx(param, acc_te, marker=".", color='r', label='test error')
    plt.xlabel(name_param)
    plt.ylabel("accuracy")
    plt.title("cross validation")
    plt.legend(loc=2)
    plt.grid(True)
   
