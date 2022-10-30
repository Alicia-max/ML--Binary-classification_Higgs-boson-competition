# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

def cross_validation_visualization(param, acc_tr, acc_te, name_param):
    """Visualization the curves of acc_tr and acc_te."""
    
    plt.semilogx(param, acc_tr, marker=".", color='salmon', label='train error')
    plt.semilogx(param, acc_te, marker=".", color='lightblue', label='test error')
    plt.xlabel(name_param)
    plt.ylabel("Accuracy")
    plt.title("Cross validation")
    plt.legend(loc=2)
    plt.grid(True)
    
def print_param_test(params) : 
    """Display the set of parameters contained in the received argument params"""
    
    print('Tested parameters\n')
    for idx, param in enumerate(params): 
        print ('-', idx+1, 'th parameter tested : ', param, '\n')
