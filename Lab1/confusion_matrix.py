from __future__ import print_function
from copy import deepcopy
import numpy as np

def get_confusion_matrix(predicted_labels, y_test, possible_labels, print_matrix=False, format_length=False):
    dict = {}
    for label in possible_labels:
        dict[label] = [0]*len(possible_labels)
    
    if not format_length:
        format_length = int(max(map(len,map(str, possible_labels))))+2
        
    for i, label in enumerate(y_test):
        predicted = predicted_labels[i]
        statisticsPredicted = dict[label]
        predicted_index = np.where(possible_labels == predicted)
        statisticsPredicted[predicted_index[0][0]]+=1

    if(print_matrix):
        
        print ('{:>{}s}'.format(" ", format_length+1), end="")
        for index, key in enumerate(sorted(dict.keys())):
            print ('{:>{}s}'.format(str(key), format_length), end="")
        print()
        for index, key in enumerate(sorted(dict.keys())):
            print ('{:>{}s}'.format(str(key), format_length), end="")
            print_dict_formatted(dict[key], format_length)

    matrix=[]
    for key in sorted(dict.keys()):
        matrix.append(np.array(dict[key]))

    return np.array(matrix)

def print_dict_formatted(array, format_length):
    print ("[",end="")
    for element in array:
        print ('{:>{}s}'.format(str(element), format_length), end="")
    print ("]")

def eval_AP(Y_ranked,Class):
    y_ranked_local = np.array(deepcopy(Y_ranked))
    loc_true = np.where(y_ranked_local == Class)
    loc_false = np.where(y_ranked_local != Class)
    y_ranked_local[loc_true] = 1
    y_ranked_local[loc_false] = 0
    numerator = np.sum([average_precision_at_index(y_ranked_local,index)*element for index,element in enumerate(y_ranked_local)])
    denominator = np.sum(y_ranked_local)
    if denominator == 0:
        return 0.0
    return numerator/denominator

def average_precision_at_index(Yranked,index):
    if index > len(Yranked):
        raise Exception("Index out of range")
    Ypredicted = np.zeros(Yranked.shape)
    Ypredicted[index:]=1
    TP,TN,FP,FN = get_statistical_data(Ypredicted,Yranked)
    if TP == 0 and FP == 0:
        return 0.0
    return get_precision(TP,FP)

def get_statistical_data(Y,Y_):
    correct_classification = Y[np.where(Y==Y_)]
    incorrect_classification = Y[np.where(Y!=Y_)]
    TP = len(correct_classification[np.where(correct_classification==1)])
    TN = len(correct_classification[np.where(correct_classification==0)])
    FP = len(incorrect_classification[np.where(incorrect_classification==1)])
    FN = len(incorrect_classification[np.where(incorrect_classification==0)])
    return TP,TN,FP,FN

def get_precision(TP,FP):
    return (1.*TP)/(TP+FP)

def eval_perf_multi(conf_matrix):
    precision = []
    recall = []
    for i in range(len(conf_matrix)):
        element = conf_matrix[i][i]
        precision_sum = 0.
        recall_sum = 0.
        for j in range(len(conf_matrix)):
            precision_sum += conf_matrix[j][i]
            recall_sum += conf_matrix[i][j]
        precision.append(element/precision_sum)
        recall.append(element/recall_sum)
    return precision, recall