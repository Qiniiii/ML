from linear_regression import linear_regression_noreg,linear_regression_invertible, regularized_linear_regression, tune_lambda, mean_absolute_error,mapping_data
from data_loader import data_processing_linear_regression
import numpy as np
import pandas as pd

filename = 'winequality-white.csv'
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [3, 4]])
print(np.concatenate((a, b), axis=1))



print("\n======== Question 1.1 and Question 1.2 ========")
Xtrain, ytrain, Xval, yval, Xtest, ytest = data_processing_linear_regression(filename, False, False, 0)
print(Xtrain)
w = linear_regression_noreg(Xtrain, ytrain)
print("dimensionality of the model parameter is ", w.shape, ".", sep="")
print("model parameter is ", np.array_str(w))
mae = mean_absolute_error(w, Xtrain, ytrain)
print("MAE on train is %.5f" % mae)
mae = mean_absolute_error(w, Xval, yval)
print("MAE on val is %.5f" % mae)
mae = mean_absolute_error(w, Xtest, ytest)
print("MAE on test is %.5f" % mae)

print("\n======== Question 1.3 ========")
Xtrain, ytrain, Xval, yval, Xtest, ytest = data_processing_linear_regression(filename, True, False, 0)
w = linear_regression_invertible(Xtrain, ytrain)
print("dimensionality of the model parameter is ", w.shape, ".", sep="")
print("model parameter is ", np.array_str(w))
mae = mean_absolute_error(w, Xtrain, ytrain)
print("MAE on train is %.5f" % mae)
mae = mean_absolute_error(w, Xval, yval)
print("MAE on val is %.5f" % mae)
mae = mean_absolute_error(w, Xtest, ytest)
print("MAE on test is %.5f" % mae)

print("\n======== Question 1.4 ========")
Xtrain, ytrain, Xval, yval, Xtest, ytest = data_processing_linear_regression(filename, True, False, 0)
w = regularized_linear_regression(Xtrain, ytrain, 0.1)
print("dimensionality of the model parameter is ", w.shape, ".", sep="")
print("model parameter is ", np.array_str(w))
mae = mean_absolute_error(w, Xtrain, ytrain)
print("MAE on train is %.5f" % mae)
mae = mean_absolute_error(w, Xval, yval)
print("MAE on val is %.5f" % mae)
mae = mean_absolute_error(w, Xtest, ytest)
print("MAE on test is %.5f" % mae)


print("\n======== Question 1.5========")
Xtrain, ytrain, Xval, yval, Xtest, ytest = data_processing_linear_regression(filename, False, False, 0)
bestlambd = tune_lambda(Xtrain, ytrain, Xval, yval)
print("Best Lambda =  " + str(bestlambd))
w = regularized_linear_regression(Xtrain, ytrain, bestlambd)
print("dimensionality of the model parameter is ", len(w), ".", sep="")
print("model parameter is ", np.array_str(w))
mae = mean_absolute_error(w, Xtrain, ytrain)
print("MAE on train is %.5f" % mae)
mae = mean_absolute_error(w, Xval, yval)
print("MAE on val is %.5f" % mae)
mae = mean_absolute_error(w, Xtest, ytest)
print("MAE on test is %.5f" % mae)


print("\n======== Question 1.6.1 (power = 2) ========")
print("if your maaping function is correct, simplely change the 'power' value to see how MAE change when 'power' changes")
power = 2
Xtrain, ytrain, Xval, yval, Xtest, ytest = data_processing_linear_regression(filename, False, True, power)
bestlambd = tune_lambda(Xtrain, ytrain, Xval, yval)
print("Best Lambda =  ", bestlambd, sep="")
w = regularized_linear_regression(Xtrain, ytrain, bestlambd)
print("dimensionality of the model parameter is ", len(w), ".", sep="")
print("model parameter is ", np.array_str(w))
mae = mean_absolute_error(w, Xtrain, ytrain)
print("MAE on train is %.5f" % mae)
mae = mean_absolute_error(w, Xval, yval)
print("MAE on val is %.5f" % mae)
mae = mean_absolute_error(w, Xtest, ytest)
print("MAE on test is %.5f" % mae)

print("\n======== Question 1.6.2 (power = higher)========")
print("if your maaping function is correct, simplely change the 'power' value to see how MSE change when 'power' changes")
power = 20
for i in range(2, 4):
    Xtrain, ytrain, Xval, yval, Xtest, ytest = data_processing_linear_regression(filename, False, True, i)
    bestlambd = tune_lambda(Xtrain, ytrain, Xval, yval)
    print('best lambd is ' + str(bestlambd))
    w = regularized_linear_regression(Xtrain, ytrain, bestlambd)
    mae = mean_absolute_error(w, Xtrain, ytrain)
    print('when power = ' + str(i), w)
    print("MAE on train is %.5f" % mae)
    mae = mean_absolute_error(w, Xval, yval)
    print("MAE on val is %.5f" % mae)
    mae = mean_absolute_error(w, Xtest, ytest)
    print("MAE on test is %.5f" % mae)




