import numpy as np

U = np.array([[-1,1],[1,-1]])
b1 = 0
w = np.array([1,1]).T
b2 = -0.1

X = [np.array([0,0]).T, np.array([0,1]).T, np.array([1,0]).T, np.array([1,1]).T]
for x in X:
    h =  np.maximum(U.T@x + b1, np.zeros((2,)))
    print("For x = ", x, "hidden layer output is ", h)
    f = w.T@h + b2

    if(f>=0):
        print("For x = ", x, "output is 1")
    else:
        print("For x = ", x, "output is 0")
