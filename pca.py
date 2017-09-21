import numpy as np
from numpy import linalg as LA    

def calculate(files, scale):
    data = np.loadtxt(files, delimiter=',')
    call(data,scale)
    

def call(data,scale):
    data -= data.mean(axis=0)
    covmatrix = np.cov(data, rowvar=False)
    eigvalue, eigvector = LA.eig(covmatrix)
    print("Eigen Values are")
    print(eigvalue)
    print("Eigen Vectors are")
    print(eigvector)
    selpc = eigvector[:,:scale]
    newx = np.matmul(data,selpc)
    print(newx)
    return newx