import numpy as np
from numpy import linalg as LA    

def calculate(files, scale):
    data = np.genfromtxt(files, delimiter=',')
    data = data[1:,:]
    call(data,scale)
    

def call(data,scale):
    data = data - data.mean(axis=0)
    covmatrix = np.cov(data, rowvar=False)
    eigvalue, eigvector = LA.eig(covmatrix)
    index = eigvalue.argsort()[::-1]   
    eigvalue = eigvalue[index]
    eigvector = eigvector[:,index]
    print("--------Eigen Values are--------")
    print(eigvalue)
    print("--------Eigen Vectors are--------")
    print(eigvector)
    selpc = eigvector[:,:scale]
    newx = np.matmul(data,selpc)
    print("--------Transformed Matrix is--------")
    print(newx)