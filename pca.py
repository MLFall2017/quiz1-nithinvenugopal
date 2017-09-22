import numpy as np
from numpy import linalg as LA    

def calculate(files, scale):
    data = np.genfromtxt(files, delimiter=',')
    data = data[1:,:]
    call(data,scale)
    

def call(data,scale):
    data = data - data.mean(axis=0)
    varx = np.var(data[:,0])
    print("--------Variance of X--------")
    print(varx)
    vary = np.var(data[:,1])
    print("\n --------Variance of Y--------")
    print(vary)    
    varz = np.var(data[:,2])
    print("\n --------Variance of Z--------")
    print(varz)
    covxy = np.cov(data[:,0],data[:,1], rowvar=False)
    print("\n --------Covariance of x and y--------")
    print(covxy[0,1])
    covyz = np.cov(data[:,1],data[:,2],rowvar=False)
    print("\n --------Covariance of y and z--------")
    print(covyz[0,1])
    
    
    covmatrix = np.cov(data, rowvar=False)
    eigvalue, eigvector = LA.eig(covmatrix)
    index = eigvalue.argsort()[::-1]   
    eigvalue = eigvalue[index]
    eigvector = eigvector[:,index]
    print("\n --------Eigen Values are--------")
    print(eigvalue)
    print("\n --------Eigen Vectors(Principal Components) are--------")
    print(eigvector)
    selpc = eigvector[:,:scale]
    newx = np.matmul(data,selpc)
    print("\n --------Transformed Matrix is--------")
    print(newx)
    
calculate('dataset_1.csv',3)