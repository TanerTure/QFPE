'''
Implementation of arnoldi iteration and other matrix exponentials
'''

import scipy
import numpy as np
import numba

#@numba.jit(nopython=True)
# def H(ndarray):
#     return np.conj(ndarray.T)
@numba.jit
def e(i=0,n=50):
    unit_vec = np.zeros(n,dtype=np.complex128)
    unit_vec[i] = 1
    return unit_vec
@numba.jit
def arnoldi_iteration(A,b,n=50):
    #n int : number of iterations
    Q_n = np.zeros((A.shape[0],n),dtype=np.complex128)
    Q_n[:,0] = b/(np.linalg.norm(b))
    H_n = np.zeros((n,n),dtype=np.complex128)
    for i in range(n):
        A_n = A@Q_n[:,i]
        #h_i = np.zeros(i+1,dtype=np.complex128)
        for j in range(i+1):
            H_n[j,i] = Q_n[:,j].T.conj()@A_n  
            A_n -= H_n[j,i]*Q_n[:,j]
        if (i==(n-1)):
            pass
            #h_last = np.linalg.norm(A_n)
            #Q_n[:,i+1] = A_n/h_last
        else:
            H_n[i+1,i] =  np.linalg.norm(A_n)
            try:
                Q_n[:,i+1] = A_n/H_n[i+1,i]
            except:
                return Q_n,H_n
    return Q_n,H_n
#@numba.jit(forceobj=True)
def arnoldi(A,b,n=75):
    Q,H = arnoldi_iteration(A,b.reshape(-1),n=n)
        #print(n)
    return (Q@scipy.linalg.expm(H)@e(0,n=n)*np.linalg.norm(b)).reshape(-1,1)

@numba.jit(forceobj=True)
def arnoldi_2(A,b,n=75):
    Q,H = arnoldi_iteration(A,b.reshape(-1),n=n)
    return (Q@taylor(H,e(0,n=n),n=n)*np.linalg.norm(b)).reshape(-1,1)

# def arnoldi(A,b,n=100):
#     if(n > A.shape[0]):
#         n=A.shape[0]
#         Q,H = arnoldi_iteration(A,b.reshape(-1),n=A.shape[0])
#         #print(n)
#     else:
#         Q,H = arnoldi_iteration(A,b.reshape(-1),n=n)
#         #print(n)
#     return (Q@scipy.linalg.expm(H)@e(0,n=n)*np.linalg.norm(b)).reshape(-1,1)


@numba.jit(forceobj=True)
def taylor(A,b,n=20):
    result =b.copy()
    storage = A@b
    for i in range(1,n):
        result += storage *1/np.complex128(np.math.factorial(i))
        storage = A@storage
    return result

# def taylor(A,b,n=20):
#     result = b.copy()
#     #storage = A@b
#     for i in range(1,n):
#         result += storage *1/np.complex128(np.math.factorial(i))
#         storage = A@storage
#     return result

#@numba.jit
def scipy_expm(A,b):
    return scipy.linalg.expm(A)@b

