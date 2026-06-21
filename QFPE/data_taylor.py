'''
Produces plot showing difference between Taylor series and Arnoldi iteration followed by 
`scipy.linalg.expm`
'''
import functools
import numpy as np
import os
import pickle

import expm
import Hamiltonians as Hams
import params


def main():
    param_name = "II"
    equations = [Hams.CL]
    num_terms = [1, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140]
    #dts = [0.1, 1, 2]  
    dts = [0.1, 0.5, 1]
    num_dts = len(dts)
    with open(f"Data/initial_condition/{param_name}", "rb") as file:
        rho = pickle.load(file).reshape(-1, 1)
    for equation in equations:
        param_equation = functools.partial(equation, **params.params[param_name])
        taylor_res = [[] for _ in range(num_dts)]
        arnoldi_res = [[] for _ in range(num_dts)]
        for i in range(num_dts):
            A = param_equation() * dts[i]
            #print(np.linalg.norm(A, ord="fro"), "is matrix_norm")
            scipy_expm = expm.scipy_expm(A, rho)
            ref_norm = np.linalg.norm(scipy_expm, ord="fro")
            #print(ref_norm, "is ref_norm")
            for j in range(len(num_terms)):
                taylor = expm.taylor(A, rho, n=num_terms[j])
                arnoldi = expm.arnoldi(A, rho, n=num_terms[j])
                error_taylor = np.linalg.norm(np.abs(scipy_expm - taylor), ord="fro")/ref_norm
                error_arnoldi = np.linalg.norm(np.abs(scipy_expm - arnoldi), ord="fro")/ref_norm
                taylor_res[i].append(np.log10(error_taylor))
                arnoldi_res[i].append(np.log10(error_arnoldi))

    os.makedirs(f"Data/taylor_error/{param_name}", exist_ok=True)
    with open(f"Data/taylor_error/{param_name}/taylor", "wb") as file:
        pickle.dump(taylor_res, file)
    with open(f"Data/taylor_error/{param_name}/arnoldi", "wb") as file:
        pickle.dump(arnoldi_res, file)

if __name__ == "__main__":
    main()
