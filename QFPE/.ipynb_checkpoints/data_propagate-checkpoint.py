'''
Given a parameter name, uses the corresponding initial condition data and propagates using ME and RK45

'''
import numpy as np
import pickle

import Hamiltonians as Hams
import params
import propagate

def main():
    param_names = ["II"]
    funcs = [Hams.GQFPE]
    for name in param_names:
        with open(f"Data/initial_condition/{name}", "rb") as file:
            rho = pickle.load(file)
        for func in funcs:
            propagate.prop_RK45(func, 0, 10, rho, None, params.params[name])



if __name__ == "__main__":
    main()
