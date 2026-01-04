'''
Given a parameter name, uses the corresponding initial condition data and propagates using ME and RK45

'''
import numpy as np
import pickle
import time
import scipy

import expectations as expect
import Hamiltonians as Hams
import os
import params
import propagate
import time


def main():
    param_names = ["I"]
    TD_funcs = [Hams.GQFPE]
    funcs = [Hams.GQFPE,
            Hams.GQFPE_S,
            Hams.GQFPE_SH,
            Hams.CL
            ]
    expect_funcs = [
            expect.trace,
            expect.x,
            expect.p,
            expect.x_squared,
            expect.p_squared,
            expect.xp_px,
            ]
    num_points = 201
    for name in param_names:
        with open(f"Data/initial_condition/{name}", "rb") as file:
            rho = pickle.load(file)
        for func in funcs:
            time_1 = time.time()
            if func in TD_funcs:
                expectation_values = propagate.prop_CFME(func, 0, 40, rho, expect_funcs, 
                                            params.params[name], num_points = num_points)
            else:
                expectation_values = propagate.prop_time_independent(func, 0, 40, rho, expect_funcs,
                        params.params[name], num_points = num_points)
            time_2 = time.time()
            print(f"{time_2 - time_1} seconds for {func.__name__}") 
            os.makedirs(f"Data/expectation_values/{name}/{func.__name__}/", exist_ok = True)
            for i in range(len(expect_funcs)):
                expect_func = expect_funcs[i]
                with open(f"Data/expectation_values/{name}/{func.__name__}/{expect_func.__name__}", "wb") as file:
                    pickle.dump(expectation_values[i], file)
       

if __name__ == "__main__":
    main()
