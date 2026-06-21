'''
Given a parameter name, uses the corresponding initial condition data and propagates using ME and RK45

'''

import numpy as np
import os
import pickle
import scipy
import shutil
import time

import expectations as expect
import Hamiltonians as Hams
import params
import propagate


def main():
    param_names = ["II"]
    TD_funcs = Hams.TD_funcs
    funcs = [#Hams.GQFPE_int,
            #Hams.GQFPE,
           Hams.GQFPE_S,
           Hams.GQFPE_SH,
           Hams.CL
            ]
    expect_funcs = expect.expect_funcs
    num_points = 201  
    t_final = 40
    t_initial = 0
    
    for name in param_names:
        with open(f"Data/initial_condition/{name}", "rb") as file:
            rho = pickle.load(file)
      #  with open(f"test_file", "rb") as file:
      #      idx = "1"
      #      t, rho = pickle.load(file)
      #      n = round(np.sqrt(rho.shape[0]))
      #      rho = rho.reshape((n,n))
      #      print(t, t_initial, t_final)
      #      for i in range(len(expect_funcs)):
      #          expect_func = expect_funcs[i]
      #          shutil.copy(
      #                  f"Data/expectation_values/{name}/{Hams.GQFPE.__name__}/{expect_func.__name__}",
      #                  f"Data/expectation_values/{name}/{Hams.GQFPE.__name__}/{expect_func.__name__ + '_' + idx}" 
      #          )
        for func in funcs:
            time_1 = time.time()
            if func in TD_funcs:
                expectation_values = propagate.prop_CFME(func, t_initial, t_final, rho, expect_funcs, 
                                            params.params[name], num_points = num_points)
           #     expectation_values = propagate.prop_RK45(func, t_initial, t_final, rho, expect_funcs,
           #                                 params.params[name])
            elif func in Hams.int_funcs:
                expectation_values = propagate.prop_CFME_int(func, Hams.int_funcs[func], t_initial, t_final, rho,
                        expect_funcs, params.params[name], num_points=num_points)
            else:
                expectation_values = propagate.prop_time_independent(func, t_initial, t_final, rho, expect_funcs,
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
