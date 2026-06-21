
'''
Plots expectation values for given parameter set
'''
import expectations as expect
import Hamiltonians as Hams
import plotting_funcs as plot

import matplotlib.pyplot as plt
import numpy as np
import pickle

def main():
    times = np.linspace(0, 40, 201)
    times_Ehrenfest = np.linspace(0, 40, 801)
    parameter_names = ["I"]
    funcs = [#Hams.GQFPE_int
            Hams.GQFPE,
            Hams.GQFPE_S,
            Hams.GQFPE_SH,
            Hams.CL,
            ]
    expect_funcs = expect.expect_funcs
    
    for name in parameter_names:
        for func in funcs:
            expectations = []
            expectations_Ehrenfest = []
            for expect_func in expect_funcs:
                directory_name = "Data/Ehrenfest/"
                with open(directory_name + f"{name}/{func.__name__}/{expect_func.__name__}", "rb") as file:
                    expectations_Ehrenfest.append(pickle.load(file))
                directory_name = "Data/expectation_values/"
                with open(directory_name + f"{name}/{func.__name__}/{expect_func.__name__}", "rb") as file:
                    expectations.append(pickle.load(file))
                #plt.plot(expectations[-1])
                #print(expectations[-1])
                #plt.show()
            plot.plot_expectations_Ehrenfest(times, times_Ehrenfest, expectations, expectations_Ehrenfest, func=func, save_name=f"{func.__name__}_{name}")
if __name__ == "__main__":
    main()


