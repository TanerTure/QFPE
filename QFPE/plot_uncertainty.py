
'''
Plots generalized uncertainty for given parameter set
'''
import expectations as expect
import Hamiltonians as Hams
import plotting_funcs as plot

import matplotlib.pyplot as plt
import numpy as np
import pickle

def main():
    Ehrenfest = False
    if Ehrenfest:
        num_points = 801
    else:
        num_points = 201
    times = np.linspace(0, 40, num_points)
    parameter_names = ["I"]
    funcs = [#Hams.GQFPE_int
            Hams.GQFPE,
            Hams.GQFPE_S,
            Hams.GQFPE_SH,
            Hams.CL,
            ]
    expect_funcs = expect.expect_funcs
    
    for name in parameter_names:
        all_expectations = []
        for func in funcs:
            expectations = []
            for expect_func in expect_funcs:
                if Ehrenfest:
                    directory_name = "Data/Ehrenfest/"
                else:
                    directory_name = "Data/expectation_values/"
                    
                if name == "II" and func == Hams.GQFPE:
                    directory_name = "Data/Ehrenfest/"
                    with open(directory_name + f"{name}/{func.__name__}/{expect_func.__name__}", "rb") as file:
                        expect_vals = pickle.load(file)
                        expectations.append([expect_vals[i] for i in range(0, len(expect_vals), 4)])
                else:
                    with open(directory_name + f"{name}/{func.__name__}/{expect_func.__name__}", "rb") as file:
                #with open(f"Data/expectation_values/{name}/{func.__name__}/{expect_func.__name__}", "rb") as file:
                        expectations.append(pickle.load(file))
                #plt.plot(expectations[-1])
                #print(expectations[-1])
                #plt.show()
            all_expectations.append(expectations)
            plot.plot_uncertainty(times, all_expectations, save_name=f"{name}_{func.__name__}")
            all_expectations = []
        #plot.plot_uncertainty(times, all_expectations, save_name=f"compare_all_{name}")
                
if __name__ == "__main__":
    main()

    
