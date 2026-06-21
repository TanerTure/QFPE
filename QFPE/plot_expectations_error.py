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
    num_points = 201
    times = np.linspace(0, 40, num_points)
    parameter_names = ["I"]
    funcs = [#Hams.GQFPE_int
            Hams.GQFPE,
            Hams.GQFPE_S,
            Hams.GQFPE_SH,
            Hams.CL,
            ]
    expect_funcs = expect.expect_funcs[:6]
    
    for name in parameter_names:
        all_expectations = []
        for func in funcs:
            if name == "II" and func == Hams.GQFPE:
                continue
            expectations = []
            for expect_func in expect_funcs:
                Ehren_dir_name = "Data/Ehrenfest/"
                prop_dir_name = "Data/expectation_values/"

                with open(Ehren_dir_name + f"{name}/{func.__name__}/{expect_func.__name__}", "rb") as file_analytical:
                    with open(prop_dir_name + f"{name}/{func.__name__}/{expect_func.__name__}", "rb") as file_prop:
                        expect_vals_analytical = pickle.load(file_analytical)
                        expect_vals_prop = pickle.load(file_prop)
                        expectations.append([expect_vals[i] for i in range(0, len(expect_vals), 4)])
                else:
                    with open(directory_name + f"{name}/{func.__name__}/{expect_func.__name__}", "rb") as file:
                #with open(f"Data/expectation_values/{name}/{func.__name__}/{expect_func.__name__}", "rb") as file:
                        expectations.append(pickle.load(file))
                #plt.plot(expectations[-1])
                #print(expectations[-1])
                #plt.show()
            all_expectations.append(expectations)
        plot.plot_expectations(times, all_expectations, save_name=f"compare_all_{name}")
if __name__ == "__main__":
    main()


