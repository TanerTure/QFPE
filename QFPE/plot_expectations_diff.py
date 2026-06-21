
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
    parameter_names = ["II"]
    funcs = [#Hams.GQFPE_int
            #Hams.GQFPE,
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
                        expect_vals_analytical = expect_vals_analytical[::4]
                        print(type(expect_vals_prop[0]))
                        print(expect_vals_prop[0].shape)
                        print(expect_vals_analytical[0].shape)
                        print(len(expect_vals_prop), "is len prop")
                        print(len(expect_vals_analytical), "is len analytical")
                        #if(expect_func == 
                        #expectations.append([expect_vals_prop[i] - expect_vals_analytical[i]for i in range(0, len(expect_vals_analytical), 4)])
                        expectations.append([expect_vals_prop[i] - expect_vals_analytical[i]for i in range(len(expect_vals_analytical))])
                        
            #else:
            #    with open(directory_name + f"{name}/{func.__name__}/{expect_func.__name__}", "rb") as file:
                #with open(f"Data/expectation_values/{name}/{func.__name__}/{expect_func.__name__}", "rb") as file:
            #        expectations.append(pickle.load(file))
                #plt.plot(expectations[-1])
                #print(expectations[-1])
                #plt.show()
            #all_expectations.append(expectations)
            plot.plot_expectations(times, [expectations], save_name=f"diff_{name}_{func.__name__}", colors=[plot.func_to_color[func]])
if __name__ == "__main__":
    main()


