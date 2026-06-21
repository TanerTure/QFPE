'''
Plots error due to Taylor series approximation vs. Arnoldi iteration
'''
import plotting_funcs as plot

import matplotlib.pyplot as plt
import pickle



def main():
    param_names = ["I"]
    dts = [0.1, 0.5, 1]
    num_terms = [1, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140]
    save_name = "taylor_series_error"

    for param_name in param_names:
        with open(f"Data/taylor_error/{param_name}/taylor", "rb") as file:
            taylor_res = pickle.load(file)
        with open(f"Data/taylor_error/{param_name}/arnoldi", "rb") as file:
            arnoldi_res = pickle.load(file)
        plot.plot_taylor_error(num_terms, arnoldi_res, taylor_res, save_name=save_name+f"{param_name}")
    return

if __name__ == "__main__":
    main()
