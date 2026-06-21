'''
Script for generating coordinate plots
'''
import numpy as np
import pickle
import os
import time


import expectations as expect
import Hamiltonians as Hams
import plotting_funcs as plot
import SHO

def main():
    funcs = [
            #Hams.GQFPE,
            Hams.GQFPE_S,
            Hams.GQFPE_SH,
            Hams.CL,
            ]
    param_names = ["II"]
    times = np.linspace(0, 40, 201)
    rho_points = [0, 67, 134, 200] 
    #rho_points = [0, 4, 8, 12]
    x_left_limit = -10
    x_right_limit = 10
    num_points = 512
    
    
    x_vals = np.linspace(x_left_limit, x_right_limit, num_points)
    time_points = [times[point] for point in rho_points]
    print(time_points)
    for name in param_names:
        rho_coords = []
        for func in funcs:
            os.makedirs(f"Data/rho_coord/{name}/{func.__name__}",exist_ok=True)
            rho_coords_func = []
            with open(f"Data/expectation_values/{name}/{func.__name__}/rho", "rb") as file:
                rhos = pickle.load(file)
            
            for rho_point in rho_points:
                try:
                    with open(f"Data/rho_coord/{name}/{func.__name__}/{rho_point}", "rb") as file:
                        rho_coords_func.append(pickle.load(file))
                except FileNotFoundError as error:
                    rho_coords_func.append(SHO.to_coordinate_space(rhos[rho_point], x_left_limit, x_right_limit, num_points))
                    with open(f"Data/rho_coord/{name}/{func.__name__}/{rho_point}", "wb") as file:
                        pickle.dump(rho_coords_func[-1], file)
            rho_coords.append(rho_coords_func)
        print("data complete for plot") 
        plot.plot_rho_coord(time_points, rho_coords, x_vals, x_vals, real = True, save_name =f"{name}_longtime")
        plot.plot_rho_coord(time_points, rho_coords, x_vals, x_vals, real = False, save_name=f"{name}_longtime")  
        
    

if __name__ == "__main__":
    main()


