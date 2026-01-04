'''
Create initial condition for a given parameter set and 

'''
import numpy as np
import pickle

import params
import SHO


def main():
    names = ["I"]
    x_vals = np.linspace(-20, 20, 2048)
    y_vals = np.linspace(-20, 20, 2048)
    X, Y = np.meshgrid(x_vals, y_vals)

    for name in names:
        sys_params = params.params[name]
        B = 1/sys_params["T"]
        print(B, "is B")
        w = sys_params["w"]
        m = sys_params["m"]
        n = sys_params["n"]
        rho_coord = SHO.thermal_function(B, w, X - 2, Y - 2, m)
        rho = SHO.to_basis_set(n, x_vals, y_vals, rho_coord)  
        print(np.einsum("ii->", rho))
        with open(f"Data/initial_condition/{name}" ,"wb") as file:
            pickle.dump(rho, file)

if __name__ == "__main__":
    main()
