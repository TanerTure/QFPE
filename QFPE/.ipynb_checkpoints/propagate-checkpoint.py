'''
Propagates the density matrix forward in time, either using Magnus Expansion or `scipy.integrate.RK45`
'''
import SHO

import functools
import numpy as np
import scipy


def prop_RK45(func, t_initial, t_final, rho_initial, measurements, params):
    n = rho_initial.shape[0]
    params_func = functools.partial(func, **params)
    new_func = lambda t,y : (params_func(t) @ y.reshape(-1,1)).reshape(-1) 
    propagator = scipy.integrate.RK45(new_func, t_initial, rho_initial.reshape(-1), t_final)
    while propagator.status != "finished":
        propagator.step()
        print(propagator.t, np.einsum("ii->", propagator.y.reshape((n,n))))
        print(np.einsum("ii->", SHO.x(params["n"], w=params["w"], hbar=params["hbar"])
            @propagator.y.reshape((n,n))))

        
        

