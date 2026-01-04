'''
Propagates the density matrix forward in time, either using Magnus Expansion or `scipy.integrate.RK45`
'''
import expm
import SHO


import functools
import numba
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

def prop_CFME(func, t_initial, t_final, rho_initial, measurements, params, num_points=401):
    n = rho_initial.shape[0]
    params_func = functools.partial(func, **params)
    times = np.linspace(t_initial, t_final, num_points)
    rho_initial = rho_initial.reshape(-1, 1)
    dt = times[1] - times[0]
    expectation_values = [[0] for _ in range(len(measurements))]
    for i in range(num_points - 1):
        if i % 5 == 0:
            print(i)
        rho_initial = CFME(
                dt,
                (params_func(times[i]), params_func(times[i]+dt/2), params_func(times[i+1])),
                rho_initial
                )
        for j in range(len(measurements)):
            expectation_values[j].append(measurements[j](rho_initial.reshape((n,n))))
    return expectation_values

def prop_time_independent(func, t_initial, t_final, rho_initial, measurements, params, num_points=401):
    n = rho_initial.shape[0]
    params_func = functools.partial(func, **params)
    times = np.linspace(t_initial, t_final, num_points)
    rho_initial = rho_initial.reshape(-1, 1)
    dt = times[1] - times[0]
    expectation_values = [[0] for _ in range(len(measurements))]
    for i in range(num_points - 1):
        if i % 5 == 0:
            print(i)
        rho_initial = expm.arnoldi(dt*params_func(), rho_initial)
        for j in range(len(measurements)):
            expectation_values[j].append(measurements[j](rho_initial.reshape((n,n))))
    return expectation_values


def CFME(dt, func_points, rho):
    #return expm.arnoldi(dt/12*(-func_points[0] +4*func_points[1] + 3*func_points[2]))
    rho_1 = expm.arnoldi(dt/12*(3*func_points[0] + 4*func_points[1] - func_points[2]), rho)
    return expm.arnoldi(dt/12 * (-func_points[0] + 4*func_points[1] + 3 *func_points[2]), rho_1)

        
        

