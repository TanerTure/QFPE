'''
Propagates the density matrix forward in time, either using Magnus Expansion or `scipy.integrate.RK45`
'''
import expectations as expect
import expm
import Hamiltonians as Hams
import SHO

import functools
import numba
import numpy as np
import pickle
import scipy
import time


def get_prop_func(func):
    if func in Hams.TD_funcs:
        return prop_CFME
    if func in Hams.int_funcs:
        return prop_CFME_int
    return prop_time_dependent

def prop_RK45(func, t_initial, t_final, rho_initial, measurements, params):
    step_time = 0
    expectation_values = [[] for _ in range(len(measurements))]
    n = rho_initial.shape[0]
    params_func = functools.partial(func, **params)
    new_func = lambda t,y : (params_func(t) @ y.reshape(-1,1)).reshape(-1) 
    propagator = scipy.integrate.RK45(new_func, t_initial, rho_initial.reshape(-1), t_final, max_step = 0.1)
    #propagator = scipy.integrate.RK45(new_func, t_initial, rho_initial.reshape(-1), t_final)
    for j in range(len(measurements)):
        expectation_values[j].append(measurements[j](rho_initial.reshape((n,n))))
    num_steps = 0 
    while propagator.status != "finished":
        num_steps += 1
        time_1 = time.time()
        propagator.step()
        step_time += time.time() - time_1
        print(propagator.t, "/", t_final)
        for j in range(len(measurements)):
            expectation_values[j].append(measurements[j](propagator.y.reshape((n,n))))
        #print(propagator.t, np.einsum("ii->", propagator.y.reshape((n,n))))
        #print(np.einsum("ii->", SHO.x(params["n"], w=params["w"], hbar=params["hbar"])
        #    @propagator.y.reshape((n,n))))
    print(num_steps)
    return expectation_values

def prop_CFME(func, t_initial, t_final, rho_initial, measurements, params, num_points=401):
    matrix_calc_time = 0
    exponential_time = 0
    expectations_time = 0
    matrix_write_time = 0
    n = rho_initial.shape[0]
    params_func = functools.partial(func, **params)
    times = np.linspace(t_initial, t_final, num_points)
    rho_initial = rho_initial.reshape(-1, 1)
    dt = times[1] - times[0]
    expectation_values = [[] for _ in range(len(measurements))]
    
    time_1 = time.time()
    for j in range(len(measurements)):
        expectation_values[j].append(measurements[j](rho_initial.reshape((n,n))))
    expectations_time += time.time() - time_1
    time_1 = time.time()
    func_vals = [params_func(times[0]), params_func(times[0] + dt/2), params_func(times[1])]
    matrix_calc_time += time.time() - time_1
    for i in range(num_points - 1):
        print(i,"/", num_points - 1)
        time_1 = time.time()
        rho_initial = CFME(
                dt,
                (func_vals[0], func_vals[1], func_vals[2]),
                rho_initial
                )
        exponential_time += time.time() - time_1
        time_1 = time.time()
        try:
            for j in range(len(measurements)):
                expectation_values[j].append(measurements[j](rho_initial.reshape((n,n))))
        except ValueError as error:
            print(error)
            last_saved = 0
            if i == i//10:
                last_saved = i - 10
            else:
                last_saved = i - i%10
            print(f"last working i was {last_saved}, corresponding time is {times[last_saved]}. Continue calculation with smaller timestep")
            print("returning expectations up to times[last_saved]")
            for j in range(len(expectation_values)):
                expectation_values[j] = [expectation_values[j][k] for k in range(last_saved+1)]
            return expectation_values
            
        expectations_time += time.time() - time_1
        time_1 = time.time() 
        if i % 10 == 0:
            with open("test_file", "wb") as file:
                pickle.dump([times[i], rho_initial.reshape((n,n))], file)
        matrix_write_time += time.time() - time_1
        time_1 = time.time()
        func_vals[0] = func_vals[2]
        func_vals[1] = params_func(times[i+1] + dt/2)
        func_vals[2] = params_func(times[i+1] + dt)
        matrix_calc_time += time.time() - time_1
    print(matrix_write_time, "is time to dump to file")
    print(matrix_calc_time, "is time to calculate matrices")
    print(expectations_time, "is time to calculate expectations")
    print(exponential_time, "is time to calculate exponential")
    return expectation_values

def prop_CFME_int(func, H_func, t_initial, t_final, rho_initial, measurements, params, num_points=401):
    '''
    Propagates `func` in interaction picture, defined by Hamiltonian `H_func`, using CFME
    '''
    matrix_calc_time = 0
    exponential_time = 0
    expectations_time = 0
    matrix_write_time = 0
    n = rho_initial.shape[0]
    params_func = functools.partial(func, **params)
    params_H_func = functools.partial(H_func, **params)
    times = np.linspace(t_initial, t_final, num_points)
    rho_initial = rho_initial.reshape(-1, 1)
    dt = times[1] - times[0]
    expectation_values = [[] for _ in range(len(measurements))]

    time_1 = time.time()
    for j in range(len(measurements)):
        expectation_values[j].append(measurements[j](rho_initial.reshape((n,n)), np.eye(n, dtype=np.complex128)))
    expectations_time += time.time() - time_1

    H_func_vals = [params_H_func(times[0] + tau) for tau in [0, dt/4, dt/2, 3*dt/4, dt]]
    U_func_vals = [np.eye(n, dtype=np.complex128), CFME(dt/2, H_func_vals[:3], np.eye(n, dtype=np.complex128), expm_func=expm.scipy_expm)]
    U_func_vals.append(CFME(dt/2, H_func_vals[2:], U_func_vals[1], expm_func=expm.scipy_expm))
    
    time_1 = time.time()
    func_vals = [params_func(times[0] + data[0], U_func_vals[data[1]]) for data in [(0, 0), (dt/2, 1), (dt, 2) ]] 
    matrix_calc_time += time.time() - time_1
    
    for i in range(num_points - 1):
        print(i,"/", num_points - 1)
        time_1 = time.time()
        rho_initial = CFME(
                dt,
                (func_vals[0], func_vals[1], func_vals[2]),
                rho_initial
                )
        exponential_time += time.time() - time_1
        time_1 = time.time()
        try:
            for j in range(len(measurements)):
                expectation_values[j].append(measurements[j](rho_initial.reshape((n,n)), U_func_vals[-1]))
        except ValueError as error:
            print(error)
            last_saved = 0
            if i == i//10:
                last_saved = i - 10
            else:
                last_saved = i - i%10
            print(f"last working i was {last_saved}, corresponding time is {times[last_saved]}. Continue calculation with smaller timestep")
            print("returning expectations up to times[last_saved]")
            for j in range(len(expectation_values)):
                expectation_values[j] = [expectation_values[j][k] for k in range(last_saved+1)]
            return expectation_values
            
        expectations_time += time.time() - time_1
        time_1 = time.time()
        if i % 10 == 0:
            with open("test_file", "wb") as file:
                pickle.dump([times[i], rho_initial.reshape((n,n))], file)
        matrix_write_time += time.time() - time_1
        H_func_vals[0] = H_func_vals[4]
        for idx,tau in enumerate([dt/4, dt/2, 3*dt/4, dt]):
            H_func_vals[idx + 1] = params_H_func(times[i + 1] + tau)
        for U in U_func_vals:
            expect.check_unitary(U)
        U_func_vals[0] = U_func_vals[2] 
        U_func_vals[1] = CFME(dt/2, H_func_vals[:3], U_func_vals[0], expm_func=expm.scipy_expm)
        U_func_vals[2] = CFME(dt/2, H_func_vals[2:], U_func_vals[1], expm_func=expm.scipy_expm)
        time_1 = time.time()
        func_vals[0] = func_vals[2]
        func_vals[1] = params_func(times[i+1] + dt/2, U_func_vals[1])
        func_vals[2] = params_func(times[i+1] + dt, U_func_vals[2])
        matrix_calc_time += time.time() - time_1
    print(matrix_write_time, "is time to dump to file")
    print(matrix_calc_time, "is time to calculate matrices")
    print(expectations_time, "is time to calculate expectations")
    print(exponential_time, "is time to calculate exponential")
    return expectation_values

def prop_time_independent(func, t_initial, t_final, rho_initial, measurements, params, num_points=401):
    #time_1 = time.time()
    matrix_calc_time = 0
    expectations_time = 0
    exponential_time = 0

    n = rho_initial.shape[0]
    params_func = functools.partial(func, **params)
    times = np.linspace(t_initial, t_final, num_points)
    rho_initial = rho_initial.reshape(-1, 1)
    dt = times[1] - times[0]
    expectation_values = [[] for _ in range(len(measurements))]
    time_1 = time.time()
    for j in range(len(measurements)):
        expectation_values[j].append(measurements[j](rho_initial.reshape((n,n))))
    expectations_time += time.time() - time_1
    
    time_1 = time.time()
    func_val = dt * params_func()
    matrix_calc_time = time.time() - time_1
    for i in range(num_points - 1):
        print(i,"/", num_points - 1)
        time_1 = time.time()
        rho_initial = expm.arnoldi(func_val, rho_initial)
        exponential_time += time.time() - time_1
        time_1 = time.time()
        for j in range(len(measurements)):
            expectation_values[j].append(measurements[j](rho_initial.reshape((n,n))))
        expectations_time += time.time() - time_1
    #print(time_2 - time_1, "for this calculation")
    print(matrix_calc_time, "is time to calculate matrices")
    print(expectations_time, "is time to calculate expectations")
    print(exponential_time, "is time to calculate exponential")
    return expectation_values


def CFME(dt, func_points, rho, expm_func=expm.arnoldi):
    #return expm.arnoldi(dt/12*(-func_points[0] +4*func_points[1] + 3*func_points[2]))
    #print(func_points[0].shape)
    #print(rho.shape, "is rho shape")
    rho_1 = expm_func(dt/12*(3*func_points[0] + 4*func_points[1] - func_points[2]), rho)
    return expm_func(dt/12 * (-func_points[0] + 4*func_points[1] + 3 *func_points[2]), rho_1)

def CFME_matrix(dt, func_points):
    U_1 = scipy.linalg.expm(dt/12*(3*func_points[0] + 4*func_points[1] - func_points[2]))
    return scipy.linalg.expm(dt/12*(3*func_points[2] + 4*func_points[1] - func_points[0])) @ U_1
        

