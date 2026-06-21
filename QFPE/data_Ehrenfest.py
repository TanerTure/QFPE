'''
Generation of Ehrenfest data for given initial condition and timestamps
'''
import expectations as expect
import Ehrenfest
import Hamiltonians as Hams
import os
import params


import matplotlib.pyplot as plt
import numpy as np
import pickle

def main():
    
    funcs = [
            Hams.GQFPE_SH,
            Hams.CL,
            Hams.GQFPE_S,
            Hams.GQFPE
            ]
    param_name = "II"
    params_dict = params.params_sp[param_name]
    num_points = 801
    times = np.linspace(0, 40, num_points)
    
    for func in funcs:
        if func in Hams.TD_funcs:
            expectation_values = Ehrenfest.Ehrenfest_data_TD(times, func=func, params_name=param_name)
            dir_name=f"Data/Ehrenfest/{param_name}/{func.__name__}"
            os.makedirs(dir_name, exist_ok=True)
            with open(dir_name + "/" + expect.expect_funcs[0].__name__, "wb") as file:
                pickle.dump(np.ones(num_points), file)
            for i in range(len(expectation_values)):
                with open(dir_name + "/" + expect.expect_funcs[i+1].__name__,"wb") as file:
                    pickle.dump(expectation_values[i], file)

        else:
            Ehrenfest_funcs = Ehrenfest.Ehrenfest_funcs(func, param_name)
            dir_name=f"Data/Ehrenfest/{param_name}/{func.__name__}"
            os.makedirs(dir_name, exist_ok=True)
            with open(dir_name + "/" + expect.expect_funcs[0].__name__, "wb") as file:
                pickle.dump(np.ones(num_points), file) 
            for i, Ehrenfest_func in enumerate(Ehrenfest_funcs):
                with open(dir_name + "/" + expect.expect_funcs[i+1].__name__, "wb") as file:
                   pickle.dump(Ehrenfest_func(times), file)
        #fig.savefig(f"test_{func.__name__}")
   # #times = [sp.Rational(i*40, 200) for i in range(201)]
   # print(x_p_sol[0])
   # print(sp.re(x_p_sol[0].rewrite().simplify())
   # print(type(x_p_sol[0]))
   # x_func = sp.lambdify(t, x_p_sol[0])
   # p_func = sp.lambdify(t, x_p_sol[1])
   # fig, ax = plt.subplots(1,1)
   # ax.plot(times, x_func(times))
   # ax.plot(times, np.imag(x_func(times)))
   # fig.savefig("test_Ehrenfest_x")
   # fig, ax = plt.subplots(1,1)
   # ax.plot(times, p_func(times))
   # ax.plot(times, np.imag(p_func(times)))
   # print(np.max(np.imag(p_func(times))))
   # print(np.max(np.imag(x_func(times))))
   # fig.savefig("test_Ehrenfest_p")
    
    #lambda_vec = c_eqs.solve(x_diff)
    #print(lambda_vec)
    #  equations = [
    #          sp.Eq(x(t).diff(t), p(t)/m_s ),
    #          sp.Eq(p(t).diff(t), -m*w**2*x(t) -2j*A*hbar*p(t)),
    #          sp.Eq(xx(t).diff(t), xp_px(t)/m_s - 2*hbar**2*D_xx),
    #          sp.Eq(pp(t).diff(t), -m*w**2*xp_px(t) - 4j*A*hbar*pp(t) - 2*hbar**2*D_pp),
    #          sp.Eq(xp_px(t).diff(t), -2*m*w**2*xx(t) + 2*pp(t)/m_s -2j*A*hbar*xp_px(t) + 2*hbar**2*D_px )
    #          ]
    #  for eq in equations:
    #      print(eq)
    

    
if __name__ == "__main__":
    main()
