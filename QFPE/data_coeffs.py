'''
Plot time-dependent coefficients for GQFPE, such as :math:`R_pq(t), R_pp(t), R_qq(t)`
'''
import Hamiltonians as Hams
import params

import matplotlib.pyplot as plt
import numpy as np


def main():
    times = np.linspace(0, 40, 401)
   # param_names = ["IV","I", "III"]
    param_names = ["I","II"]
    colors = ["k.", "b-", "r-."]
    fig, ax = plt.subplots(4, 1, figsize=(12, 6), sharex=True)
    for i in range(len(param_names)):
        param_name = param_names[i]
        param_values = params.params[param_name]
        w_c = param_values["w_c"]
        T = param_values["T"]
        gamma = param_values["gamma"]
        gamma_s = gamma/w_c
        R_pq_vals, R_pp_vals, R_qq_vals, Gamma_vals = [], [], [],[]
        for time in times:
            R_pq_vals.append(Hams.R_pq(time, w_c=w_c, T=T, gamma_s=gamma_s))
            R_pp_vals.append(Hams.R_pp(time, w_c=w_c, T=T, gamma_s=gamma_s))
            R_qq_vals.append(Hams.R_qq(time, w_c=w_c, T=T, gamma_s=gamma_s))
            Gamma_vals.append(Hams.Gamma(time, w_c=w_c, T=T, gamma_s=gamma_s)) 
        Dekker_cond = [4 * R_pp_vals[j] * R_qq_vals[j] - R_pq_vals[j]**2 - Gamma_vals[j]**2 for j in range(len(R_pq_vals))]
        for k in range(len(Dekker_cond)):
            if Dekker_cond[k] < 0:
                print(f"for T ={T}, fails positivity condition at t={times[k]}")
        ax[0].plot(times, R_pq_vals, colors[i])
        ax[1].plot(times, R_qq_vals, colors[i])
        ax[2].plot(times, R_pp_vals, colors[i])
        ax[3].plot(times, Dekker_cond, colors[i])
    fig.savefig("coeffs", dpi=300, bbox_inches="tight")
   # T = 100
   # gamma_s = .1
   # hbar = 1
   # m = 1
   # coeff_D_pq_SH = 2 * T * gamma_s/hbar**2 * (1 - 4*gamma_s)/(1-2*gamma_s)**2
   # coeff_D_pp_SH = -2*m*gamma*T/hbar**2 * (1 - 6 * gamma_s + 10 * gamma_s**2)/(1-2*gamma_s)**2
   # coeff_D_qq_SH = - gamma_s * T/(m * hbar **2 * w_c * (1 - 2*gamma_s)**2)
   # 
   # coeff_D_pq_S = gamma/hbar * Hams.R_pq(1000, w_c = w_c, T = T, gamma_s = gamma_s)
   # coeff_D_pp_S = -m * gamma**2/hbar * Hams.R_pp(1000, w_c = w_c, T = T, gamma_s = gamma_s)
   # coeff_D_qq_S = -1/(m*hbar) * Hams.R_qq(1000, w_c = w_c, T = T, gamma_s = gamma_s)

   # print(f"D_pq coefficient SH:{coeff_D_pq_SH} S:{coeff_D_pq_S}") 
   # print(f"D_pp coefficient SH:{coeff_D_pp_SH} S:{coeff_D_pp_S}") 
   # print(f"D_qq coefficient SH:{coeff_D_qq_SH} S:{coeff_D_qq_S}") 

if __name__ == "__main__":
    main()
