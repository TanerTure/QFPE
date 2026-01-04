import numpy as np
import sympy as sp

F = sp.symbols('F0:3' , cls=sp.Function)
G = sp.symbols(r'G0:3', cls=sp.Function)
R_pq, R_pp, R_qq, Gamma, G_1, r_m, m_e = sp.symbols(r'\mathcal{R}_{pq} \mathcal{R}_{pp} \mathcal{R}_{qq} \Gamma \tilde{G}_1 r_m m_e', cls=sp.Function)
T_s, t, t_s, g_s, m, w_c = sp.symbols(r'\T_s t t_s \gamma_s  m \omega_c', constant=True, real=True, positive=True)
F_1 = sp.symbols(r'\tilde{F}_1', cls=sp.Function)
n = sp.symbols('n', constant=True, real=True, positive=True)
W = sp.Wild("W")

func_defs = {
        R_pq(W): 1/r_m(W)*G_1(W)-2*g_s/r_m(W)**2*G[2](W*w_c)*F_1(W*w_c),
        Gamma(W):  F_1(W*w_c)/r_m(W),
        r_m(W) : m_e(W)/m,
        m_e(W) : m - 2*m*g_s*F[2](W*w_c),
        F[0](W) : 1-sp.exp(-W),
        F[1](W) : 1-(1+W)*sp.exp(-W),
        F[2](W) : 1- (1+W+W**2/2)*sp.exp(-W),
        F_1(W) : 1 - (1 + W - W**2/2)*sp.exp(-W),
        G[0](W): sp.cot(1/(2*T_s))*F[0](W)+ 4*T_s* sp.Sum(F[0](2*sp.pi*n*W)/((2*sp.pi*n*T_s)**2-1),
                                                        (n, 0, sp.oo)),
        G[1](W): sp.cot(1/(2*T_s))*F[1](W)+ 4*T_s* sp.Sum(F[1](2*sp.pi*n*W)/((2*sp.pi*n*T_s)*(2*sp.pi*n*T_s)**2-1), 
                                                        (n, 0, sp.oo)) ,
        G[2](W): sp.cot(1/(2*T_s))*F[2](W)+ 4*T_s* sp.Sum(F[2](2*sp.pi*n*W)/((2*sp.pi*n*T_s)**2*(2*sp.pi*n*T_s)**2-1),
                                                        (n, 0, sp.oo)),
        G_1(W): sp.cot(1/(2*T_s))*F_1(W)+ 4*T_s* sp.Sum(F_1(2*sp.pi*n*W)/((2*sp.pi*n*T_s)*(2*sp.pi*n*T_s)**2-1),
                                                        (n, 0, sp.oo))
        }
def reduce_term(term, func_defs):
    for func_def in func_defs:
        term = term.replace(func_def, func_defs[func_def])
    return term

for key in list(func_defs.keys()):
    for i in range(2):
        func_defs[key] = reduce_term(func_defs[key], func_defs)

def G_0(t_s_np, T_s_np):
    subs_dict = { 
            W: t_s_np,
            T_s:T_s_np
            }
    func = func_defs[G[0](W)].subs(subs_dict)
    return np.float64(func.evalf(n=16))


    
                
    
