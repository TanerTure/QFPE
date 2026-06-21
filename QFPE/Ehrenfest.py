import expm
import Hamiltonians as Hams
import params
import propagate

import numpy as np
import sympy as sp

q, p, qp_pq, qq, pp = sp.symbols('q p qp_pq qq pp', cls=sp.Function)
D_pp, D_pq, D_qq = sp.symbols('D_pp D_pq D_qq', constant=True, real=True)
A = sp.symbols('A', constant=True, imaginary=True)
t = sp.symbols('t')
m_es, m, w, w_c, hbar, T, gamma = sp.symbols('m_es m w w_c hbar T gamma', constant=True, positive=True)
q_0, p_0, qq_0, pp_0, qp_pq_0 = sp.symbols('q_0 p_0 qq_0 pp_0 (qp+pq)_0', constant=True, real=True)
R_pp, R_pq, R_qq = sp.symbols('R_pp R_pq R_qq', constant=True, real=True)
gamma_s = gamma/w_c
CL_subs = {
        A: - sp.I*gamma/hbar,
        D_pq: 0,
        D_pp:-2*m*gamma*T/hbar**2,
        D_qq:0,
        m_es:m
        }
GQFPE_S_subs = {
        A: -sp.I * gamma/hbar* (1/(1-2*(gamma_s))),
        D_pq: gamma/hbar*R_pq,
        D_pp: -m*gamma**2/hbar * R_pp,
        D_qq: -R_qq/(m*hbar),
        m_es:m_es
        }
GQFPE_SH_subs = {
        A:-sp.I*gamma/hbar*(1/(1-2*(gamma_s))),
        D_pq:2*T*gamma_s/hbar**2 * (1-4*(gamma_s)) / (1-2*(gamma_s))**2,
        D_pp:-2*m*gamma*T/hbar**2 * (1 - 6*(gamma_s) + 10 * gamma_s**2)/(1-2*gamma_s)**2,
        D_qq:-gamma_s*T/(m*hbar**2*w_c*(1-2*gamma_s)**2),
        m_es:m_es
        }
def make_params(params):
    '''
    Converts keys of params dict from string to corresponding sympy symbol
    '''
    new_params = {}
    conv_dict = {
            "T": T,
            "gamma":gamma,
            "m":m,
            "w":w,
            "hbar":hbar,
            "w_c":w_c,

            }
    for var_name in params:
        try:
            new_params[conv_dict[var_name]] = params[var_name]
        except KeyError as e:
            (e)
    new_params[m_es] = new_params[m] *(1-2*new_params[gamma]/new_params[w_c])
    return new_params
 
def Ehrenfest_funcs(eq=Hams.CL,params_name="I"):
    params_subs = make_params(params.params_sp[params_name])
    eq_to_subs = {
        Hams.GQFPE_SH: GQFPE_SH_subs,
        Hams.GQFPE_S: GQFPE_S_subs,
        Hams.CL: CL_subs,
        }
    eq_subs = eq_to_subs[eq]
    if eq == Hams.GQFPE_S:
        w_c_np = params.params[params_name]["w_c"]
        T_np = params.params[params_name]["T"]
        gamma_s_np = params.params[params_name]["gamma"]/params.params[params_name]["w_c"]
        gamma = params.params[params_name]
        GQFPE_S_subs[R_pq] = Hams.R_pq(1000, T=T_np, w_c=w_c_np, gamma_s = gamma_s_np)
        GQFPE_S_subs[R_pp] = Hams.R_pp(1000, T=T_np, w_c=w_c_np, gamma_s = gamma_s_np)
        GQFPE_S_subs[R_qq] = Hams.R_qq(1000, T=T_np, w_c=w_c_np, gamma_s = gamma_s_np)

    A_eqs = sp.Matrix(
            [
                [0, 1/m_es, 0, 0, 0],
                [-m*w**2, -2*sp.I*A*hbar, 0, 0, 0],
                [0, 0, 0, 0, 1/m_es],
                [0, 0, 0, -4*sp.I*A*hbar, -m*w**2],
                [0, 0, -2*m*w**2, 2/m_es, -2*sp.I*A*hbar]
                ]
            )
    
    #A_eqs = A_eqs.subs(CL_subs)
    eigen_data = A_eqs.eigenvects()
    #for i in range(len(eigen_data)):
    #    (eigen_data[i][0], "is eigenvalue")
    #    (eigen_data[i][1], "is multiplicity")
    #    (eigen_data[i][2], "are the eigenvectors")
    
    inhom_vec = sp.Matrix(
            [
                [0, 0, -2*hbar**2*D_qq, -2*hbar**2*D_pp, 2*hbar**2*D_pq]
                ]
            )
    inhom_vec = inhom_vec.T
    inhom_vec = inhom_vec.subs(eq_subs)
    (inhom_vec, "is inhom_vec")
    q_inhom = A_eqs.solve(-inhom_vec)
    (q_inhom, "is q_inhom")
    
    pp_0_exact = hbar/(2*m*w) * (1 - sp.exp(-2*hbar*w/T))/(1-sp.exp(-hbar*w/T))**2
    q_vec_0_exact = {
            q_0:2,
            p_0:0,
            qq_0 : pp_0_exact + 4,
            pp_0 : pp_0_exact,
            qp_pq_0:0,
            }
    q_vec_0 = sp.Matrix(
            [
                [q_0, p_0, qq_0, pp_0, qp_pq_0]
                ]
            )
    q_vec_0 = q_vec_0.T
    q_diff = q_vec_0 - q_inhom
    c_eqs = sp.Matrix( 5, 5, lambda i,j: eigen_data[j][0] * eigen_data[j][2][0][i])
    q_diff = q_diff.subs(eq_subs).subs(params_subs)
    c_eqs = c_eqs.subs(eq_subs).subs(params_subs)
    (c_eqs, "is c_eqs")
    (q_diff, "is q_diff")
    c_eq_small = c_eqs[:2,3:5]
    c_eq_second = c_eqs[2:,:3]
    #(c_eq_small)
    #(type(c_eq_small))
    #(type(q_diff))
    #(type(q_diff[:2]))
    ("waiting for solution")
    c_sol_small = c_eq_small.solve(sp.Matrix([q_diff[:2]]).T)
    ("solution small done, next started")
    (c_eq_second)
    (q_diff)
    c_sol_second = c_eq_second.solve(sp.Matrix([q_diff[2:]]).T)
    ("solution both done")
    #c_sol = c_sol.subs(q_vec_0_exact)
    c_sol_small = c_sol_small.subs(eq_subs).subs(params_subs)
    c_sol_second = c_sol_second.subs(eq_subs).subs(params_subs)
    #c_sol= c_sol.subs(params_I)
    q_inhom = q_inhom.subs(eq_subs).subs(params_subs)
    q_p_sol_small = (c_sol_small[0]*eigen_data[3][0] * sp.exp(eigen_data[3][0]*t) * eigen_data[3][2][0]
            +c_sol_small[1]*eigen_data[4][0] * sp.exp(eigen_data[4][0]*t) * eigen_data[4][2][0]
            + q_inhom
            )
    q_p_sol_second = (c_sol_second[0]*eigen_data[0][0] * sp.exp(eigen_data[0][0] * t) * eigen_data[0][2][0]
            + c_sol_second[1] * eigen_data[1][0] * sp.exp(eigen_data[1][0] * t) * eigen_data[1][2][0]
            + c_sol_second[2] * eigen_data[2][0] * sp.exp(eigen_data[2][0] * t) * eigen_data[2][2][0]
            + q_inhom
            )
    q_p_sol_small = q_p_sol_small.subs(eq_subs).subs(params_subs).subs(q_vec_0_exact).subs(params_subs)
    q_p_sol_second = q_p_sol_second.subs(eq_subs).subs(params_subs).subs(q_vec_0_exact).subs(params_subs) 
    funcs = []
    for i in range(2):
        funcs.append(sp.lambdify(t, q_p_sol_small[i]))
        (q_p_sol_small[i])
    (q_p_sol_second, "is final solution for second moments")
    for i in range(3):
        if q_p_sol_second[i+2] == sp.Integer(0):
            funcs.append(lambda t: t * 0)
        else:
            funcs.append(sp.lambdify(t, q_p_sol_second[i+2]))
        (q_p_sol_second[i+2])
    return funcs

def Ehrenfest_data_TD(times=np.linspace(0,40,801), func=Hams.GQFPE, params_name = "I"):
    dt = times[1] - times[0]
    params_dict = params.params[params_name]
    initial_vec = np.zeros((5,1), dtype=np.float64)
    pp_0 = get_pp_0(**params_dict)
    initial_vec[0,0] = 2
    initial_vec[2,0] = pp_0 + 4
    initial_vec[3,0] = pp_0
    expectations = [np.zeros(len(times)) for _ in range(5)]
    for i in range(len(expectations)):
        expectations[i][0] = initial_vec[i,0]
    A = [Ehrenfest_A_TD(dt*i/4, **params_dict) for i in range(5)]
    inhom_vecs = [Ehrenfest_inhom_TD(dt*i/2, **params_dict) for i in range(3)]
    for i in range(len(times) - 1):
        U_1 = propagate.CFME_matrix(dt/2, A[:3])
        U_2 = propagate.CFME_matrix(dt/2, A[2:])
        U_total = U_2@U_1
        initial_vec = (U_total@initial_vec 
                + 1/6*dt*(U_total@inhom_vecs[0] + 4*U_2@inhom_vecs[1] + inhom_vecs[2])
                )
        A[0] = A[4]
        for j in range(1,5):
            A[j] = Ehrenfest_A_TD(times[i]+dt+dt*j/4, **params_dict)
        inhom_vecs[0] = inhom_vecs[2]
        for j in range(1,3):
            inhom_vecs[j] = Ehrenfest_inhom_TD(times[i]+dt+dt*j/2,**params_dict)
        for j in range(len(expectations)):
            expectations[j][i+1] = initial_vec[j,0]
    return expectations

    return

def Ehrenfest_inhom_TD(t, hbar=1, m=1, w=1 , w_c=1, T=1, gamma =0.1, **params):
    '''
    Returns the numerical time-dependent inhom vector of D_pp, D_xp, D_xx
    defining the inhomogenous part of the Ehrenfest equations
    '''
    gamma_s = gamma/w_c
    R_pq = Hams.R_pq(t, w_c=w_c, T=T, gamma_s=gamma_s)
    R_pp = Hams.R_pp(t, w_c=w_c, T=T, gamma_s=gamma_s)
    R_qq = Hams.R_qq(t, w_c=w_c, T=T, gamma_s=gamma_s)
    inhom_vec = np.zeros((5,1), dtype=np.float64)
    inhom_vec[2,0] = 2*hbar*R_qq/m
    inhom_vec[3,0] = 2*hbar*m*gamma**2*R_pp
    inhom_vec[4,0] = 2*hbar*gamma*R_pq
    return inhom_vec

def Ehrenfest_A_TD(t, w=1, w_c = 1, T=1, m=1, gamma=0.1, hbar=1, **params):
    '''
    Returns the numerical time-dependent matrix A defining the homogenous part of the Ehrenfest equations
    '''
    A = np.zeros((5,5), dtype=np.complex128)
    m_es = Hams.m_e(t, m=m, gamma_s = gamma/w_c, w_c=w_c)
    A[0,1] = 1/m_es
    A[1,0] = -m*w**2
    A[1,1] = -2*gamma*Hams.Gamma(t, w_c=w_c, gamma_s=gamma/w_c, T=T)
    A[2,4] = A[0,1]
    A[3,3] = 2*A[1,1]
    A[3,4] = A[1,0]
    A[4,2] = 2*A[1,0]
    A[4,3] = 2*A[0,1]
    A[4,4] = A[1,1]

    return A
   # A_eqs = sp.Matrix(
   #         [
   #             [0, 1/m_es, 0, 0, 0],
   #             [-m*w**2, -2*sp.I*A*hbar, 0, 0, 0],
   #             [0, 0, 0, 0, 1/m_es],
   #             [0, 0, 0, -4*sp.I*A*hbar, -m*w**2],
   #             [0, 0, -2*m*w**2, 2/m_es, -2*sp.I*A*hbar]
   #             ]
   #         )


def get_pp_0(hbar=1, m=1, w=1, T=1, **params):
    '''
    return the numerical value of initial p^2
    '''
    return hbar/(2*m*w) * (1 - np.exp(-2*hbar*w/T))/(1-np.exp(-hbar*w/T))**2
    
    
