'''
Defines the Hamiltonian and superoperators corresponding to each QFPE
'''
import numba
import numpy as np
import scipy
import sympy as sp


#import series
import SHO
# def CL_Liouvillian(t,T=1,m=1,hbar=1,gamma =.2,omega=1,n=30,time_dependent=True):
#     if(time_dependent==True):
#         return 0
#     else:
#         H = SHO.H(m=m,hbar=hbar,omega=omega,n=n)
#         x = SHO.x(m=m,hbar=hbar,omega=omega,n=n)
#         p = SHO.p(m=m,hbar=hbar,omega=omega,n=n)
#         I = np.eye(n)
#         return (-1j/hbar*(np.kron(H,I)-np.kron(I,H.T)) 
#          -1j/hbar*gamma/2*(np.kron(x@p,I)+np.kron(x,p.T)-np.kron(p,x.T) - np.kron(I,(p@x).T))
#                            -m*gamma/(hbar**2*(1/T))*(np.kron(x@x,I) - 2*np.kron(x,x.T) + np.kron(I,(x@x).T)
#                                                 )
#     )

def A_term(n=100, w=1, hbar=1, m=1):
    '''
    Returns the matrix superoperator corresponding to the [x,{p,\rho}] term.
    '''
    _, xp, px, _ = SHO.get_second_moments(n, w, hbar, m)
    x = SHO.x(n, w, hbar, m)
    p = SHO.p(n, w, hbar, m)
    I = np.eye(n, dtype=np.complex128)
    return np.kron(xp, I) + np.kron(x, p.T) - np.kron(p, x.T) - np.kron(I, px.T)  

def D_pq_term(n=100, w=1, hbar=1, m=1):
    '''
    Returns the matrix superoperator corresponding to the [x,[p,\rho]] term.
    '''
    _, xp, px, _ = SHO.get_second_moments(n, w, hbar, m)
    x = SHO.x(n, w, hbar, m)
    p = SHO.p(n, w, hbar, m)
    I = np.eye(n, dtype=np.complex128)
    return np.kron(xp, I) + np.kron(I, px.T) - np.kron(x, p.T) - np.kron(p, x.T)

def D_pp_term(n=100, w=1, hbar=1, m=1):
    '''
    Returns the matrix superoperator corresponding to the [x,[x,\rho]] term.
    '''

    xx, _, _, _ = SHO.get_second_moments(n, w, hbar, m)
    x = SHO.x(n, w, hbar, m)
    I = np.eye(n, dtype=np.complex128)
    return np.kron(xx, I) + np.kron(I, xx.T) - 2*np.kron(x, x.T)

def D_qq_term(n=100, w=1, hbar=1, m=1):
    '''
    Returns the matrix superoperator corresponding to the [p,[p,\rho]] term.
    '''
    _, _, _, pp = SHO.get_second_moments(n, w, hbar, m)
    p = SHO.p(n, w, hbar, m)
    I = np.eye(n, dtype=np.complex128)
    #print(f"p is {p}")
    #print(np.kron(pp, I), np.kron(I, pp.T), -2*np.kron(p, p.T))
    
    return np.kron(pp, I) + np.kron(I, pp.T) - 2*np.kron(p, p.T)

def H_term(n=100, w=1, hbar=1, m=1):
    H = SHO.H(n, w, hbar, m)
    I = np.eye(n, dtype=np.complex128)
    return -1j/hbar * (np.kron(H, I) - np.kron(I,H.T))

def H_eff_term(n=100, w=1, hbar=1, m=1, m_es=.8):
    H = SHO.H_eff(n, w, hbar, m, m_es)
    I = np.eye(n, dtype=np.complex128)
    return -1j/hbar * (np.kron(H, I) - np.kron(I,H.T))

def CL(n=100, w=1, hbar=1, m=1, gamma = .1, T= 1, **kwargs):
    '''
    Returns the superoperator corresponding to the Caldeira-Leggett equation
    '''
    #k_b = 1
    return (H_term(n, w, hbar, m) -1j * gamma / hbar * A_term(n, w, hbar, m) 
            -2 * gamma * T * m/ hbar**2 * D_pp_term(n, w, hbar, m)
            )

def GQFPE_SH(n=100, w=1, hbar=1, m=1, gamma=0.1, T=1, w_c=1, **kwargs):
    gamma_s = gamma/w_c
    '''
    Returns the superoperator corresponding to the GQFPE in the high temperature, steady state limit
    '''
    coeff_A = -1j * gamma /(hbar * (1 - 2*gamma_s))
    coeff_D_pq = 2 * T * gamma_s/hbar**2 * (1 - 4*gamma_s)/(1-2*gamma_s)**2
    coeff_D_pp = -2*m*gamma*T/hbar**2 * (1 - 6 * gamma_s + 10 * gamma_s**2)/(1-2*gamma_s)**2
    coeff_D_qq = - gamma_s * T/(m * hbar **2 * w_c * (1 - 2*gamma_s)**2)
    m_s = m * (1-2*gamma_s)
    #print(f"coeff_A, {coeff_A}")
    #print(f"coeff_D_pq {coeff_D_pq}")
    #print(f"coeff_D_pp {coeff_D_pp}")
    #print(f"coeff_D_qq {coeff_D_qq}")
    return (H_eff_term(n, w, hbar, m, m_s) + coeff_A * A_term(n, w, hbar, m) 
            + coeff_D_pp * D_pp_term(n, w, hbar, m) 
            + coeff_D_pq * D_pq_term(n, w, hbar, m) + coeff_D_qq * D_qq_term(n, w, hbar, m)
            )


def GQFPE_S(n=100, w=1, hbar=1, m=1, gamma=0.1, T=1, w_c=1, **kwargs):
    gamma_s = gamma/w_c
    coeff_A = -1j * gamma/(hbar * (1 - 2*gamma_s))
    coeff_D_pq = gamma/hbar * R_pq(1000, w_c = w_c, T = T, gamma_s = gamma_s)
    coeff_D_pp = -m * gamma**2/hbar * R_pp(1000, w_c = w_c, T = T, gamma_s = gamma_s)
    coeff_D_qq = -1/(m*hbar) * R_qq(1000, w_c = w_c, T = T, gamma_s = gamma_s)
    m_s = m * (1 - 2*gamma_s)
    
    #print(f"coeff_A, {coeff_A}")
    #print(f"coeff_D_pq {coeff_D_pq}")
    #print(f"coeff_D_pp {coeff_D_pp}")
    #print(f"coeff_D_qq {coeff_D_qq}")
    
    return (H_eff_term(n, w, hbar, m, m_s) + coeff_A * A_term(n, w, hbar, m) 
            + coeff_D_pp * D_pp_term(n, w, hbar, m) 
            + coeff_D_pq * D_pq_term(n, w, hbar, m) + coeff_D_qq * D_qq_term(n, w, hbar, m)
            )

def GQFPE(t, n=100, w=1, hbar=1, m=1, gamma=0.1, T=1, w_c=1, **kwargs):
    gamma_s = gamma/w_c
    coeff_A = -1j * gamma/hbar * Gamma(t, w_c = w_c, T = T, gamma_s = gamma_s)
    coeff_D_pq = gamma/hbar * R_pq(t, w_c = w_c, T = T, gamma_s = gamma_s)
    coeff_D_pp = -m * gamma**2/hbar * R_pp(t, w_c = w_c, T = T, gamma_s = gamma_s)
    coeff_D_qq = -1/(m*hbar) * R_qq(t, w_c = w_c, T = T, gamma_s = gamma_s)
    m_s = m * (1 - 2*gamma_s*F_2(t*w_c))
   
  #  print(f"H_eff_term, {H_eff_term(n, w, hbar, m, m_s)}")
  #  print(f"gamma, {gamma}")
  #  print(f"coeff_A, {coeff_A}")
  #  print(f"coeff_D_pq {coeff_D_pq}")
  #  print(f"coeff_D_pp {coeff_D_pp}")
  #  print(f"coeff_D_qq {coeff_D_qq}")
    
    return (H_eff_term(n, w, hbar, m, m_s) + coeff_A * A_term(n, w, hbar, m) 
            + coeff_D_pp * D_pp_term(n, w, hbar, m) 
            + coeff_D_pq * D_pq_term(n, w, hbar, m) + coeff_D_qq * D_qq_term(n, w, hbar, m)
            )
#----------------------------------------------- Begin Jang Terms calculation------------------
NUM_TERMS = 100000
@numba.jit
def F_0(t_s):
    return 1 - np.exp(-t_s)

@numba.jit
def F_1(t_s):
    return 1 - (1+t_s)*np.exp(-t_s)

@numba.jit
def F_2(t_s):
    return 1 - (1 + t_s + t_s**2/2)*np.exp(-t_s)

@numba.jit
def F_1t(t_s):
    return 1 - (1 + t_s - t_s**2/2) * np.exp(-t_s)

@numba.jit
def m_e(t, m=1, gamma_s=0.1, w_c=1):
    return m - 2*m*gamma_s*F_2(t*w_c)

@numba.jit
def r_m(t, gamma_s=0.1, w_c=1):
    return 1 - 2*gamma_s*F_2(t*w_c)

@numba.jit
def G_0(t_s, T_s=1):
    #res = 1/np.tan(1/(2*T_s))*F_0(t_s)
    res = 0
    for i in range(NUM_TERMS, 0, -1):
        res += 4 * T_s * F_0(2*np.pi*i*t_s*T_s)/((2*np.pi*i*T_s)**2  - 1)
    #print("summation final is ", res)
    res += 1/np.tan(1/(2*T_s))*F_0(t_s)
    #print("cotan term is " , 1/np.tan(1/(2*T_s))*F_0(t_s))
    #print("type of res is :", type(res))
    return res

@numba.jit
def G_1(t_s, T_s=1):
    res = 1/np.tan(1/(2*T_s))*F_1(t_s)
    for i in range(NUM_TERMS, 0, -1):
        res += 4 * T_s * F_1(2*np.pi*i*t_s*T_s)/((2*np.pi*i*T_s)*((2*np.pi*i*T_s)**2 - 1))
    return res

@numba.jit
def G_2(t_s, T_s=1):
    res = 1/np.tan(1/(T_s*2)) * F_2(t_s)
    for i in range(NUM_TERMS, 0, -1):
        res += 4 * T_s * F_2(2*np.pi*i*t_s*T_s)/((2*np.pi*i*T_s)**2*((2*np.pi*i*T_s)**2 - 1))
    return res

@numba.jit
def G_1t(t_s, T_s=1):
    res = 1/np.tan(1/(T_s*2)) * F_1t(t_s)
    for i in range(NUM_TERMS, 0, -1):
        res += 4 * T_s * F_1t(2*np.pi*i*t_s*T_s)/((2*np.pi*i*T_s)*((2*np.pi*i*T_s)**2 - 1))
    return res

@numba.jit
def Gamma(t, w_c = 1, T = 1, gamma_s = 0.1):
    return F_1t(t*w_c) / r_m(t, gamma_s = gamma_s, w_c = w_c) 

@numba.jit
def R_pq(t, w_c = 1, T = 1, gamma_s = 0.1):
    r_m_val = r_m(t, gamma_s = gamma_s, w_c = w_c) 
    return G_1t(t*w_c, T_s = T/w_c)/r_m_val - 2 * gamma_s/r_m_val**2*G_2(t*w_c, T_s = T/w_c) * F_1t(t*w_c)

@numba.jit
def R_qq(t, w_c = 1, T = 1, gamma_s = 0.1):
    return gamma_s/r_m(t, gamma_s = gamma_s, w_c = w_c)**2*G_2(t*w_c, T_s = T/w_c)

@numba.jit
def R_pp(t, w_c = 1, T = 1, gamma_s = 0.1):
    #print(G_0(t*w_c, T_s = T/w_c)/gamma_s, - 2 * Gamma(t, w_c=w_c, T=T, gamma_s=gamma_s)*G_1t(t*w_c, T/w_c)
    #        , 2*gamma_s*Gamma(t, w_c=w_c, T=T, gamma_s = gamma_s)**2*G_2(t*w_c, T/w_c)
    #    )
    return (G_0(t*w_c, T_s = T/w_c)/gamma_s - 2 * Gamma(t, w_c=w_c, T=T, gamma_s=gamma_s)*G_1t(t*w_c, T/w_c)
            + 2*gamma_s*Gamma(t, w_c=w_c, T=T, gamma_s=gamma_s)**2*G_2(t*w_c, T/w_c)
            )


#SUM_TERMS = 6400
# can consider making beta_s a global variable
#def R_pq(time, beta_s, gamma_s):
#    return 1/r_m(time, gamma_s) * tilde_G_1(time, beta_s) -2 * gamma_s/((r_m(time, gamma_s))**2) * G_2(time, beta_s) * tilde_F_1(time)
#
#def R_qq(time, beta_s, gamma_s):
#    return .5*gamma_s/(r_m(time,gamma_s)**2) * G_2(time, beta_s)
#def R_pp(time, beta_s, gamma_s):
#    return 1/gamma_s * G_0(time, beta_s) - 2 * gamma_t(time, gamma_s) * tilde_G_1(time, beta_s) +2 * gamma_s * gamma_t(time, gamma_s)**2 * G_2(time, beta_s)
#def gamma_t(time, gamma_s):
#    return tilde_F_1(time)/r_m(time, gamma_s)
#def r_m(time, gamma_s):
#    return (1 - 2*gamma_s*F_2(time))
#
#def F_2(time):
#    return (1 - (1 + time + (time**2)/2)* np.exp(-time))
#
## may improve these calculations by saving some temporary results
#def tilde_G_1 (time, beta_s):
#    sum = 1/np.tan(beta_s/2) *tilde_F_1(time)
#    for n in range (1, SUM_TERMS):
#        sum += (4/beta_s)*tilde_F_1(2 * np.pi * n * time/beta_s)/((2*np.pi*n/beta_s)*((2*np.pi*n/beta_s)**2-1))
#    return sum
#def tilde_F_1(time):
#     return (1 - (1 + time - (time**2)/2)* np.exp(-time))
#
#def G_2(time, beta_s):
#    sum = 1/np.tan(beta_s/2) *F_2(time)
#    for n in range (1, SUM_TERMS):
#        sum += ((4/beta_s)*F_2(2 * np.pi * n * time/beta_s))/(((2*np.pi*n/beta_s)**2)*((2*np.pi*n/beta_s)**2-1))
#    return sum
#
#def G_0(time, beta_s):
#    sum = 1/np.tan(beta_s/2)* F_0(time)
#    for n in range (1, SUM_TERMS):
#        sum += (4/beta_s)*F_0(2 * np.pi * n * time/beta_s)/(((2*np.pi*n/beta_s)**2-1))
#    return sum
#def F_0 (time):
#    return 1 - np.exp(-time)


#-----------------------------------------------End Jang Terms calculation------------------
def friction_Liouvillian(t,gamma_e=1, w_c=10, T = 1, m=1,hbar=1,omega=1,n=30,time_dependent=False):
    if(time_dependent==True):
        return 0
    else:
        x = SHO.x(m=m,hbar=hbar,omega=omega,n=n)
        p = SHO.p(m=m,hbar=hbar,omega=omega,n=n)
        I = np.eye(n)
        gamma_s,m_es,A,B,D_p,D_q = get_constants_CL_Wigner(gamma_e,w_c,T,m)
        B = 0
        D_q= 0
        D_p = 0
        # return (-1j/hbar*(np.kron(p@p/(2*m_es)+1/2*m*omega**2*x@x,I) - np.kron(I,(p@p/(2*m_es)+1/2*m*omega**2*x@x).T))
        #         -1j/(2*hbar)*A*(np.kron(x@p,I)+np.kron(x,p.T)-np.kron(p,x.T)-np.kron(I,(p@x).T))
        #         +B/hbar**2*(np.kron(x@p,I)-np.kron(x,p.T)-np.kron(p,x.T)+np.kron(I,(p@x).T))
        #         -D_p/hbar**2*(np.kron(p@p,I)-2*np.kron(p,p.T)+np.kron(I,(p@p).T))
        #        )
        return (-1j/hbar*(np.kron(p@p/(2*m_es)+1/2*m*omega**2*x@x,I) - np.kron(I,(p@p/(2*m_es)+1/2*m*omega**2*x@x).T))
                -1j/(2*hbar)*A*(np.kron(x@p,I)+np.kron(x,p.T)-np.kron(p,x.T)-np.kron(I,(p@x).T))
                +B/hbar**2*(np.kron(x@p,I)-np.kron(x,p.T)-np.kron(p,x.T)+np.kron(I,(p@x).T))
                -D_q/hbar**2*(np.kron(p@p,I)-2*np.kron(p,p.T)+np.kron(I,(p@p).T))
                -D_p/hbar**2*(np.kron(x@x,I)+np.kron(I,(x@x).T)-2*np.kron(x,x.T))
               )
def CL_Liouvillian(t,gamma_e=1, w_c=10, T = 1, m=1,hbar=1,omega=1,n=30,time_dependent=False):
    if(time_dependent==True):
        return 0
    else:
        x = SHO.x(m=m,hbar=hbar,omega=omega,n=n)
        p = SHO.p(m=m,hbar=hbar,omega=omega,n=n)
        I = np.eye(n)
        xx,xp,px,pp = SHO.get_second_moments(n=n,omega=omega,hbar=hbar,m=m,lib="np")
        gamma_s,m_es,A,B,D_p,D_q = get_constants_CL_Wigner(gamma_e,w_c,T,m)
        # return (-1j/hbar*(np.kron(p@p/(2*m_es)+1/2*m*omega**2*x@x,I) - np.kron(I,(p@p/(2*m_es)+1/2*m*omega**2*x@x).T))
        #         -1j/(2*hbar)*A*(np.kron(x@p,I)+np.kron(x,p.T)-np.kron(p,x.T)-np.kron(I,(p@x).T))
        #         +B/hbar**2*(np.kron(x@p,I)-np.kron(x,p.T)-np.kron(p,x.T)+np.kron(I,(p@x).T))
        #         -D_p/hbar**2*(np.kron(p@p,I)-2*np.kron(p,p.T)+np.kron(I,(p@p).T))
        #        )
        return (-1j/hbar*(np.kron(pp/(2*m_es)+1/2*m*omega**2*xx,I) - np.kron(I,(pp/(2*m_es)+1/2*m*omega**2*xx).T))
                -1j/(2*hbar)*A*(np.kron(xp,I)+np.kron(x,p.T)-np.kron(p,x.T)-np.kron(I,(px).T))
                +B/hbar**2*(np.kron(xp,I)-np.kron(x,p.T)-np.kron(p,x.T)+np.kron(I,(px).T))
                -D_q/hbar**2*(np.kron(pp,I)-2*np.kron(p,p.T)+np.kron(I,(pp).T))
                -D_p/hbar**2*(np.kron(xx,I)+np.kron(I,(xx).T)-2*np.kron(x,x.T))
               )
def get_constants_Jang_Wigner(gamma_e=1,w_c=10,T=1,m=1,time_independent=True):
    if(time_independent==True):
        gamma_s = gamma_e/w_c
        m_es = 1-2*gamma_s
        A = 2 * gamma_e/(1-2*gamma_s)
        B = 2*T*gamma_s*(1-4*gamma_s)/(1-2*gamma_s)**2
        D_p = gamma_e*2 * m *T *(1-6*gamma_s+10*gamma_s**2)/(1-2*gamma_s)**2
        D_q = gamma_s*T/(m*w_c*(1-2*gamma_s)**2)
        return gamma_s, m_es, A, B, D_p, D_q
    
def Jang_Wigner(t,gamma_e=1, w_c=10, T = 1, m=1,hbar=1,omega=1,n=30,time_dependent=False):
    #k_b=1, or T = k_b*T
    if(time_dependent==True):
        return 0
    else:
        #H = SHO.H(m=m,hbar=hbar,omega=omega,n=n)
        x = SHO.x(m=m,hbar=hbar,omega=omega,n=n)
        p = SHO.p(m=m,hbar=hbar,omega=omega,n=n)
        I = np.eye(n)
        gamma_s,m_es,A,B,D_p,D_q = get_constants_Jang_Wigner(gamma_e,w_c,T,m)
        return (-1*np.kron(p,x.T)/m_es/(-1j*hbar) + np.kron(x,p.T)*m*omega**2/(1j*hbar) 
            + A * np.kron(I,(x@p).T)/(1j*hbar) + B*np.kron(p,p.T)/(1j*hbar)/(-1j*hbar)
                +D_p*np.kron(I,p.T@p.T)/(1j*hbar)**2+D_q*np.kron(p@p,I)/(-1j*hbar)**2
               )
        #A * np.kron(I,p.T@x.T)
        #A * np.kron(I,(x@p).T) looks correct;trying new one because of wrong analytical results
        
def Jang_Liouvillian(t,gamma_e=1, w_c=10, T = 1, m=1,hbar=1,omega=1,n=30,time_dependent=False):
    if(time_dependent==True):
        return 0
    else:
        x = SHO.x(m=m,hbar=hbar,omega=omega,n=n)
        p = SHO.p(m=m,hbar=hbar,omega=omega,n=n)
        I = np.eye(n)
        xx,xp,px,pp = SHO.get_second_moments(n=n,omega=omega,hbar=hbar,m=m,lib="np")
        gamma_s,m_es,A,B,D_p,D_q = get_constants_Jang_Wigner(gamma_e,w_c,T,m)
     
    return (-1j/hbar*(np.kron(pp/(2*m_es)+1/2*m*omega**2*xx,I) - np.kron(I,(pp/(2*m_es)+1/2*m*omega**2*xx).T))
                -1j/(2*hbar)*A*(np.kron(xp,I)+np.kron(x,p.T)-np.kron(p,x.T)-np.kron(I,(px).T))
                +B/hbar**2*(np.kron(xp,I)-np.kron(x,p.T)-np.kron(p,x.T)+np.kron(I,(px).T))
                -D_q/hbar**2*(np.kron(pp,I)-2*np.kron(p,p.T)+np.kron(I,(pp).T))
                -D_p/hbar**2*(np.kron(xx,I)+np.kron(I,(xx).T)-2*np.kron(x,x.T))
               )


def get_constants_CL_Wigner(gamma_e=1,w_c=10,T=1,m=1,time_independent=True):
     if(time_independent==True):
        gamma_s = 0
        m_es = 1-2*gamma_s
        A = 2 * gamma_e/(1-2*gamma_s)
        B = 2*T*gamma_s*(1-4*gamma_s)/(1-2*gamma_s)**2
        D_p = gamma_e*2 * m *T *(1-6*gamma_s+10*gamma_s**2)/(1-2*gamma_s)**2
        D_q = gamma_s*T/(m*w_c*(1-2*gamma_s)**2)
        return gamma_s, m_es, A, B, D_p, D_q

def CL_Wigner(t,gamma_e=1, w_c=10, T = 1, m=1,hbar=1,omega=1,n=30,time_dependent=False):
    if(time_dependent==True):
        return 0
    else:
        x = SHO.x(m=m,hbar=hbar,omega=omega,n=n)
        p = SHO.p(m=m,hbar=hbar,omega=omega,n=n)
        I = np.eye(n)
        gamma_s,m_es,A,B,D_p,D_q = get_constants_CL_Wigner(gamma_e,w_c,T,m)
        return (-1*np.kron(p,x.T)/m_es/(-1j*hbar) + np.kron(x,p.T)*m*omega**2/(1j*hbar)
                +A * np.kron(I,(x@p).T)/(1j*hbar) + B*np.kron(p,p.T)/(1j*hbar)/(-1j*hbar)
                + D_p*np.kron(I,p.T@p.T)/(1j*hbar)**2+D_q*np.kron(p@p,I)/(-1j*hbar)**2
               )

def CHO(t, w = 1, m=1, x_C = 0,time_dependent=False):
    if(time_dependent==True):
        return 0
    else:
        result = np.zeros((2,2),dtype=np.complex128)
        result[0,1] = 1
        result[1,0] = -m*w**2
        return result

def calc_jang_params (m=1,w_L=1,w_C=1,w_R=1,x_L=-1,x_R=1):
    x_LC =w_L**2 * x_L/(w_L **2 + w_C **2)
    x_RC = w_R **2 * x_R/(w_R **2 + w_C **2)
    E_L = (.5 * (m * (-w_L**2 * w_C ** 4 - w_C **2 * w_L **4 )* x_L**2)
            /((w_L ** 2 + w_C ** 2) ** 2))
    E_R = (.5 * (m * (-w_R**2 * w_C ** 4 - w_C **2 * w_R **4 )*x_R**2)
            /((w_R ** 2 + w_C ** 2) ** 2))
    return x_LC,x_RC,E_L,E_R

e_1 = np.arange(1,-1,-1,dtype=np.complex128).reshape(2,1)
def quadratic_piecewise(t,m=1,initial_list=["C"],vec=e_1,w_L=1,w_C=1,w_R=1,x_L=-1,x_R=1,time_dependent=True):
    #not explicit time-dependence; time dependence due to x_LC, x_RC, E_L, E_R
    initial = initial_list[0] #list is so that I can make changes to the original object
    if(time_dependent==False):
        return 0
    elif(time_dependent==True):
        x = vec[0]
        H = np.zeros((2,2),dtype=np.complex128)
        H[0,1] = 1
        x_LC,x_RC,E_L,E_R = calc_jang_params(m=m,w_L=w_L,w_C=w_C,w_R=w_R,x_L=x_L,x_R=x_R)
        bounds = cross_overpoints(initial=initial,x_LC=x_LC,x_RC=x_RC,x_L=x_L,x_R=x_R)
        if((x < bounds[0]) or (x > bounds[1])):
            if(x < bounds[0]):
                new_initial, new_center,dp_dt = shift_term(initial=initial,bound="left",
                                                           x_L =x_L,x_R=x_R, m=m,
                                                           w_L = w_L, w_C = w_C,
                                                           w_R = w_R)
            if(x > bounds[1]):
                #print("here,success ",x)
                new_initial, new_center,dp_dt = shift_term(initial=initial,bound="right",
                                                           x_L =x_L,x_R=x_R, m=m,
                                                           w_L = w_L, w_C = w_C,
                                                           w_R = w_R)
            vec[0]-= new_center
            initial_list[0] = new_initial
            H[1,0] = dp_dt
            return H
        else:
            H[1,0] = get_dp_dt(initial=initial,m=m,w_C=w_C,w_L=w_L,w_R=w_R)
            return H
def get_dp_dt(initial="C",m=1,w_L=1,w_C=1,w_R=1):
    if(initial=="L"):
        return -m*w_L**2
    if(initial=="C"):
        return m*w_C**2
    if(initial=="R"):
        return -m*w_R**2

def cross_overpoints(initial="C",x_LC=-.5,x_RC=.5,x_L=-1,x_R=1):
    #is this even a viable choice for these 5 parameters? Likely is, with strange frequencies. Should have
    #better initial values, later
    if(initial=="C"):
        #print([x_LC, x_RC])
        return [x_LC, x_RC]
    if(initial == "L"):
        #print( [-10000,x_LC-x_L])
        return [-10000,x_LC-x_L] 
    if(initial == "R"):
        #print([x_RC -x_R,10000])
        return [x_RC -x_R,10000] 
    #choosing 10,000 as a ridiculous large number which won't be crossed

def shift_term(initial="C",bound="left",x_L=-1,x_R=1,m=1,w_L=1,w_C=1,w_R=1):
    #only describes the shifts (e.g. from C to L); 
    #otherwise force described in the main function
    if(initial=="C" and bound=="left"):
        return "L", x_L, get_dp_dt(initial="L",m=m,w_L=w_L)
    if(initial=="C" and bound == "right"):
        return "R",x_R, get_dp_dt(initial="R",m=m,w_R=w_R)
    if(initial=="L" and bound == "right"):
        return "C", -x_L, get_dp_dt(initial="C",m=m, w_C=w_C)
    if(initial=="R" and bound == "left"):
        return "C",-x_R, get_dp_dt(initial="C",m=m, w_C=w_C)
    return 

# def quadratic_piecewise(t,m=1,vec=e_1,w_L=1,w_C=1,w_R=1,x_L=-1,x_R=1,time_dependent=True):
#     #not explicit time-dependence; time dependence due to x_LC, x_RC, E_L, E_R
#     if(time_dependent==False):
#         return 0
#     elif(time_dependent==True):
#         x = vec[0]
#         x_LC,x_RC,E_L,E_R = calc_jang_params(m=m,w_L=w_L,w_C=w_C,w_R=w_R,x_L=x_L,x_R=x_R)
#         if(x <= x_LC):
#             return .5 * m * w_L ** 2 * (x - x_L)**2 + E_L
#         if( x_LC < x and x <= x_RC):
#             return -.5*m*w_C**2*x**2
#         if(self.x_RC < x_val):
#             return .5*m * w_R**2 *(x-x_R)**2+E_R
    
    
    
    
def jang_potential (self, x_val):
        if(x_val <= self.x_LC):
            return .5 * self.mass * self.omega_L ** 2 * (x_val - self.x_L)**2 + self.H_L
        if( self.x_LC < x_val and x_val <= self.x_RC):
            return -.5 * self.mass * self.omega_C ** 2 * (x_val)**2
        
        if(self.x_RC < x_val):
            return .5 * self.mass * self.omega_R ** 2 * (x_val - self.x_R)**2 + self.H_R
        
def Jang_Liouvillian_CL_like(t,gamma_e=1, w_c=10, T = 1, m=1,hbar=1,omega=1,n=30,time_dependent=False):
    if(time_dependent==True):
        return 0
    else:
        x = SHO.x(m=m,hbar=hbar,omega=omega,n=n)
        p = SHO.p(m=m,hbar=hbar,omega=omega,n=n)
        xx,xp,px,pp = SHO.get_second_moments(n=n,omega=omega,hbar=hbar,m=m,lib="np")
        I = np.eye(n)
        gamma_s,m_es,A,B,D_p,D_q = get_constants_Jang_Wigner(gamma_e,w_c,T,m)
        B = 0
        D_q = 0
    return (-1j/hbar*(np.kron(pp/(2*m_es)+1/2*m*omega**2*xx,I) - np.kron(I,(pp/(2*m_es)+1/2*m*omega**2*xx).T))
                -1j/(2*hbar)*A*(np.kron(xp,I)+np.kron(x,p.T)-np.kron(p,x.T)-np.kron(I,(px).T))
                +B/hbar**2*(np.kron(xp,I)-np.kron(x,p.T)-np.kron(p,x.T)+np.kron(I,(px).T))
                -D_q/hbar**2*(np.kron(pp,I)-2*np.kron(p,p.T)+np.kron(I,(pp).T))
                -D_p/hbar**2*(np.kron(xx,I)+np.kron(I,(xx).T)-2*np.kron(x,x.T))
               )

def Jang_Liouvillian_steady_state(t,gamma_e=1, w_c=10, T = 1, m=1,hbar=1,omega=1,n=30,time_dependent=False):
    if(time_dependent==True):
        return 0
    else:
        time = 1000
        x = SHO.x(m=m,hbar=hbar,omega=omega,n=n)
        p = SHO.p(m=m,hbar=hbar,omega=omega,n=n)
        I = np.eye(n)
        xx,xp,px,pp = SHO.get_second_moments(n=n,omega=omega,hbar=hbar,m=m,lib="np")

        gamma_s,m_es,A,B,D_p,D_q = get_constants_Jang_Wigner(gamma_e,w_c,T,m)
        B_s = w_c/T
        
        R_pq_s = R_pq(1000,B_s,gamma_s)
        R_qq_s = R_qq(1000,B_s,gamma_s)
        R_pp_s = R_pp(1000,B_s,gamma_s)
    return (-1j/hbar*(np.kron(pp/(2*m_es)+1/2*m*omega**2*xx,I) - np.kron(I,(pp/(2*m_es)+1/2*m*omega**2*xx).T))
                -1j/(2*hbar)*A*(np.kron(xp,I)+np.kron(x,p.T)-np.kron(p,x.T)-np.kron(I,(px).T))
                +gamma_e/hbar*R_pq_s*(np.kron(xp,I)-np.kron(x,p.T)-np.kron(p,x.T)+np.kron(I,(px).T))
                -R_qq_s/(m*hbar)*(np.kron(pp,I)-2*np.kron(p,p.T)+np.kron(I,(pp).T))
                -m*gamma_e**2/hbar*R_pp_s*(np.kron(xx,I)+np.kron(I,(xx).T)-2*np.kron(x,x.T))
               )

def Jang_Liouvillian_time_dependent(t,gamma_e=1, w_c=10, T = 1, m=1,hbar=1,omega=1,n=30,time_dependent=False):
    if(time_dependent==True):
        x = SHO.x(m=m,hbar=hbar,omega=omega,n=n)
        p = SHO.p(m=m,hbar=hbar,omega=omega,n=n)
        I = np.eye(n)
        xx,xp,px,pp = SHO.get_second_moments(n=n,omega=omega,hbar=hbar,m=m,lib="np")

        B_s = w_c/T
        gamma_s,m_es,A,B,D_p,D_q = get_constants_Jang_Wigner(gamma_e,w_c,T,m)
        #t_s = w_c*t
        R_pq_t = R_pq(t,B_s,gamma_s) #only works with w_c=1;t_s=t
        R_qq_t = R_qq(t,B_s,gamma_s)
        R_pp_t = R_pp(t,B_s,gamma_s)
        Gamma_t = gamma_t(t,gamma_s)
        k_e =2*m*gamma_e*w_c
        K_I_0 = m*gamma_e*w_c*F_0(t) 
        m_e_t = m*(1-2*gamma_s*F_2(t)) #only works with w_c=1;t_s=t
        return (-1j/hbar*(np.kron(pp/(2*m_e_t)+(1/2*m*omega**2+k_e/2-K_I_0)*xx,I) - np.kron(I,(pp/(2*m_e_t)+(1/2*m*omega**2+k_e/2-K_I_0)*xx).T))
                -1j*Gamma_t*gamma_e/(hbar)*(np.kron(xp,I)+np.kron(x,p.T)-np.kron(p,x.T)-np.kron(I,(px).T))
                +gamma_e/hbar*R_pq_t*(np.kron(xp,I)-np.kron(x,p.T)-np.kron(p,x.T)+np.kron(I,(px).T))
                -R_qq_t/(m*hbar)*(np.kron(pp,I)-2*np.kron(p,p.T)+np.kron(I,(pp).T))
                -m*gamma_e**2/hbar*R_pp_t*(np.kron(xx,I)+np.kron(I,(xx).T)-2*np.kron(x,x.T))
               )

    else:
        return 0


def Jang_Liouvillian_no_D_qq(t,gamma_e=1, w_c=10, T = 1, m=1,hbar=1,omega=1,n=30,time_dependent=False):
    if(time_dependent==True):
        return 0
    else:
        x = SHO.x(m=m,hbar=hbar,omega=omega,n=n)
        p = SHO.p(m=m,hbar=hbar,omega=omega,n=n)
        I = np.eye(n)
        gamma_s,m_es,A,B,D_p,D_q = get_constants_Jang_Wigner(gamma_e,w_c,T,m)
        #B = 0
        D_q = 0
    return (-1j/hbar*(np.kron(p@p/(2*m_es)+1/2*m*omega**2*x@x,I) - np.kron(I,(p@p/(2*m_es)+1/2*m*omega**2*x@x).T))
                -1j/(2*hbar)*A*(np.kron(x@p,I)+np.kron(x,p.T)-np.kron(p,x.T)-np.kron(I,(p@x).T))
                +B/hbar**2*(np.kron(x@p,I)-np.kron(x,p.T)-np.kron(p,x.T)+np.kron(I,(p@x).T))
                -D_q/hbar**2*(np.kron(p@p,I)-2*np.kron(p,p.T)+np.kron(I,(p@p).T))
                -D_p/hbar**2*(np.kron(x@x,I)+np.kron(I,(x@x).T)-2*np.kron(x,x.T))
               )

def Jang_Liouvillian_no_D_pq(t,gamma_e=1, w_c=10, T = 1, m=1,hbar=1,omega=1,n=30,time_dependent=False):
    if(time_dependent==True):
        return 0
    else:
        x = SHO.x(m=m,hbar=hbar,omega=omega,n=n)
        p = SHO.p(m=m,hbar=hbar,omega=omega,n=n)
        I = np.eye(n)
        gamma_s,m_es,A,B,D_p,D_q = get_constants_Jang_Wigner(gamma_e,w_c,T,m)
        B = 0
        #D_q = 0
    return (-1j/hbar*(np.kron(p@p/(2*m_es)+1/2*m*omega**2*x@x,I) - np.kron(I,(p@p/(2*m_es)+1/2*m*omega**2*x@x).T))
                -1j/(2*hbar)*A*(np.kron(x@p,I)+np.kron(x,p.T)-np.kron(p,x.T)-np.kron(I,(p@x).T))
                +B/hbar**2*(np.kron(x@p,I)-np.kron(x,p.T)-np.kron(p,x.T)+np.kron(I,(p@x).T))
                -D_q/hbar**2*(np.kron(p@p,I)-2*np.kron(p,p.T)+np.kron(I,(p@p).T))
                -D_p/hbar**2*(np.kron(x@x,I)+np.kron(I,(x@x).T)-2*np.kron(x,x.T))
               )

#----------------------------------Ehrenfest Begin---------------------#

def Ehrenfest_TD(t,gamma_e=1, w_c=10, T = 1, m=1,hbar=1,omega=1,n=30,time_dependent=False):
    if(time_dependent==True):
        matrix = np.zeros((5,5),dtype=np.complex128)    
        gamma_s,m_es,A,B,D_p,D_q = get_constants_Jang_Wigner(gamma_e,w_c,T,m)
        #t_s = w_c*t
        B_s = w_c/T
        R_pq_t = R_pq(t,B_s,gamma_s) #only works with w_c=1;t_s=t
        R_qq_t = R_qq(t,B_s,gamma_s)
        R_pp_t = R_pp(t,B_s,gamma_s)
        Gamma_t = gamma_t(t,gamma_s)
        k_e =2*m*gamma_e*w_c
        K_I_0 = m*gamma_e*w_c*F_0(t) 
        m_e_t = m*(1-2*gamma_s*F_2(t)) #only works with w_c=1;t_s=t
        M = -m*omega**2-(k_e-2*K_I_0)
        matrix[0,1] = 1/m_e_t
        matrix[1,0] = M
        matrix[1,1] = -2*gamma_e*Gamma_t
        matrix[2][3] = 1/m_e_t
        matrix[3][2] = 2*M
        matrix[3][3] = -2*gamma_e*Gamma_t
        matrix[3][4] = 2/m_e_t
        matrix[4][3] = M
        matrix[4][4] = -4*gamma_e*Gamma_t
        return matrix
    else:
        return 0

def Ehrenfest_TD_inhom(t,gamma_e=1, w_c=10, T = 1, m=1,hbar=1,omega=1,n=30,time_dependent=False):
    inhom_vec = np.zeros((5,1),dtype=np.complex128)    
    #t_s = w_c*t
    B_s = w_c/T #boltzmann=1
    gamma_s = gamma_e/w_c
    k_e =2*m*gamma_e*w_c
    K_I_0 = m*gamma_e*w_c*F_0(t) 
    R_pq_t = R_pq(t,B_s,gamma_s) #only works with w_c=1;t_s=t
    R_qq_t = R_qq(t,B_s,gamma_s)
    R_pp_t = R_pp(t,B_s,gamma_s)
    Gamma_t = gamma_t(t,gamma_s)
    M = -m*omega**2-(k_e-2*K_I_0)
    inhom_vec[2,0] = 2*R_qq_t*hbar/m
    inhom_vec[3,0] = 2*R_pq_t*hbar*gamma_e
    inhom_vec[4,0] = 2*hbar*m*gamma_e**2*R_pp_t
    return inhom_vec
