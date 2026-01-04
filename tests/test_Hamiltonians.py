'''
Testing functions for the Hamiltonians (rather, Liouvillians) defined in Hamiltonians.py
'''
import os
import sys

import numpy as np

#sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "QFPE"))

import SHO
import Hamiltonians as Hams

params = {
        "n":2,
        "w":1,
        "T":1,
        "gamma":.1,
        "hbar":1,
        "gamma_s":.1,
        "m":1,
        "w_c":1,
        }


params_2 = {
        "n":2,
        "w":1,
        "hbar":1,
        "m":1,
        "gamma":.1,
        "T":5,
        "w_c":0.5,
        #"gamma_s":.4,
         }

def test_A():
    A = Hams.A_term(n=2, w=1, hbar=1, m=1)
    A_exact = np.zeros((4,4), dtype=np.complex128)
    A_exact[0][0] = 2
    A_exact[0][3] = 2
    A_exact[3][0] = -2
    A_exact[3][3] = -2
    A_exact *= 1j/2
    print(A_exact)
    print(A)
    assert np.allclose(A_exact, A)

def test_D_pq_term():
    D_pq = Hams.D_pq_term(n=2, w=1, hbar=1, m=1)
    D_pq_exact = np.zeros((4,4), dtype=np.complex128)
    D_pq_exact[1][1] = 2
    D_pq_exact[1][2] = 2
    D_pq_exact[2][1] = -2
    D_pq_exact[2][2] = -2

    D_pq_exact *= 1j/2
    print(D_pq_exact)
    print(D_pq)
    assert np.allclose(D_pq_exact, D_pq)

def test_D_pp_term():
    D_pp = Hams.D_pp_term(n=2, w=2, hbar=1, m=1)
    D_pp_exact = np.zeros((4,4), dtype=np.complex128)
    for i in range(4):
        D_pp_exact[i,i] = 2
        D_pp_exact[i, 3-i] = -2
    D_pp_exact *= 1/2/2 
    print(D_pp_exact)
    print(D_pp)
    assert np.allclose(D_pp_exact, D_pp)

def test_D_qq_term():
    D_qq = Hams.D_qq_term(n=2, w=2, hbar=1, m=1)
    D_qq_exact = np.zeros((4,4), dtype=np.complex128)
    for i in range(4):
        D_qq_exact[i,i] = -1
    D_qq_exact[0][3] = 1
    D_qq_exact[1][2] = -1
    D_qq_exact[2][1] = -1
    D_qq_exact[3][0] = 1
    D_qq_exact *= -2 
    print(D_qq_exact)
    print(D_qq)
    assert np.allclose(D_qq_exact, D_qq)

def test_H_term():
    H = Hams.H_term(n=2, w=1 , hbar=1, m=1)
    H_exact = np.zeros((4,4), dtype=np.complex128)
    H_exact[1][1] = -1
    H_exact[2][2] = +1
    H_exact *= -1j
    print(H)
    print(H_exact)
    assert np.allclose(H_exact, H)

def test_H_eff_term():
    H = Hams.H_eff_term(n=2, w=1, hbar=1, m=1, m_es=.8)
    H_exact = np.zeros((4,4), dtype=np.complex128)
    H_exact[1][1] = -.5 -.5/(.8)
    H_exact[2][2] = .5 + .5/(.8)
    H_exact *= -1j
    print(H)
    print(H_exact)
    assert np.allclose(H_exact, H)

def test_CL_term():
    CL = Hams.CL(**params)
    CL_exact = np.zeros((4,4), dtype=np.complex128)
    A_coeff = -params["gamma"]*params["T"]/(params["hbar"] * params["w"])
    for i in range(4):
        CL_exact[i, i] += 2*A_coeff
        CL_exact[i, 3-i] += -2*A_coeff
    CL_exact[0, 0] += params["gamma"]
    CL_exact[1, 1] += 1j/params["w"]
    CL_exact[2, 2] += -1j/params["w"]
    CL_exact[0, 3] += params["gamma"]
    CL_exact[3, 0] += -params["gamma"]
    CL_exact[3, 3] += -params["gamma"]
    print(CL)
    print(CL_exact)
    assert np.allclose(CL_exact, CL)
    
def test_GQFPE_SH():
#    GQFPE_SH = Hams.GQFPE_SH(**params)
#    GQFPE_exact = np.zeros((4,4), dtype=np.complex128)
#    A_coeff = -1j * params["gamma"]/params["hbar"] * (1 - 2*params["gamma_s"])
#    D_pq = 2 * params["T"]/hbar**2 * (1-4*params["gamma_s"])/(1-2*params["gamma_s"])**2
#    D_pp = -2 * params["m"] * params["gamma"i]
    GQFPE_SH = Hams.GQFPE_SH(**params)
    GQFPE_SH_exact = np.zeros((4,4), dtype=np.complex128)
    A = - 0.125j
    D_pq = 0.1875
    D_pp = -0.15625
    D_qq = -0.15625
    # for i in range(4):
    #    GQFPE_SH_exact[i][i] += D_pp + D_qq

    GQFPE_SH_exact[0][0] = 1j*A + D_pp + D_qq
    GQFPE_SH_exact[0][3] = 1j*A -D_pp - D_qq
    GQFPE_SH_exact[1][1] = 1j*D_pq + D_pp + D_qq +18/16*1j
    GQFPE_SH_exact[1][2] = -D_pp + 1j*D_pq + D_qq
    GQFPE_SH_exact[2][1] = -D_pp -1j*D_pq + D_qq
    GQFPE_SH_exact[2][2] = D_pp -1j*D_pq + D_qq -18/16*1j
    GQFPE_SH_exact[3][0] = -1j*A -D_pp -D_qq
    GQFPE_SH_exact[3][3] = -1j*A + D_pp + D_qq
    print(GQFPE_SH_exact)
    print(GQFPE_SH)
    assert np.allclose(GQFPE_SH_exact, GQFPE_SH)

def test_F_0():
    assert Hams.F_0(0) == 0
    assert np.allclose(Hams.F_0(1000), 1)
    assert np.allclose(Hams.F_0(1), 0.6321205588285577)
    assert np.allclose(Hams.F_0(0.5), 0.39346934028736658)

def test_F_1():
    assert Hams.F_1(0) == 0
    assert np.allclose(Hams.F_1(1000), 1)
    assert np.allclose(Hams.F_1(1), 0.26424111765711533)

def test_F_2():
    assert Hams.F_2(0) == 0
    assert np.allclose(Hams.F_2(1000), 1)
    assert np.allclose(Hams.F_2(1), 0.08030139707139416)
    assert np.allclose(Hams.F_2(0.5), 0.014387677966970687)

def test_m_e():
    assert Hams.m_e(t=0, m=2, gamma_s=.1, w_c=1) == 2
    assert np.allclose(Hams.m_e(t=1000, m=1, gamma_s=.1, w_c=1), .8)

def test_r_m():
    assert Hams.r_m(0, gamma_s=0.1, w_c=1) == 1
    assert np.allclose(Hams.r_m(1000, gamma_s=0.1, w_c=1), 0.8)

def test_F_1t():
    assert Hams.F_1t(0) == 0
    assert np.allclose(Hams.F_1t(1000), 1)
    assert np.allclose(Hams.F_1t(1), 1 - 1.5/np.exp(1))

def test_G_0():
    assert np.allclose(Hams.G_0(10000, T_s = 1), 2)
    assert np.allclose(Hams.G_0(0, T_s = 1), 0)
    assert np.allclose(Hams.G_0(0, T_s = 5), 0)
    assert np.allclose(Hams.G_0(1, T_s = 5), 6.3334764192169191)
    assert np.allclose(Hams.G_0(0.5, T_s = 10), 7.8794973352845082)

def test_G_1():
    assert np.allclose(Hams.G_1(10000, T_s = 1), 1.8503060683563158) 
    assert np.allclose(Hams.G_1(0, T_s = 1), 0)
    assert np.allclose(Hams.G_1(0, T_s = 5), 0)
    assert np.allclose(Hams.G_1(1, T_s = 5), 2.6343733036404876) 
    assert np.allclose(Hams.G_1(0.5, T_s = 10), 1.8027704410846626)

def test_G_2():
    assert np.allclose(Hams.G_2(10000, T_s = 1), 1.8333333333333333) 
    assert np.allclose(Hams.G_2(0, T_s = 1), 0)
    assert np.allclose(Hams.G_2(0, T_s = 5), 0)
    assert np.allclose(Hams.G_2(1, T_s = 5), 0.80035771470896522)
    assert np.allclose(Hams.G_2(0.5, T_s = 10), 0.28751650317065951)

def test_G_1t():
    assert np.allclose(Hams.G_1t(10000, T_s = 1), 1.8503060683563158) 
    assert np.allclose(Hams.G_1t(0, T_s = 1), 0)
    assert np.allclose(Hams.G_1t(0, T_s = 5), 0)
    assert np.allclose(Hams.G_1t(1, T_s = 5), 4.4676350940320351)
    assert np.allclose(Hams.G_1t(0.5, T_s = 10), 3.3178332741741008)

def test_gamma():
    assert np.allclose(Hams.Gamma(10000, T = 1), 1.25) 
    assert np.allclose(Hams.Gamma(0, T = 1), 0)
    assert np.allclose(Hams.Gamma(0, T = 5), 0)
    assert np.allclose(Hams.Gamma(1, T = 5, gamma_s = 0.1), 0.45549623504988978 ) 
    assert np.allclose(Hams.Gamma(1, T = 5, gamma_s = 0.2), 0.46305440448285066)
    assert np.allclose(Hams.Gamma(1, T = 5, w_c = 0.5, gamma_s = 0.1), 0.16649945099163806)
    assert np.allclose(Hams.Gamma(1, T = 1, w_c = 0.5, gamma_s = 0.2), 0.16698133234966612)

def test_R_pq():
    assert np.allclose(Hams.R_pq(1000, T = 1, w_c = 1, gamma_s = 0.1), 1.7399659187787281) 
    assert np.allclose(Hams.R_pq(0, T = 1, w_c = 1, gamma_s = 0.1), 0)
    assert np.allclose(Hams.R_pq(1, T = 1, w_c = 1, gamma_s = 0.1), 0.84045322611412772 )
    assert np.allclose(Hams.R_pq(1, T = 5, w_c = 1, gamma_s = 0.1), 4.4664556343627677)
    assert np.allclose(Hams.R_pq(1, T = 1, w_c = 1, gamma_s = 0.2), 0.83984101446133159)
    assert np.allclose(Hams.R_pq(1, T = 5, w_c = 1, gamma_s = 0.2), 4.4627370355352005)
    assert np.allclose(Hams.R_pq(1, T = 5, w_c = 0.5, gamma_s = 0.2), 3.3177230510018835)

def test_R_qq():
    assert np.allclose(Hams.R_qq(1000, T = 1, w_c = 1, gamma_s = 0.1), 0.28645833333333333) 
    assert np.allclose(Hams.R_qq(0, T = 1, w_c = 1, gamma_s = 0.1), 0)
    assert np.allclose(Hams.R_qq(1, T = 1, w_c = 1, gamma_s = 0.1), 0.015463033098088407 )
    assert np.allclose(Hams.R_qq(1, T = 5, w_c = 1, gamma_s = 0.1), 0.082669850004023584)
    assert np.allclose(Hams.R_qq(1, T = 1, w_c = 1, gamma_s = 0.2), 0.031960910051200621)
    assert np.allclose(Hams.R_qq(1, T = 5, w_c = 1, gamma_s = 0.2), 0.17087227474482244)
    assert np.allclose(Hams.R_qq(1, T = 5, w_c = 0.5, gamma_s = 0.2), 0.058170929646053914)

def test_R_pp():
    assert np.allclose(Hams.R_pp(1000, T = 1, w_c = 1, gamma_s = 0.1), 15.947151495775877) 
    assert np.allclose(Hams.R_pp(0, T = 1, w_c = 1, gamma_s = 0.1), 0)
    assert np.allclose(Hams.R_pp(1, T = 1, w_c = 1, gamma_s = 0.1), 12.504507760489013)
    assert np.allclose(Hams.R_pp(1, T = 5, w_c = 1, gamma_s = 0.1), 59.297993397077693)
    assert np.allclose(Hams.R_pp(1, T = 1, w_c = 1, gamma_s = 0.2), 5.8663938861237142)
    assert np.allclose(Hams.R_pp(1, T = 5, w_c = 1, gamma_s = 0.2), 27.598510762727389)
    assert np.allclose(Hams.R_pp(1, T = 5, w_c = 0.5, gamma_s = 0.2), 38.292660937228488)
    #assert np.allclose(Hams.R_pq(1, T = 5, w_c = 0.2, gamma_s = 0.2), 22.415604265092263)
 
    
def test_GQFPE_S():
#    GQFPE_SH = Hams.GQFPE_SH(**params)
#    GQFPE_exact = np.zeros((4,4), dtype=np.complex128)
#    A_coeff = -1j * params["gamma"]/params["hbar"] * (1 - 2*params["gamma_s"])
#    D_pq = 2 * params["T"]/hbar**2 * (1-4*params["gamma_s"])/(1-2*params["gamma_s"])**2
#    D_pp = -2 * params["m"] * params["gamma"i]
    GQFPE_S = Hams.GQFPE_S(**params)
    GQFPE_S_exact = np.zeros((4,4), dtype=np.complex128)
    A = -0.125j
    D_pq = 0.1 * 1.7399659187787281
    D_pp = -.01 * 15.947151495775877
    D_qq = -0.28645833333333333
    # for i in range(4):
    #    GQFPE_SH_exact[i][i] += D_pp + D_qq

    GQFPE_S_exact[0][0] = 1j*A + D_pp + D_qq
    GQFPE_S_exact[0][3] = 1j*A -D_pp - D_qq
    GQFPE_S_exact[1][1] = 1j*D_pq + D_pp + D_qq +18/16*1j
    GQFPE_S_exact[1][2] = -D_pp + 1j*D_pq + D_qq
    GQFPE_S_exact[2][1] = -D_pp -1j*D_pq + D_qq
    GQFPE_S_exact[2][2] = D_pp -1j*D_pq + D_qq -18/16*1j
    GQFPE_S_exact[3][0] = -1j*A -D_pp -D_qq
    GQFPE_S_exact[3][3] = -1j*A + D_pp + D_qq
    print(GQFPE_S_exact)
    print(GQFPE_S)
    print(GQFPE_S_exact - GQFPE_S)
    print(D_pq,"D_pq")
    print(D_pp, "D_pp")
    print(D_qq, "D_qq")
    assert np.allclose(GQFPE_S_exact, GQFPE_S)


def test_GQFPE():
#    GQFPE_SH = Hams.GQFPE_SH(**params)
#    GQFPE_exact = np.zeros((4,4), dtype=np.complex128)
#    A_coeff = -1j * params["gamma"]/params["hbar"] * (1 - 2*params["gamma_s"])
#    D_pq = 2 * params["T"]/hbar**2 * (1-4*params["gamma_s"])/(1-2*params["gamma_s"])**2
#    D_pp = -2 * params["m"] * params["gamma"i]
    GQFPE = Hams.GQFPE(1, **params_2)
    GQFPE_exact = np.zeros((4,4), dtype=np.complex128)
    A = -1j * 0.1 * 0.16698133234966612 
    D_pq = 0.1 * 3.3177230510018835
    D_pp = -.01 * 38.292660937228488
    D_qq = -0.058170929646053914
    H_val = -1j/4 * ( -2 - 2/(1-0.014387677966970687*0.4))
    
    # for i in range(4):
    #    GQFPE_SH_exact[i][i] += D_pp + D_qq

    GQFPE_exact[0][0] = 1j*A + D_pp + D_qq
    GQFPE_exact[0][3] = 1j*A -D_pp - D_qq
    GQFPE_exact[1][1] = 1j*D_pq + D_pp + D_qq + H_val
    GQFPE_exact[1][2] = -D_pp + 1j*D_pq + D_qq
    GQFPE_exact[2][1] = -D_pp -1j*D_pq + D_qq
    GQFPE_exact[2][2] = D_pp -1j*D_pq + D_qq -H_val
    GQFPE_exact[3][0] = -1j*A -D_pp -D_qq
    GQFPE_exact[3][3] = -1j*A + D_pp + D_qq
    print(GQFPE_exact)
    print(GQFPE)
    print(GQFPE_exact - GQFPE)
    print(H_val, "H_val")
    print(A, "A")
    print(D_pq,"D_pq")
    print(D_pp, "D_pp")
    print(D_qq, "D_qq")
    assert np.allclose(GQFPE_exact, GQFPE)
#def test_Rp
