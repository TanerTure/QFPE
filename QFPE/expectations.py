'''
Calculates expectation values
'''
import SHO

import numpy as np

def trace(rho):
    return np.einsum('ii->', rho)


def x(rho, hbar=1, w=1, m=1):
  #  if rho.shape[1] == 1:
  #      n = round(np.sqrt(rho.shape[0]))
  #      rho = rho.reshape(n,n)
    return np.einsum('ii->', SHO.x(rho.shape[0], hbar=hbar, w=w, m=m) @ rho)

def p(rho, hbar=1, w=1, m=1):
    return np.einsum('ii->', SHO.p(rho.shape[0], hbar=hbar, w=w, m=m) @ rho)

def p_squared(rho, hbar=1, w=1, m=1):
    _, _, _, pp = SHO.get_second_moments(rho.shape[0], hbar=hbar, w=w, m=m)
    return np.einsum('ii->', pp @ rho)

def x_squared(rho, hbar=1, w=1, m=1):
    xx, _, _, _ = SHO.get_second_moments(rho.shape[0], hbar=hbar, w=w, m=m)
    return np.einsum('ii->', xx @ rho)

def xp_px(rho, hbar=1, w=1, m=1):
    _, xp, px, _ = SHO.get_second_moments(rho.shape[0], hbar=hbar, w=w, m=m)
    return np.einsum('ii->', (xp + px) @ rho)


