'''
Calculates expectation values
'''
import SHO

import numpy as np

def trace(rho):
    res = np.einsum('ii->', rho)
    if not np.allclose(res, 1):
        raise ValueError(f"trace is currently {res}, the calculation has blown up")
    return res
    

def x(rho, hbar=1, w=1, m=1, **kwargs):
  #  if rho.shape[1] == 1:
  #      n = round(np.sqrt(rho.shape[0]))
  #      rho = rho.reshape(n,n)
    
    res = np.einsum('ii->', SHO.x(rho.shape[0], hbar=hbar, w=w, m=m) @ rho)
    if np.abs(res) > 2:
        raise ValueError(f"x is currently {res}, the calculation has blown up")
    return res

def p(rho, hbar=1, w=1, m=1, **kwargs):
    return np.einsum('ii->', SHO.p(rho.shape[0], hbar=hbar, w=w, m=m) @ rho)

def p_squared(rho, hbar=1, w=1, m=1, **kwargs):
    _, _, _, pp = SHO.get_second_moments(rho.shape[0], hbar=hbar, w=w, m=m)
    return np.einsum('ii->', pp @ rho)

def x_squared(rho, hbar=1, w=1, m=1, **kwargs):
    xx, _, _, _ = SHO.get_second_moments(rho.shape[0], hbar=hbar, w=w, m=m)
    res = np.einsum('ii->', xx @ rho)
    if np.abs(res) > 10:
        raise ValueError(f"x_squared is currently {res}, the calculation has blown up")
    return np.einsum('ii->', xx @ rho)

def xp_px(rho, hbar=1, w=1, m=1, **kwargs):
    _, xp, px, _ = SHO.get_second_moments(rho.shape[0], hbar=hbar, w=w, m=m)
    return np.einsum('ii->', (xp + px) @ rho)

def xp_com(rho, hbar=1, w=1, m=1, **kwargs):
    _, xp, px, _ = SHO.get_second_moments(rho.shape[0], hbar=hbar, w=w, m=m)
    return np.einsum('ii->', (xp - px) @ rho)

def rho(rho, hbar=1, w=1, m=1, **kwargs):
    return rho

def purity(rho, hbar=1, w=1, m=1, **kwargs):
    return np.trace(rho@rho)

def trace_int(rho, U):
    res = np.einsum('ii->', rho)
    if not np.allclose(res, 1):
        raise ValueError(f"trace is currently {res}, the calculation has blown up")
    return res

def x_int(rho, U, hbar=1, w=1, m=1, **kwargs):
    res = np.einsum('ii->', np.conj(U.T) @ SHO.x(rho.shape[0], hbar=hbar, w=w, m=m) @ U @ rho)
    if np.abs(res) > 2:
        raise ValueError(f"x is currently {res}, the calculation has blown up")
    return res

def p_int(rho, U, hbar=1, w=1, m=1, **kwargs):
    return np.einsum('ii->', np.conj(U.T) @ SHO.p(rho.shape[0], hbar=hbar, w=w, m=m) @ U @ rho)

def p_squared_int(rho, U, hbar=1, w=1, m=1, **kwargs):
    _, _, _, pp = SHO.get_second_moments(rho.shape[0], hbar=hbar, w=w, m=m)
    return np.einsum('ii->', np.conj(U.T) @ pp @ U @ rho)

def x_squared_int(rho, U, hbar=1, w=1, m=1, **kwargs):
    xx, _, _, _ = SHO.get_second_moments(rho.shape[0], hbar=hbar, w=w, m=m)
    res = np.einsum('ii->', np.conj(U.T) @ xx @ U @ rho)
    if np.abs(res) > 10:
        raise ValueError(f"x_squared is currently {res}, the calculation has blown up")
    return res

def xp_px_int(rho, U, hbar=1, w=1, m=1, **kwargs):
    _, xp, px, _ = SHO.get_second_moments(rho.shape[0], hbar=hbar, w=w, m=m)
    return np.einsum('ii->', np.conj(U.T) @ (xp + px) @ U @ rho)


def xp_com_int(rho, U, hbar=1, w=1, m=1, **kwargs):
    _, xp, px, _ = SHO.get_second_moments(rho.shape[0], hbar=hbar, w=w, m=m)
    return np.einsum('ii->', np.conj(U.T) @ (xp - px) @ U @ rho)

def rho_int(rho, U, hbar=1, w=1, m=1, **kwargs):
    return rho

def check_unitary(U):
    assert np.allclose(U @ np.conj(U.T), np.eye(U.shape[0]))


expect_funcs = [
        trace,
        x,
        p,
        x_squared,
        p_squared,
        xp_px,
        xp_com,
        rho,
        purity
        ]
expect_funcs_int = [
        trace_int,
        x_int,
        p_int,
        x_squared_int,
        p_squared_int,
        xp_px_int,
        xp_com_int,
        rho_int,
        purity
        ]
