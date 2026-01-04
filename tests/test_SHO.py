import os
import sys

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "QFPE"))
import SHO


def test_a():
    a = np.zeros((3, 3), dtype=np.complex128)
    a[0][1] = np.sqrt(1)
    a[1][2] = np.sqrt(2)
    assert (SHO.a(3) == a).all()
    try:
        SHO.a(1)
    except ValueError:
        assert True
    else:
        assert False


def test_a_dagger():
    a_dagger = np.zeros((3, 3), dtype=np.complex128)
    a_dagger[1][0] = np.sqrt(1)
    a_dagger[2][1] = np.sqrt(2)
    assert (SHO.a_dagger(3) == a_dagger).all()
    try:
        SHO.a_dagger(1)
    except ValueError:
        assert True
    else:
        assert False


def make_x(hbar=1, m=1, w=1):
    x = np.zeros((3, 3), dtype=np.complex128)
    x[0][1] = np.sqrt(1)
    x[1][2] = np.sqrt(2)
    x[1][0] = np.sqrt(1)
    x[2][1] = np.sqrt(2)
    x *= np.sqrt(hbar / m / w / 2)
    return x


def test_x():
    assert (SHO.x(3) == make_x()).all()
    assert (SHO.x(3, hbar=1, m=2, w=0.5) == make_x(hbar=1, m=2, w=0.5)).all()
    try:
        SHO.x(3, hbar=0)
    except ValueError:
        assert True
    else:
        assert False
    try:
        SHO.x(3, hbar=1, m=-1)
    except ValueError:
        assert True
    else:
        assert False


def make_p(hbar=1, m=1, w=1):
    p = np.zeros((3, 3), dtype=np.complex128)
    p[0][1] = -np.sqrt(1)
    p[1][2] = -np.sqrt(2)
    p[1][0] = np.sqrt(1)
    p[2][1] = np.sqrt(2)
    p *= 1j * np.sqrt(hbar * m * w / 2)
    return p


def test_p():
    assert (SHO.p(3) == make_p()).all()
    assert (SHO.p(3, hbar=1, m=2, w=2) == make_p(hbar=1, m=2, w=2)).all()
    try:
        SHO.p(3, hbar=0)
    except ValueError:
        assert True
    else:
        assert False
    try:
        SHO.p(3, hbar=1, m=-1)
    except ValueError:
        assert True
    else:
        assert False


def get_H(hbar=1, w=1):
    H = np.zeros((3, 3), dtype=np.complex128)
    H[0][0] = 0.5
    H[1][1] = 1.5
    H[2][2] = 2.5
    H *= hbar * w
    return H


def test_H():
    assert np.allclose(SHO.H(3), get_H(), atol=1e-15)
    assert np.allclose(SHO.H(3, hbar=2, w=1), get_H(hbar=2, w=1), atol=1e-15)
    try:
        SHO.H(0)
    except ValueError:
        assert True
    else:
        assert False
    try:
        SHO.H(3, hbar=0)
    except ValueError:
        assert True
    else:
        assert False


def get_second_moments(hbar=1, m=1, w=1):


    p_squared = np.zeros((3, 3), dtype=np.complex128)
    p_squared[0][0] = -1
    p_squared[0][2] = np.sqrt(2)
    p_squared[1][1] = -3
    p_squared[2][0] = np.sqrt(2)
    p_squared[2][2] = -2
    p_squared *= -hbar * m * w / 2

    x_squared = np.zeros((3, 3), dtype=np.complex128)
    x_squared[0][0] = 1
    x_squared[0][2] = np.sqrt(2)
    x_squared[1][1] = 3
    x_squared[2][0] = np.sqrt(2)
    x_squared[2][2] = 2
    x_squared *= hbar / m / w / 2

    xp = np.zeros((3, 3), dtype=np.complex128)
    xp[0][0] = 1
    xp[0][2] = -np.sqrt(2)
    xp[1][1] = 1
    xp[1][2] = 0
    xp[2][0] = np.sqrt(2)
    xp[2][2] = -2
    xp *= 1j * hbar / 2

    px = xp.conj().T

    return x_squared, xp, px, p_squared
#    p_squared[0][0] = -1
#    p_squared[0][2] = np.sqrt(2)
#    p_squared[1][1] = -3
#    p_squared[2][0] = np.sqrt(2)
#    p_squared[2][2] = -5
#    p_squared *= -hbar * m * w / 2
#
#    x_squared = np.zeros((3, 3), dtype=np.complex128)
#    x_squared[0][0] = 1
#    x_squared[0][2] = np.sqrt(2)
#    x_squared[1][1] = 3
#    x_squared[2][0] = np.sqrt(2)
#    x_squared[2][2] = 5
#    x_squared *= hbar / m / w / 2
#
#    xp = np.zeros((3, 3), dtype=np.complex128)
#    xp[0][0] = 1
#    xp[0][1] = 0
#    xp[0][2] = -np.sqrt(2)
#    xp[1][1] = 1
#    xp[1][2] = 0
#    xp[2][0] = np.sqrt(2)
#    xp[2][2] = 1
#    xp *= 1j * hbar / 2
#
#    px = xp.conj().T
#
    return x_squared, xp, px, p_squared


def test_get_second_moments():
    second_moments = get_second_moments()
    second_moments_SHO = SHO.get_second_moments(3)
    assert np.allclose(
        second_moments_SHO[0], second_moments[0], atol=1e-15
    )  # x_squared
    assert np.allclose(second_moments_SHO[1], second_moments[1], atol=1e-15)  # xp
    assert np.allclose(second_moments_SHO[2], second_moments[2], atol=1e-15)  # px
    assert np.allclose(
        second_moments_SHO[3], second_moments[3], atol=1e-15
    )  # p_squared


def test_SHO_distribution():
    try: 
        SHO.SHO_distribution(-1)
    except ValueError:
        assert True
    else:
        assert False
    try:
        SHO.SHO_distribution(1, hbar = 0)
    except ValueError:
        assert True
    else:
        assert False
        
    assert SHO.SHO_distribution(1, n=1) == [1]
    assert (SHO.SHO_distribution(0, n=2) == [1, 0]).all()
    assert np.allclose(
        SHO.SHO_distribution(1, n=2),
        [
            np.exp(-0.5) / (np.exp(-0.5) + np.exp(-1.5)),
            np.exp(-1.5) / (np.exp(-0.5) + np.exp(-1.5)),
        ],
    )
    partition_function = 0.5 / np.sinh(0.5)
    energies = np.linspace(0.5, 999.5, 1000)
    SHO_distribution = np.exp(-energies) / partition_function
    assert np.allclose(SHO.SHO_distribution(1, n=1000), SHO_distribution)
    T = 2
    partition_function = 0.5 / np.sinh(1 / 2 / T)
    SHO_distribution = np.exp(-energies / T) / partition_function
    assert np.allclose(SHO.SHO_distribution(T, n=1000), SHO_distribution)
    
def test_SHO_funcs():
    x_vals = np.linspace(-5, 5, 512)
    m, w, hbar = 1, 1, 1
    y_vals_test = SHO.SHO_funcs(x_vals,n=0, w=w, hbar=hbar, m=m)
    gamma = m * w / hbar
    x_vals *= np.sqrt(gamma)
    y_vals = (gamma/np.pi)**.25 * np.exp (-x_vals**2/2)
    assert np.allclose(y_vals,y_vals_test)
    
    m, w, hbar = 1, 2, 1
    x_vals = np.linspace(-5, 5, 512)
    y_vals_test = SHO.SHO_funcs(x_vals, n=1, w=w, hbar=hbar, m=m)
    gamma = m * w / hbar
    x_vals *= np.sqrt(gamma)
    y_vals = (gamma/np.pi)**.25 * np.exp(-x_vals**2/2) * np.sqrt(2) * x_vals
    # return ( 1/(2 ** n * math.factorial(n)) **.5 * (gamma/np.pi)**.25 * np.exp(-1 * x_vals **2/2) 
    #    * np.polynomial.hermite.hermval(x_vals, herm_coefficients))
    assert np.allclose(y_vals, y_vals_test)
    
    m, w, hbar = 2, 2, 1
    x_vals = np.linspace(-5, 5, 512)
    y_vals_test = SHO.SHO_funcs(x_vals, n=2, w=w, hbar=hbar, m=m)
    gamma = m * w / hbar
    x_vals *= np.sqrt(gamma)
    y_vals = (gamma/np.pi) **.25 * np.exp (-x_vals**2/2) * (4 * x_vals**2 - 2) / np.sqrt(8)
    assert np.allclose(y_vals, y_vals_test)
    
def test_to_basis_set():
    x_vals = np.linspace(-10, 10, 1024)
    wfn = SHO.SHO_funcs(x_vals, n=0)
    wfn = wfn.reshape(-1, 1)
    rho = wfn @ np.conj(wfn.T)
    rho_basis_exact = np.zeros((5, 5), dtype=np.float64)
    rho_basis_exact[0, 0] = 1
    rho_basis = SHO.to_basis_set(5, x_vals, x_vals, rho)
    assert np.allclose(rho_basis, rho_basis_exact)   
    
    wfn_4 = SHO.SHO_funcs(x_vals, n=4)
    wfn_4 = 1j*wfn_4.reshape(-1, 1)
    rho = 1/2 * (wfn_4 @ np.conj(wfn_4.T) + wfn @ np.conj(wfn.T)
            + wfn @ np.conj(wfn_4.T) + wfn_4 @ np.conj(wfn.T))
    rho_basis_exact = np.zeros((5, 5), dtype=np.complex128)
    rho_basis_exact[0, 0] = 0.5
    rho_basis_exact[4, 4] = 0.5
    rho_basis_exact[0, 4] = -0.5j
    rho_basis_exact[4, 0] = 0.5j
    rho_basis = SHO.to_basis_set(5, x_vals, x_vals, rho)
    print(rho_basis, "rho_basis")
    print(rho_basis_exact, "rho_basis_exact")
    assert np.allclose(rho_basis, rho_basis_exact)

def test_to_coordinate_space():
    x_vals = np.linspace(-10, 10, 1024)
    wfn = SHO.SHO_funcs(x_vals, n = 0).reshape(-1,1)
    rho_coord_exact = wfn @ np.conj(wfn.T)
    rho_basis = np.zeros((5,5), dtype=np.float64)
    rho_basis[0][0] = 1
    rho_coord = SHO.to_coordinate_space(rho_basis, num_points=1024)
    assert np.allclose(rho_coord, rho_coord_exact)
    
    wfn_3 = 1j*SHO.SHO_funcs(x_vals, n = 3).reshape(-1,1)
    rho_coord_exact = 1/2*(wfn @ np.conj(wfn_3.T) + wfn_3 @ np.conj(wfn.T) 
            + wfn @ np.conj(wfn.T) + wfn_3 @ np.conj(wfn_3.T))
    rho_basis = np.zeros((5,5), dtype=np.complex128)
    rho_basis[0][0] = .5
    rho_basis[3][3] = .5
    rho_basis[0][3] = -.5j
    rho_basis[3][0] = .5j
    rho_coord = SHO.to_coordinate_space(rho_basis, num_points=1024)
    print(rho_coord_exact - rho_coord)
    print(abs(rho_coord_exact - rho_coord).max())
    assert np.allclose(rho_coord_exact, rho_coord)


if __name__ == "__main__":
    pass   
