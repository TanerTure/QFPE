"""This SHO.py module generates the matrix form for several common operators in the energy eigenbasis of the harmonic oscillator, and position space functions.

Functions within can generate :math:`\\hat{x}, \\hat{p}, \\hat{a}^+, \\hat{a}, \\hat{H}`, and second moments of :math:`\\hat{x}` and :math:`\\hat{p}\\ (\\hat{x}^2,\\hat{x}\\hat{p}`, :math:`\\hat{p}\\hat{x}, \\hat{p}^2)`.

Example
-------
>>> import SHO
>>> SHO.a(3)
array([[0.        +0.j, 1.        +0.j, 0.        +0.j],
       [0.        +0.j, 0.        +0.j, 1.41421356+0.j],
       [0.        +0.j, 0.        +0.j, 0.        +0.j]])   
>>> SHO.a_dagger(3)
array([[0.        +0.j, 0.        +0.j, 0.        +0.j],
       [1.        +0.j, 0.        +0.j, 0.        +0.j],
       [0.        +0.j, 1.41421356+0.j, 0.        +0.j]])

The matrices above are used to calculate the energy using the relation :math:`H = a^+a + 1/2`

>>> import numpy as np
>>> SHO.a_dagger(3) @ SHO.a(3) + 0.5 * np.eye(3)
array([[0.5+0.j, 0. +0.j, 0. +0.j],
       [0. +0.j, 1.5+0.j, 0. +0.j],
       [0. +0.j, 0. +0.j, 2.5+0.j]])
        
"""
import math

import numpy as np
import numba
import sympy as sp
import matplotlib.pyplot as plt

def a(n=100):
    """
    Make the matrix representation of the lowering operator :math:`a`

    Parameters
    ----------
    n : int, optional
        length and width of the resultant square matrix

    Returns
    -------
    out : complex ndarray
        matrix of size n by n that represents the lowering operator :math:`\hat{a}`

    Raises
    ------
    ValueError
        If n is less than or equal to 1

    See Also
    --------
    SHO : overall description of module
    a_dagger : raising operator

    Notes
    -----
    The matrix :math:`\\hat{a}` in the eigenbasis of the Hamiltonian reads:
    
    .. math::

        \\hat{a} = \\begin{bmatrix}
            0 & \\sqrt{1} & 0 \\\\ 
            0 & 0        & \\sqrt{2} \\\\
            0 &   0      & 0 \\\\
            \\end{bmatrix}

    """
    if n <= 1:
        raise ValueError("n should be 2 or greater")
    matrix = np.zeros((n, n), dtype=np.complex128)
    for i in range(n - 1):
        matrix[i, i + 1] = sp.sqrt(i + 1)
    return matrix


def a_dagger(n=100):
    """
    Make the matrix representation of the raising operator :math:`\\hat{a}^\\dagger`

    Parameters
    ----------
    n : int, optional
        length and width of the resultant square matrix

    Returns
    -------
    out : complex ndarray
        matrix of size n by n that represents the raising operator :math:`\\hat{a}^\\dagger`

    Raises
    ------
    ValueError
        If n is less than or equal to 1

    See Also
    --------
    SHO : overall description of harmonic oscillator functions
    a : lowering operator

    Notes
    -----
    The matrix :math:`a^\\dagger` in the eigenbasis of the Hamiltonian reads
    
    .. math::

         a^\\dagger = \\begin{bmatrix}
             0 & 0 & 0 \\\\
             \\sqrt{1} & 0 & 0 \\\\
             0 &   \\sqrt{2} & 0 \\\\
             \\end{bmatrix}

    """
    if n <= 1:
        raise ValueError("n should be 2 or greater")
    matrix = np.zeros((n, n), dtype=np.complex128)
    for i in range(n - 1):
        matrix[i + 1, i] = np.sqrt(i + 1)
    return matrix


def p(n=100, w=1, hbar=1, m=1):
    """
    Make the matrix representation of the momentum operator :math:`\\hat{p}`

    Parameters
    ----------
    n : int, optional
        length and width of the resultant square matrix
    w : float, optional
        frequency :math:`\\omega` of the harmonic oscilator
    hbar : float, optional
        numerical value of physical constant :math:`\\hbar`
    m : float, optional
        mass :math:`m` of harmonic oscillator

    Returns
    -------
    out : complex ndarray
        matrix of size n by n that represents the momentum operator :math:`\\hat{p}`

    Raises
    ------
    ValueError
        If n is less than or equal to 1, or w, hbar, m <= 0

    See Also
    --------
    SHO : overall description of harmonic oscillator functions
    x : position operator
    

    Notes
    -----
    The matrix :math:`\\hat{p}` in the eigenbasis of the Hamiltonian reads
    
    .. math::

         \\hat{p}  =i\\sqrt{\\frac{m\\hbar\\omega}{2}} \\begin{bmatrix}
             0 & -\\sqrt{1} & 0 \\\\
             \\sqrt{1} & 0 & -\\sqrt{2} \\\\
             0 &   \\sqrt{2} & 0 \\\\
             \\end{bmatrix}

    """

    # if too time-consuming, may look to improve
    if n <= 1:
        raise ValueError("n should be 2 or greater")
    if any([w <= 0, hbar <= 0, m <= 0]):
        raise ValueError(
            " dimensional constants w,hbar, m should not be zero or negative"
        )
    matrix = -1j * (a(n=n) - a_dagger(n=n)) * np.sqrt(m * hbar * w / 2)
    return matrix


def x(n=100, w=1, hbar=1, m=1):
    """
    Make the matrix representation of the position operator :math:`\\hat{x}`

    Parameters
    ----------
    n : int, optional
        length and width of the resultant square matrix
    w : float, optional
        frequency :math:`\\omega` of the harmonic oscilator
    hbar : float, optional
        numerical value of physical constant :math:`\\hbar`
    m : float, optional
        mass :math:`m` of harmonic oscillator

    Returns
    -------
    out : complex ndarray
        matrix of size n by n that represents the position operator :math:`\\hat{x}`

    Raises
    ------
    ValueError
        If n is less than or equal to 1, or w, hbar, m <= 0

    See Also
    --------
    SHO : overall description of harmonic oscillator functions
    p : momentum operator
    

    Notes
    -----
    The matrix :math:`\\hat{x}` in the eigenbasis of the Hamiltonian reads
    
    .. math::

         \\hat{x}  =\\sqrt{\\frac{\\hbar}{2m\\omega}} \\begin{bmatrix}
             0 & \\sqrt{1} & 0 \\\\
             \\sqrt{1} & 0 & \\sqrt{2} \\\\
             0 &   \\sqrt{2} & 0 \\\\
             \\end{bmatrix}

    """
    
    if n <= 1:
        raise ValueError("n should be 2 or greater")
    if any([w <= 0, hbar <= 0, m <= 0]):
        raise ValueError(
            " dimensional constants w,hbar, m should not be zero or negative"
        )
    matrix = np.sqrt(hbar / m / w / 2) * (a(n=n) + a_dagger(n=n))
    return matrix


def H(n=100, w=1, hbar=1, m=1):
    """
    Make the matrix representation of the Hamiltonian operator :math:`\\hat{H}`

    Parameters
    ----------
    n : int, optional
        length and width of the resultant square matrix
    w : float, optional
        frequency :math:`\\omega` of the harmonic oscilator
    hbar : float, optional
        numerical value of physical constant :math:`\\hbar`
    m : float, optional
        mass :math:`m` of harmonic oscillator

    Returns
    -------
    out : complex ndarray
        matrix of size n by n that represents the Hamiltonian operator :math:`\\hat{H}`

    Raises
    ------
    ValueError
        If n is less than or equal to 1, or w, hbar, m <= 0

    See Also
    --------
    SHO : overall description of harmonic oscillator functions
    p : momentum operator
    x : position operator
    

    Notes
    -----
    The matrix :math:`\\hat{H}` in the eigenbasis of the Hamiltonian reads
    
    .. math::

         \\hat{H}  = \\hbar \\omega \\begin{bmatrix}
             \\frac{1}{2} & 0 & 0 \\\\
              0 & \\frac{3}{2} & 0 \\\\
             0 &   0 & \\frac{5}{2} \\\\
             \\end{bmatrix}

    """
    p_matrix = p(
        n=n + 1,
        w=w,
        hbar=hbar,
        m=m,
    )
    x_matrix = x(
        n=n + 1,
        w=w,
        hbar=hbar,
        m=m,
    )
    return (
        np.linalg.matrix_power(p_matrix, 2) / (2 * m)
        + 1 / 2 * m * w**2 * np.linalg.matrix_power(x_matrix, 2)
    )[:-1, :-1]


def H_eff(n=100, w=1, hbar=1, m=1, m_es = .8):
    '''
    Returns H_eff, which has time-dependent mass dividing the KE term.
    '''
    p_matrix = p(
            n=n + 1,
            w=w,
            hbar=hbar,
            m=m
            )
    x_matrix = x(n=n + 1,
            w=w,
            hbar=hbar,
            m=m
            )
    return (
        np.linalg.matrix_power(p_matrix, 2) / (2 * m_es)
        + 1 / 2 * m * w**2 * np.linalg.matrix_power(x_matrix, 2)
    )[:-1, :-1]


def SHO_distribution(T, n=50, hbar=1, w=1, k=1):
    if T < 0:
        raise ValueError(" T should not be negative")
    if any([n <= 0, hbar <= 0, w <= 0, k <= 0]):
        raise ValueError(" size of matrix and physical constants should be positive")
    if T == 0:
        probabilities = np.zeros(n, dtype=np.float64)
        probabilities[0] = 1
        return probabilities

    probabilities = np.zeros(n, dtype=np.float64)
    energies = np.arange(n, dtype=np.float64)
    energies = (energies + 0.5) * (hbar * w)
    probabilities[:] = np.exp(-energies / (k * T))
    return probabilities / sum(probabilities)


def get_second_moments(n=100, w=1, hbar=1, m=1):
    """
    Get all second moments of :math:`\\hat{x}` and :math:`\\hat{p}` :math:`(\\hat{x}^2, \\hat{x}\\hat{p}, \\hat{p}\\hat{x}, \\hat{p}^2)`

    Parameters
    ----------
    n : int, optional
        length and width of the resultant square matrices
    w : float, optional
        frequnecy :math:`\\omega` of the harmonic oscillator
    hbar : float, optional
        value of the physical constant :math: `\\hbar`
    m : float, optional
        mass :math: `m` of harmonic oscillator
    
    Returns
    -------
    out : list of four complex ndarrays
        out[0] corresponds to :math:`\\hat{x}^2`, out[1] corresponds to :math:`\\hat{x}\\hat{p}`,
        out[2] corresponds to :math:`\\hat{p}\\hat{x}`, and out[3] corresponds to :math:`\\hat{p}^2`

    Raises
    ------
    ValueError: if `n` <= 1 or any of `w`,`hbar`,`m` <= 0

    Notes
    -----
    The matrices for the second-order operators in the basis of the eigenvectors of the Hamiltonian are given by:

    .. math::

        \\hat{x}^2 =  \\frac{\\hbar}{2m\\omega} \\begin{bmatrix} 
                       1 & 0 & \\sqrt{2} \\\\
                       0 & 3 & 0 \\\\
                       \\sqrt{2} & 0 & 5 
                       \\end{bmatrix}
        \\hat {x}\\hat{p} = \\frac{i\\hbar}{2} \\begin{bmatrix}
                            1 & 0 & -\\sqrt{2} \\\\
                            0 & 1 & 0 \\\\
                            \\sqrt{2} & 0 & 1
                            \\end{bmatrix}

    .. math::

        \\hat{p}\\hat{x} = -\\frac{i\\hbar}{2} \\begin{bmatrix}
                           1 & 0 & \\sqrt{2} \\\\
                           0 & 1 & 0 \\\\
                           -\\sqrt{2} & 0 & 1 
                           \\end{bmatrix}
        \\hat{p}^2 = \\frac{m\\hbar\\omega}{2} \\begin{bmatrix}
                           1 & 0 & -\\sqrt{2} \\\\
                           0 & 3 & 0 \\\\
                           -\\sqrt{2} & 0 & 5 
                           \\end{bmatrix}

    """
#    p_matrix = p(n=n + 1, w=w, hbar=hbar, m=m)
#    x_matrix = x(n=n + 1, w=w, hbar=hbar, m=m)
#    return (
#        (x_matrix @ x_matrix)[:-1, :-1],
#        (x_matrix @ p_matrix)[:-1, :-1],
#        (p_matrix @ x_matrix)[:-1, :-1],
#        (p_matrix @ p_matrix)[:-1, :-1],
#    )
    p_matrix = p(n=n, w=w, hbar=hbar, m=m)
    x_matrix = x(n=n, w=w, hbar=hbar, m=m)
    return (
        (x_matrix @ x_matrix),
        (x_matrix @ p_matrix),
        (p_matrix @ x_matrix),
        (p_matrix @ p_matrix),
    )

@numba.jit(forceobj=True)
def SHO_funcs (x_vals,n=0, w=1, hbar=1, m=1):
    """
    Get an eigenstate of the harmonic oscillator in position-space evaluated at x_vals.
    
    Parameters
    ----------
    x_vals : ndarray
        x_vals: 
    

    """
   # if any([w <= 0, hbar <= 0, m <= 0]):
   #     raise ValueError(
   #         " dimensional constants w,hbar, m should not be zero or negative"
   #     )
    gamma = m * w / hbar
    x_vals_weighted = x_vals * np.sqrt(gamma)
    herm_coefficients = np.zeros(n + 1)
    herm_coefficients[n] = 1
    return ( 1/(2 ** n * math.factorial(n)) **.5 * (gamma/np.pi)**.25 * np.exp(-1 * x_vals_weighted **2/2) 
        * np.polynomial.hermite.hermval(x_vals_weighted, herm_coefficients))

#dx = x[1]-x[0]
#dy= y[1]-y[0]
#rho_initial_1 = np.zeros((n,n),dtype=np.complex128)

@numba.jit(forceobj=True)
def to_basis_set(n, x_vals, y_vals, rho_coord, w=1, hbar=1, m=1):
    """ Transformation from coordinate space to energy eigenbasis
    
    """
    rho = np.zeros((n, n), dtype=np.complex128)
    dx = x_vals[1]- x_vals[0]
    dy = y_vals[1] - y_vals[0]
    for i in range(n):
        for j in range(i+1):
            x_SHO = SHO_funcs(x_vals, n=j, w=w, hbar=hbar, m=m).reshape(1, -1)
            y_SHO = SHO_funcs(y_vals, n=i, w=w, hbar=hbar, m=m).reshape(-1, 1)
            
            rho[i][j] = integrate((y_SHO@x_SHO) * rho_coord, dx) * dy
            rho[j][i] = np.conj(rho[i][j])
    return rho

@numba.jit(forceobj=True)
def to_coordinate_space(rho, left_limit=-10, right_limit=10, num_points=512):
    """ Transformation from energy eigenbasis to coordinate space
    
    """
    n = rho.shape[0]
    x = np.linspace(left_limit, right_limit, num_points)
    rho_coord = np.zeros((num_points, num_points), dtype=np.complex128)
    for i in range(n):
        for j in range(n):
            x_SHO = SHO_funcs(x,n=i).reshape(1,-1)
            y_SHO = SHO_funcs(x,n=j).reshape(-1,1)
            rho_coord += rho[j][i]*(y_SHO@x_SHO)
    return rho_coord

@numba.jit(forceobj=True)
def integrate(function, difference):
    # uses linear approximation for derivative
    
    return np.sum(function)*difference

#Marginals for two-d case only for now

def marginals(function, diff_x,diff_y):
    marginal_x = np.sum(function, axis=0)*diff_y
    marginal_y = np.sum(function, axis=1)*diff_x
    return marginal_x, marginal_y

def thermal_function(B, w, x, x_prime, m=1, hbar=1):
    return ( 1/partition_function(B, w, hbar=hbar) * np.sqrt(m*w/(2*np.pi*hbar*np.sinh(B*hbar*w))) 
            * np.exp(-m*w/(2*hbar*np.sinh(B*w*hbar)) * ((x**2+x_prime**2) * np.cosh(B*hbar*w)  - 2*x*x_prime)))

def partition_function(B, w, hbar = 1):
    #print(1/(2*np.sinh(B*w/2)))
    return 1/(2*np.sinh(B*w*hbar/2))

def make_contourplot(X,Y,Z,xlim=(-6,6),ylim=(-6,6), figname = "initial_condition"):
    fig,ax=plt.subplots(1,1)
    cp = ax.contourf(X, Y, Z, levels=30)
    ax.set_xlabel("X")
    ax.set_xlim(xlim)
    ax.set_ylabel("X'")
    ax.set_ylim(ylim)
    fig.colorbar(cp)
    fig.savefig("plotting/Figures/" + figname, dpi = 300, bbox_inches = "tight")
    plt.show()

# @numba.jit(forceobj=True)
# def make_SHO_func(X,Y,n = 0,const_ax="x"):
#     if(const_ax == "x"):
#         return SHO_funcs(Y,n=n)
#     else:
#         return SHO_funcs(X,n=n)
    
# @numba.jit(forceobj=True)
# def make_rho_initial_faster(rho_initial_, x, y):
#     for i in range(n):
#         for j in range(i+1):
#             x_SHO = make_SHO_func(x,y,n=i,const_ax="y").reshape(1,-1)
#             y_SHO = make_SHO_func(x,y,n=j,const_ax="x").reshape(-1,1)

#             #Z_x = make_SHO_func(X,Y,n=i,const_ax="y")
#            # Z_y = make_SHO_func(X,Y,n=j,const_ax="x")
#             #these may be switched; ok since
#             rho_initial[i][j] = integrate((y_SHO@x_SHO)*p,dx)*dy
#             rho_initial[j][i] = np.conj(rho_initial[i][j])
    
