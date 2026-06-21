'''
Contains dictionary which maps a name to the corresponding system parameters

'''
import sympy as sp

params = { 
        "I": 
        { "gamma":0.1,
            "w_c" : 1,
            "T" : 1,
            "n" : 30,
            "hbar":1,
            "m" : 1,
            "w" : 1,
            },
        "1":
        { "gamma":0.1,
            "w_c" : 1,
            "T" : 1,
            "n" : 30,
            "hbar":1,
            "m" : 1,
            "w" : 1,
        "2":
        { "gamma": 0.1,
            "w_c" : 1,
            "T" : 5,
            "n" : 100,
            "hbar":1,
            "m" : 1,
            "w" : 1,

            },
        "II":
        { "gamma": 0.1,
            "w_c" : 1,
            "T" : 5,
            "n" : 100,
            "hbar":1,
            "m" : 1,
            "w" : 1,

            },
        "III":
        { "gamma": 0.1,
            "w_c": 1,
             "T": 0.2,
             "n": 30,
             "hbar" : 1,
             "m" : 1,
             "w" : 1,
            },
        "IV":
        { "gamma":0.1,
            "w_c": 1,
            "T" : 2,
            "n" : 50,
            "hbar" : 1,
            "m" : 1,
            "w" : 1
            }
        }


params_sp = { 
        "I": 
        { "gamma": sp.Rational(1,10),
            "w_c" : 1,
            "T" : 1,
            "n" : 30,
            "hbar":1,
            "m" : 1,
            "w" : 1,
            },
        "II":
        { "gamma": sp.Rational(1,10),
            "w_c" : 1,
            "T" : 5,
            "n" : 100,
            "hbar":1,
            "m" : 1,
            "w" : 1,

            },
        "III":
        { "gamma": sp.Rational(1,10),
            "w_c": 1,
             "T": sp.Rational(2, 10),
             "n": 30,
             "hbar" : 1,
             "m" : 1,
             "w" : 1,
            },
        "IV":
        { "gamma":sp.Rational(1,10),
            "w_c": 1,
            "T" : 2,
            "n" : 50,
            "hbar" : 1,
            "m" : 1,
            "w" : 1
            }
        }


