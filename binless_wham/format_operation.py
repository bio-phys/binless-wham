import numpy as np

def MBAR_to_WHAM(u_kn, index_ref):
    '''The function transforms a MBAR like energy input matrix into a WHAM
    like bias energy matrix.

    Parameters
    ----------
    u_kn : 2D numpy array
        Array containing energies.
    index_ref : int
        Index of reference potential.

    Return
    ------
    u_kn_WHAM : 2D numpy array
        Array containing bias energies.

    '''

    u_kn_WHAM = []
    for t in range(len(u_kn)):
        if t != index_ref:
            u_kn_WHAM.append(u_kn[t])

    u_kn_WHAM.append(u_kn[index_ref])
    u_kn_WHAM = np.array(u_kn_WHAM)
    u_kn_WHAM = (u_kn_WHAM - u_kn_WHAM[-1]).T
    return u_kn_WHAM
