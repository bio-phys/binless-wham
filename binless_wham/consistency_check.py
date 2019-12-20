import numpy as np

def swap_index(a, i1, i2):
    ''' This function swaps indices or keys for dictionaries. '''
    b = a.copy()
    b[i2], b[i1] = a[i1], a[i2]
    #a[[i2,i1]] = a[[i1,i2]]
    return b

def make_n_il_outoff_state_assignment(pre_n_il, nj, state_assignment):
    """Calculates the number of structures in bin l calculated in bias i

    Parameters
    ----------



    Returns
    -------
    n_il : 2D numpy array
    Array being a matix containing for every bias potential i the
    counts in bin l (count=1 for binlessWHAM).


    """

    n_il = pre_n_il.copy()

    for counts, i in enumerate(nj):
        frac_state = state_assignment[counts*i:(counts+1)*i]

        uni, c = np.unique(frac_state, return_counts=True)
        for ind, u in enumerate(uni):
            n_il[counts][u] = c[ind]

    return n_il


def consistency_check(n_il, p_WHAM_l, w_il):
    """Consistency measure from Zhu et al. 2011

    paper:
    Convergence and Error Estimation in Free Energy Calculation
    Using the Wheighted Histogram Analysis Method.

    Indices
    -------
    i : bias potential
    l : histogram bin

    Parameters
    ----------
    n_il : 2D numpy array
        Array being a matix containing for every bias potential i the
        counts in bin l (count=1 for binlessWHAM).
    p_WHAM_l : 1D numpy array
        Weight of bin (structure in binlessWHAM) l resulting from a
        WHAM calculation.
    w_il : 2D numpy array
        Energy contribution of bias potential i for structure l.


    Return
    ------
    nu_i : 1D numpy array
        Metric for consistency of simulation at bias potential i.
    """

    N_i = np.sum(n_il, axis=1) # sum along all bins (structures in binlessWHAM)
    number_biases = len(N_i)

    N_i = N_i.reshape(number_biases,1)

    # equation (6), TODO: norm by bin size is missing
    c_il = np.exp(-w_il)

    # equation (7)
    f_i = 1./np.sum(c_il*p_WHAM_l, axis=1)
    f_i = f_i.reshape(number_biases,1)

    ## equation (41)
    mat_1 = n_il/N_i * (np.log(n_il/N_i) - np.log(p_WHAM_l) - np.log(f_i) + w_il)
    mat_1[n_il == 0.] = 0. # lim x->0 x*log(x) = 0
    nu_i = np.sum(mat_1, axis=1)

    return nu_i

#def prep_and_calc_nu_i(n_wins, state_assignment, us_y0, nj, potential_function):
#
#    #list of all y0 ... u_y0
#    w_il = np.zeros((n_wins, len(np.unique(state_assignment))))
#    #doublewell_2d_us(x,y, x0=0.0,y0=0,kx=0.0, ky=0.0)
#
#    uni, c = np.unique(state_assignment, return_counts=True)
#
#    expanded_us_y0 = np.append(us_y0[0] - abs(us_y0[1] - us_y0[0]), us_y0)
#    expanded_us_y0 = np.append(expanded_us_y0, us_y0[-1] + abs(us_y0[-2] - us_y0[-1]))
#
#    for i, y0 in enumerate(us_y0):
#        for l, yl in enumerate(expanded_us_y0):
#            w_il[i][l] = potential_function(0,yl, x0=0.0,y0=y0,kx=0.0, ky=ky)
#
#    # This works for equalli distant bins
#    p_WHAM_l = np.exp(-rep1_pmf) *(abs(us_y0[0] - us_y0[1]))/np.sum(np.exp(-rep1_pmf)*(abs(us_y0[0] - us_y0[1])))
#
#    pre_n_il = np.zeros((n_wins, len(np.unique(state_assignment))))
#
#    n_il = make_n_il_outoff_state_assignment(pre_n_il, nj, state_assignment)
#    #n_il_swaped = swap_index(n_il, 8, 39)
#
#    # Freeing up space
#    pre_n_il = 0
#
#    nu_i = consistency_check(n_il_swaped, p_WHAM_l, w_il)
#
#    return nu_i
