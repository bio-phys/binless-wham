from collections import defaultdict, Counter
import numpy as np


def boundary_states_1d(init_ar_x):
    """
    Add bin/states at the edge of the reaction coordinate. These dummy states
    ensure that all the states/bin we are interested in have the same width in
    the calculation of a PMF.
    
    Parameters:
    -----------
    init_ar_x: array
        Bins/states states along the reaction coordinate
    
    Returns:
    --------
    states: array
        Bin/states with two boundary states.
    
    """
    dx = init_ar_x[1] - init_ar_x[0]
    low = init_ar_x[0] - dx
    high = init_ar_x[-1] + dx
    states = np.append(low, init_ar_x)
    states = np.append(states, high)
    return states
    
    
    
def which_bin(x, xc):
    """
    Assign a configuration to the closest bin/structural state.
    
    Parameters:
    -----------
    x: array
        Coordinates or order parameter
    xc: array
        Center of states
    
    Returns:
    --------
    traj_state_ar: array
        State assigned trajetory
    """
    return np.argmin(np.absolute(x-xc))


def assign_state_traj_ar(traj_st, states):
    return np.array([which_bin(tra, states) for tra in traj_st])


def assign_state_traj_stack(us_d, x0_y0_ar, coord_index, skip=1,
                            window_names=None, return_state_def=False): 
    """
    Assign a set of trajectories to the closest structural state/bin,
    using one of potentially many order parameters calculate for the 
    trajectories.    

    
    Parameters
    ----------
    
    Returns
    -------
    assigned_traj: array
        Trajectories assigned to microstates
    states: array
        Centers of states (including boundary states)
    -------
    """
    if np.all(window_names) is None:
       n_wins = len(us_d.items())
       window_names = range(n_wins)
     
    traj_st = np.vstack([us_d[i][::skip] for i in window_names])
    states = boundary_states_1d(x0_y0_ar[:,coord_index])
    assigned_traj = assign_state_traj_ar(traj_st[:,coord_index], states)

    if return_state_def:
       return assigned_traj, states
    else:
         return assigned_traj


def calculate_statistical_inefficiency_runs(traj_l):
    """
    Using fast autocorrelation calculation to estimate statistical inefficiency. This code wraps
    a function from pymbar.
    
    References
    ----------
    [1] Shirts MR and Chodera JD. Statistically optimal analysis of samples from
        multiple equilibrium states. J. Chem. Phys. 129:124105, 2008
        http://dx.doi.org/10.1063/1.2978177
    [2] J. D. Chodera, W. C. Swope, J. W. Pitera, C. Seok, and K. A. Dill. Use of the weighted
        histogram analysis method for the analysis of simulated and parallel tempering simulations.
        JCTC 3(1):26-41, 2007.
    """
    try:
        import pymbar
    except ImportError as err:
        err.args = (err.args[0] + "\n You need to install pymbar to use this function.",)
        raise
        
    iinv = np.array([pymbar.timeseries.statisticalInefficiency_fft(tra) for tra in traj_l])
    return (iinv - 1.0) / 2.0

