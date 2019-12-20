from binless_wham import *
import pyprind
import pandas as pd



def u_harmonic_potential(x,x0,k):
    return k/2.0*(x-x0)**2


def prepare_wham_gromacs_bias(x0_ar, us_rc_ar_l, k_u=1000., verbose=False):
    '''
    Parameters
    ----------
    x0_ar: array
        Center of umbrella potentials
    us_rc_fn_l: list
        list of arrays
    k_u: float
        Spring constant in kJ/mol/nm
    verbose: Boolean

    Returns
    --------
    u_ar: array
        Input array for binless WHAM.
    '''

    n_wins = len(x0_ar)
    assert (n_wins == len(us_rc_ar_l))

    u_l = []

    bar = pyprind.ProgBar(n_wins, update_interval=15)
    #for win_i in pyprind.prog_bar(range(n_wins)):
    for win_i in range(n_wins):
        if verbose:
           print(win_i)
        x_ar = us_rc_ar_l[win_i]
        u_ar = np.zeros((x_ar.shape[0], n_wins+1))
        u_ar[:,0] = win_i

        for i in range(n_wins):
            u_ar[:, i+1] = u_harmonic_potential(x_ar[:,1],
                                                x0_ar[i], k_u)
        u_l.append(u_ar)
        bar.update()
    return np.vstack(u_l)


def load_gromacs_reaction_coord_files(us_path, n_wins, step=10, verbose=False):
    """
    Parameters
    ----------
    us_path: string
        Path to the xvg files with sampled reaction coordinate values
    n_wins: integer
        Number of umbrella runs
    step: integer
        Time interval for analysis
    verbose: Boolean
        Verbosity

    Outputs
    -------
    us_pull_l: list
        list of reaction coordinates values sampled in the umbrella runs
    """
    us_pull_l = []
    bar = pyprind.ProgBar(n_wins, update_interval=15)
    for win_i in (range(1, n_wins+1)):
        if verbose:
           print(win_i)
        us_pull_l.append(
        np.loadtxt(us_path.format(win_i), skiprows=17)[::step])
        bar.update(force_flush=False)
    return us_pull_l


def loop_gromacs_bias_calc(wham_inp_fn, us_rc_ar_l, x0_ar, recalc=False,
                           R=8.3144621, T=300., verbose=False):
    """
    Parameters
    ----------
    wham_inp_fn: str
       File for bias array
    us_rc_ar_l: list
       list of array, with bias arrays typically in units of kJ/mol
    R: float (optional)
        Gast constant; default= 8.3144621 J/mol/K
    T: float (optional)
       T in Kelvin
    recalc: Boolean
    verbose: Boolean

    Outputs
    -------
    u_beta: array
        Bias array (in kT)

    """

    if os.path.exists(wham_inp_fn) and not recalc:
        u_beta = np.load(wham_inp_fn)
    else:
        u_ar =  prepare_wham_gromacs_bias(x0_ar, us_rc_ar_l, k_u=1000.,
                                          verbose=verbose)
        # convert into units of kT
        u_ar[:,1:] = u_ar[:,1:] / (R*T/1000.)
        np.save(wham_inp_fn, u_ar)
    return u_ar


def loop_micro_state_assignment(traj_pickle_fn, us_traj_l, x0_ar, recalc=False,
                                return_state_def=True, window_names=None):
    """

    N.B.: window names are 1 based by default. Should this be changed?
    """

    if os.path.exists(traj_pickle_fn) and not recalc:
        state_assignment = pd.read_pickle(traj_pickle_fn)
        centers = boundary_states_1d(x0_ar)
    else:
        us_traj_d = {}
        n_wins = len(us_traj_l)

        if np.all(window_names) is None:
           window_names = np.arange(x0_ar.shape[0]) +1

        #window_names=np.arange(1,n_wins+1)

        for i, traj in enumerate(us_traj_l):
            us_traj_d[i+1] = traj[:,1].reshape((traj[:,1].shape[0],1))

        state_assignment, centers = assign_state_traj_stack(us_traj_d, x0_ar.reshape((x0_ar.shape[0],1)),
                        0, skip=1, window_names=window_names,
                        return_state_def=return_state_def)
        pd.to_pickle(state_assignment, traj_pickle_fn)
    return state_assignment, centers
