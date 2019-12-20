#!/usr/bin/env python

import numpy as np
try:
    import cPickle
except ImportError:
      import pickle
from scipy.optimize import fmin_bfgs, fmin_l_bfgs_b
import sys
import os
from builtins import range


def calc_wfk(gi,ni,wki):
    """processing of data to perform binlessWHAM calculations

    Parameters
    ----------
    gi : 1D numpy array
        vector of relative free energies, with lenght N-1
    ni : 1D numpy array
        vector with lenght equal to the number of biases (weighted by the inefficiency factor), containing their lenghts as elements
    wki: 2D numpy array
        matrix (5) in binlessWHAM.pdf containing the potential energy of structure k and bias i

    Returns
    -------
    gi : 1D numpy array
        vector of relative free energies, with lenght N
    wfk : 1D numpy array
        inverse weight from the product of matrix multiplication of w and f
    wfki : 2D numpy array
        components product of w and f

    """
    nfix = 2
    gj = np.append(gi,0.) # Here we assume that g_N is the fixed free energy
    #gj = np.append(gi[:nfix],0.)
    #gj = np.append(gj,gi[nfix:])
    fi = ni*np.exp(gj)
    wfki = fi*wki
    wfk = np.sum(wfki,axis=1) # sum over i, len(wfk) should be M
    return gj, wfk, wfki


def F(gi,ni,wki,Ii,Ii_kini):
    """formula (7) in binlessWHAM.pdf

    In the nomenclature i is an index for the bias and k is an index for the structure.

    Parameters
    ----------
    gi : 1D numpy array
        vector of relative free energies, with length N-1
    ni : 1D numpy array
        vector with lenght equal to the number of biases, containing their lenghts as elements
    wki: 2D numpy array
        matrix (5) in binlessWHAM.pdf containing the potential energy of structure k and bias i
    Ii : 1D numpy array
        vector with length equal to the number of biases, containing their inefficiency factors
        efficiency factor Ii = 1/(1+2*tau_i) with tau_i the correlation time in biased run i
    Ii_kini : 1D numpy array
        vector with length equal to the number of structures, containing the inefficiency of their respective run

    Returns
    -------
    np.sum(Ii_kini*np.log(wfk/Ii_kini),axis=0) - np.sum(Ii*gi*ni,axis=0) : 1D numpy array
        value of the funtion F

    """
    gi, wfk, wfki = calc_wfk(gi,ni,wki)
    return np.sum(Ii_kini*np.log(wfk/Ii_kini),axis=0) - np.sum(gi*ni,axis=0)


def grad_F(gi,ni,wki,Ii,Ii_kini):
    """gradient of F, formula (8) in binlessWHAM.pdf

    In the nomenclature i is an index for the bias and k is an index for the structure.

    Parameters
    ----------
    gi : 1D numpy array
        vector of relative free energies, with lenght N-1
    ni : 1D numpy array
        vector with lenght equal to the number of biases, containing their lenghts as elements
    wki: 2D numpy array
        matrix (5) in binlessWHAM.pdf containing the potential energy of structure k and bias i
    Ii : 1D numpy array
        vector with length equal to the number of biases, containing their inefficiency factors
        efficiency factor Ii = 1/(1+2*tau_i) with tau_i the correlation time in biased run i
    Ii_kini : 1D numpy array
        vector with length equal to the number of structures, containing the inefficiency of their respective run

    Returns
    -------
    grad[:-1] : 1D numpy array
        F gradient as vector for components 1 ... N-1

    """
    gi, wfk, wfki = calc_wfk(gi,ni,wki)
    wfkiT = wfki.transpose()
    grad = np.sum(Ii_kini * wfkiT/wfk,axis=1)-ni # sum over k, len should be N
    return grad[:-1] # Here we assume that g_N is the fixed free energy


def hessian_F(gi, ni, wki, Ii, Ii_kini):
    """Hessian matrix of F

    In the nomenclature i is an index for the bias and k is an index for the structure.

    Parameters
    ----------
    gi : 1D numpy array
        vector of relative free energies, with lenght N-1
    ni : 1D numpy array
        vector with lenght equal to the number of biases, containing their lenghts as elements
    wki: 2D numpy array
        matrix (5) in binlessWHAM.pdf containing the potential energy of structure k and bias i
    Ii : 1D numpy array
        vector with length equal to the number of biases, containing their inefficiency factors
        efficiency factor Ii = 1/(1+2*tau_i) with tau_i the correlation time in biased run i
    Ii_kini : 1D numpy array
        vector with length equal to the number of structures, containing the inefficiency of their respective run

    Returns
    -------
    hessian : 2D numpy array
        Hessian of F for components 1 ... N (later only components 1 ... N-1 will be used for covariance)

    """

    gi, wfk, wfki = calc_wfk(gi,ni,wki)
    wfkiT = wfki.transpose()

    # General form of the Hessian
    hessian = np.diag(np.sum(Ii_kini*wfkiT/wfk,axis=1)) - np.dot(Ii_kini*wfkiT/wfk, np.transpose(wfkiT/wfk))

    # Simpler form of the Hessian at the optimum (not tested yet!)
    # hessian = np.diag(Ii*ni) - np.dot(Ii_kini*wfkiT/wfk, np.transpose(wfkiT/wfk))

    return hessian


def covariance_windows_free_energy(gi, ni, wki, Ii, Ii_kini, verbose=False):
    """Uncertainty of relative free energies of the windows (i.e., runs with different bias)

    Parameters
    ----------
    gi : 1D numpy array
        vector of relative free energies, with lenght N-1
    ni : 1D numpy array
        vector with lenght equal to the number of biases, containing their lenghts as elements
    wki: 2D numpy array
        matrix (5) in binlessWHAM.pdf containing the potential energy of structure k and bias i
    Ii : 1D numpy array
        vector with length equal to the number of biases, containing their inefficiency factors
        efficiency factor Ii = 1/(1+2*tau_i) with tau_i the correlation time in biased run i
    Ii_kini : 1D numpy array
        vector with length equal to the number of structures, containing the inefficiency of their respective run
    verbose:


    Returns
    -------
    c : 2D numpy array
        covariance of uncertainties of various windows
        c_ij = < delta g_i delta g_j >
    sigma : 2D numpy array
        uncertainties for free-energy differences between the relative weights gi and gj

    """

    # Inverse of the restricted Hessian
    #h = hessian_F(gi,ni,wki)[:-1,:-1]
    h = hessian_F(gi, ni, wki, Ii, Ii_kini)[:-1,:-1] # Here we assume that g_N is the fixed free energy
    if verbose:
        print("Hessian")
        print(h)
    c = np.linalg.inv(h)

    # Uncertainties for differences between different weights
    sigma = np.zeros(c.shape)
    for i in range(sigma.shape[0]):
        for j in range(sigma.shape[1]):
            sigma[i,j] = np.sqrt( c[i,i]-2*c[i,j]+c[j,j] )

    return c, sigma


def grad_p(gi,ni,wki,Ii,Ii_kini):
    """Gradient in free energies gi of probability p

    Parameters
    ----------
    gi : 1D numpy array
        vector of relative free energies, with lenght N-1
    ni : 1D numpy array
        vector with lenght equal to the number of biases, containing their lenghts as elements
    wki: 2D numpy array
        matrix (5) in binlessWHAM.pdf containing the potential energy of structure k and bias i
    Ii : 1D numpy array
        vector with length equal to the number of biases, containing their inefficiency factors
        efficiency factor Ii = 1/(1+2*tau_i) with tau_i the correlation time in biased run i
    Ii_kini : 1D numpy array
        vector with length equal to the number of structures, containing the inefficiency of their respective run

    Returns
    -------
    gp : 1D numpy array
        array containing the gradient in free energies gi of probability p


    """
    gi, wfk, wfki = calc_wfk(gi,ni,wki)
    #gp = -Ii_kini**2 *(wfki/Ii).T/wfk**2
    gp = - np.outer(Ii_kini, Ii).T*(wfki/Ii).T/wfk**2
    return gp


def grad_lnp(gi,ni,wki,Ii):
    """gradient in free energies gi of logarithmic probability p

    Parameters
    ----------
    gi : 1D numpy array
        vector of relative free energies, with lenght N-1
    ni : 1D numpy array
        vector with lenght equal to the number of biases, containing their lenghts as elements
    wki: 2D numpy array
        matrix (5) in binlessWHAM.pdf containing the potential energy of structure k and bias i
    Ii : 1D numpy array
        vector with length equal to the number of biases, containing their inefficiency factors
        efficiency factor Ii = 1/(1+2*tau_i) with tau_i the correlation time in biased run i

    Returns
    -------
    glnp : 1D numpy array
        array containing the gradient in free energies gi of logarithmic probability p


    """

    gi, wfk, wfki = calc_wfk(gi,ni,wki)
    glnp = -wfki.T/wfk

    return glnp


def calc_delta_p(gi,ni,wki,Ii,Ii_kini, calc_lnp=False, verbose=False):
    """Calculates the error in logaritmic probabilities

    Parameters
    ----------
    gi : 1D numpy array
        vector of relative free energies, with lenght N-1
    ni : 1D numpy array
        vector with lenght equal to the number of biases, containing their lenghts as elements
    wki: 2D numpy array
        matrix (5) in binlessWHAM.pdf containing the potential energy of structure k and bias i
    Ii : 1D numpy array
        vector with length equal to the number of biases, containing their inefficiency factors
        efficiency factor Ii = 1/(1+2*tau_i) with tau_i the correlation time in biased run i
    Ii_kini : 1D numpy array
        vector with length equal to the number of structures, containing the inefficiency of their respective run

    Returns
    -------
    delta_lnp : 1D numpy array
        array containing the error of the (logarithmic) probability p


    """
    if calc_lnp:
        # Gradient in free energies gi of logarithmic probability p
        gradient = grad_lnp(gi,ni,wki,Ii)
    else:
        # Default: gradient in free energies gi of probability p
        gradient = grad_p(gi, ni, wki, Ii, Ii_kini)

    # Uncertainty of relative free energies of the windows (i.e., runs with different bias)
    c, sigma = covariance_windows_free_energy(gi, ni, wki, Ii, Ii_kini, verbose=verbose)

    # Variance
    variance = np.zeros(gradient.shape[1])
    for a in range(c.shape[0]):
        for b in range(c.shape[1]):
            variance += (gradient[a]*gradient[b])*c[a,b]

    # End result
    delta_lnp = np.sqrt(variance)

    return delta_lnp


def read_input_from_file(input_NM):
    """Processes information in the input file

    Parameters
    ----------
    input_Nm : string
        path to input file with name of input file

    Returns
    -------
    M : 2D numpy list
        list containing number of bias, windows, bias energy for every structure in every bias and optionally correlation time


    """

    # reads in the input file and saves complete data in array M
    with open(input_NM) as file_NM:
        M = [i.strip().split() for i in file_NM]

    return M


def process_counts_windows(bias_def, tau=None):
    """
    Setup nj, number of points/structures sampled in every window. The counts nj
    can be corrected using estimates for the correlation time in the windows.

    Parameters
    ----------
    n_wins: int
         number of biased simulations to analyze
    ukj: array
         matrix (5) in binlessWHAM.pdf containing the potential energy of structure k and bias j (=i)

    Returns
    -------
    nj : 1D numpy array
        vector with lenght equal to the number of biases, containing their lengths as elements
    Ii : 1D numpy array
        vector with length equal to the number of biases, containing their inefficiency factors
        efficiency factor Ii = 1/(1+2*tau_i) with tau_i the correlation time in biased run i
    Ii_kini : 1D numpy array
        vector with length equal to the number of structures, containing the inefficiency of their respective run

    """

    # bias = first line in input file stating the number of bias used
    #bias = int(M[0][0])
    # correlation time
    #if len(M[0]) == bias + 1:
    #    t = np.array(M[0][1:],dtype=float)
    #else:
    #    t = np.zeros(bias)
    # creates an array nj having containing the number of biased energies


    # !!! OLD CODE !!! CAUTION: The way this is written it only works if every bias hase same number of energies
    #nj = np.array([len(M[1:])/float(M[0][0]) for i in range(bias)])
    #nj = np.array([len(u_kj[:,0])/float(n_wins)]*n_wins)
    # !!! NEW CODE !!! untested, should work with any number of structures per bias

    n_wins, nj = np.unique(bias_def, return_counts=True)
    if np.all(tau) is None:
        t = np.zeros(len(n_wins))
    else:
        t = tau

    # correlation time correction factor
    Ij = 1./(1.+2*t)
    # vector containing the correlation time correction factor of the bias in which target structure is produced.
    Ij_kinj = []
    for j,n in enumerate(nj):
        for nn in range(int(nj[j])):
            Ij_kinj.append(Ij[j])
    Ij_kinj = np.array(Ij_kinj)

    return nj, Ij, Ij_kinj


def run_wham_bfgs_structure_weights(nj, ukj, beta, Ij, Ij_kinj, verbose=False, norm_weights=True,
                                    lbfgs=False, subtract_min_bias=True):
    """
    Run numerical optimization to determine optimal weights of structures using WHAM.

    Parameters:
    -----------
    nj: array
        vector with lenght equal to the number of biases, containing their lengths as elements
    wkj: array
         matrix (5) in binlessWHAM.pdf containing the potential energy of structure k and bias j (=i)
    verbose: Boolean
         print the weights of the biased runs
    norm_weights: Boolean
         normalize the weights of the structures
    Ii : 1D numpy array
        vector with length equal to the number of biases, containing their inefficiency factors
        efficiency factor Ii = 1/(1+2*tau_i) with tau_i the correlation time in biased run i
    Ii_kini : 1D numpy array
        vector with length equal to the number of structures, containing the inefficiency of their respective run

    Returns:
    --------
    struc_weights: array
        weight of each structure
    """

    # Subtract the minimum bias energies to reduce the risk of over-/underflow
    if subtract_min_bias:
        ukjmin = np.amin(ukj, axis=0)
        ukj   -= ukjmin

    # Calculate the exponent of the bias energy
    wkj = np.exp(-beta*ukj)

    # Number of biased simulation runs (e.g. umbrella windows)
    n_wins = len(nj)

    # Initialize weights of biased simulation runs
    gj0 = np.ones(n_wins-1)

    # Weight number of structures per run by inefficiency factor
    nj = nj.astype('float') * Ij

    # Optimize the relative free energies of the windows
    if lbfgs:
        r = fmin_l_bfgs_b(F, gj0, grad_F, args=(nj, wkj, Ij, Ij_kinj), iprint=2)
        g = r[0]
        print(r[1:])
    else:
        g = fmin_bfgs(F, gj0, grad_F, args=(nj, wkj, Ij, Ij_kinj))

    # Non-normalized weights of all structures
    gj, wfk, wfki = calc_wfk(g,nj,wkj)
    struc_weights = Ij_kinj/wfk

    # Uncertainty of the structure weights
    sigma_struc_weights = calc_delta_p(g, nj, wkj, Ij, Ij_kinj, calc_lnp=False, verbose=verbose)

    # Normalization of weights and uncertainties
    if norm_weights:
        z = struc_weights.sum()
        struc_weights /= z
        sigma_struc_weights /= z

    # correct the free energies for the initial subtraction of the minimum bias energies
    if subtract_min_bias:
        gj += + ukjmin - ukjmin[-1]

    if verbose:
        print(gj)

    return struc_weights, sigma_struc_weights, gj


def calculate_pmf_from_structure_weights(p, sigma_p, state_assignment, Ij_kinj):
    """
    Calculate the potential of mean force for the states according to structure weights.

    Parameters:
    -----------
    p: array
         vector with the weights of all structures
    sigma_p: array
         vector with the uncertainties of the weights of all structures
    state_assignment: array
         array that contains the state indices of all structures
    Ii_kini : 1D numpy array
        vector with length equal to the number of structures, containing the inefficiency of their respective run

    Returns:
    --------
    pmf: array
        PMF (potential of mean force) for each state
    sigma_pmf: array
        uncertainty of the PMF for each state
    """

    # Initialize the histogram counter and the uncertainty
    counter  = np.zeros(np.max(state_assignment)+1) # assumes indices going from 0 to N
    sigma_px = np.zeros(np.max(state_assignment)+1)

    # Loop over all structures and add count and uncertainty to their states
    for k,s in enumerate(state_assignment):
        state = s
        counter[state]  += p[k] #* Ij_kinj[k]
        sigma_px[state] += sigma_p[k]

    # Calculate the PMF and its uncertainty from the histogram
    pmf = -np.log(counter)
    pmf -= np.min(pmf)
    sigma_pmf = sigma_px/counter

    return pmf, sigma_pmf


def run_wham_pmf(bias_name_k, ukj, state_assignment, tauj=None, beta=1., verbose=False, return_g=False,
                lbfgs=False):
    """
    Run WHAM with numerical optimization to calculate the potential of mean force for given states

    Parameters:
    -----------
    bias_name_k: array
         index of the bias simulations from structe k orginates
    ukj: array
         matrix (5) in binlessWHAM.pdf containing the potential energy of structure k and bias j (=i)
    state_assignment: array
         array that contains the microstate indices of all structures
    tauj : 1D numpy array
        vector with length equal to the number of biases, with tau_i the correlation time in biased run i
    beta: float
        inverse temperature
    verbose: Boolean
        Set verbosity of output

    Returns:
    --------
    pmf: array
        PMF (potential of mean force) for each state
    sigma_pmf: array
        uncertainty of the PMF for each state
    """

    # Calculate the independent counts/structures observed in each window
    nj, Ij, Ij_kinj = process_counts_windows(bias_name_k, tau=tauj)

    # Calculate the weights of each structure
    p, sigma_p, gj = run_wham_bfgs_structure_weights(nj, ukj, beta, Ij, Ij_kinj, norm_weights=True,
                                                    verbose=verbose, lbfgs=lbfgs)

    # Calculate the PMF from the structure weights
    pmf, sigma_pmf = calculate_pmf_from_structure_weights(p, sigma_p, state_assignment,
                                                          Ij_kinj)

    if return_g:
        return pmf, sigma_pmf, gj
    else:
        return pmf, sigma_pmf


def parse_first_line(fname):
    with open(fname, "r") as f:
        first_line = f.readline()
    first_line = first_line.rstrip('\n').split()
    n_wins = int(first_line[0])
    if len(first_line) > 1:
        tau = first_line[1:]
    else:
        tau = None
    return n_wins, tau


def main():

    input_NM = sys.argv[1]
    beta = 1.

    # First row: number of windows (optionally followed by correlation times of the windows)
    n_wins, tau = parse_first_line(input_NM)

    # Load bias energies (first column gives the bias to which the structure belongs)
    input_all = np.genfromtxt(input_NM, skip_header=1)
    bias_def  = input_all[:,0]
    u_kj      = input_all[:,1:]

    # Process window counts
    nj, Ij, Ij_kinj = process_counts_windows(bias_def, tau=tau)

    # Run WHAM
    p, p_sigma, gj = run_wham_bfgs_structure_weights(nj, u_kj, beta, Ij, Ij_kinj,
                                                          verbose=True, norm_weights=True)

    #final_struc_weights = 1./calc_wfk(gj_final[:-1],nj,wkj,Ij)[1]
    #print final_struc_weights

    np.savetxt(input_NM+"_FreeEnergies.out", gj)
    np.savetxt(input_NM+"_StatisticalWeights.out", p)


if __name__ == "__main__":
   main()
