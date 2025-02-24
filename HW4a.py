import math
import numpy as np
from scipy.integrate import quad
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
#end imports

def ln_PDF(D, mu, sig):
    """
    Computes f(D) for the log-normal probability density function.
    :param D: Rock diameter
    :param mu: Mean of ln(D)
    :param sig: Standard deviation of ln(D)
    :return: f(D)
    """
    if D <= 0: #gives back zero if D is negative
        return 0.0
    p = 1 / (D * sig * math.sqrt(2 * math.pi)) #constant coefficient before e^ (1st half of eq.)
    _exp = -((math.log(D) - mu) ** 2) / (2 * sig ** 2) #e^ term (2nd half of ln_PDF equation)
    return p * math.exp(_exp) #multiplies together for full equation

def tln_PDF(D, mu, sig, F_DMin, F_DMax):
    """
    Given equation provided which takes ln_PDF to find (truncated) tln_PDF.
    """
    return ln_PDF(D, mu, sig) / (F_DMax - F_DMin) #calls ln_PDF and truncated by dividing by sieve ranges

def F_tlnpdf(D, mu, sig, D_Min, D_Max, F_DMax, F_DMin):
    """
    Integrates tln_PDf D_Min to D.
    """
    if D > D_Max or D < D_Min: #if out of range we don't do the integration it just returns zero
        return 0
    P, _ = quad(tln_PDF, D_Min, D, args=(mu, sig, F_DMin, F_DMax)) #using quad to integrate tln_PDF by adaptive quadrature (sounds fancy)
    return P #outputs integral of tln_PDF solved with quad method from scipy


def makeSample(mu, sig, D_Min, D_Max, F_DMax, F_DMin, N=100):
    """
    Generate a sample based on the tln_PDF Using fsolve.
    """
    probs = np.random.rand(N)  # Generate random from numpy module

    # initial guess Uses midpoint of the range so I'm not always using D_Min
    initial_guess = (D_Min + D_Max) / 2

    d_s = []
    for p in probs:
        solution = fsolve(lambda D: F_tlnpdf(D, mu, sig, D_Min, D_Max, F_DMax, F_DMin) - p, initial_guess)


        if D_Min <= solution[0] <= D_Max: #checks if solution is within our sieve sizes
            d_s.append(solution[0]) #updates d_s if in range .375"-1"
        else:
            d_s.append(initial_guess)  #chatGPT added a "fallback" incase our "if" fails it goes back to midpoint to stay in range

    return d_s

def sampleStats(D):
    """
    Get mean and variance of the sample using the numpy module.
    """
    mean = np.mean(D)
    var = np.var(D, ddof=1)
    return mean, var

def getPreSievedParameters():
    """
    Get user input for mean and standard deviation of ln(D).
    """
    mean_ln = float(input("Mean of ln(D)? (default: ln(2)) ") or math.log(2)) #default ln(2)
    sig_ln = float(input("Standard deviation of ln(D)? (default: 1) ") or 1)
    return mean_ln, sig_ln

def getSieveParameters():
    """
    Get sieve parameters from the user.
    """
    D_Min = float(input("Small aperture size? (default: 3/8 inch) ") or (3.0 / 8.0)) #.375 Dmin
    D_Max = float(input("Large aperture size? (default: 1 inch) ") or 1.0) #1 inch Dmax
    return D_Min, D_Max

def getSampleParameters():
    """
    Get sample size parameters from the user.
    """
    N_samples = int(input("How many samples? (default: 11) ") or 11) #11df
    N_sampleSize = int(input("How many items in each sample? (default: 100) ") or 100) #100 samples
    return N_samples, N_sampleSize

def getFDMaxFDMin(mu, sig, D_Min, D_Max):
    """
    Finds F_DMax and F_DMin using the log-normal distribution.
    """
    F_DMax, _ = quad(ln_PDF, 0, D_Max, args=(mu, sig)) #integrates ln_PdF, assigns to F_Dmax (F signaling that it was integrated)
    F_DMin, _ = quad(ln_PDF, 0, D_Min, args=(mu, sig)) #integrates with D_Min (lower lim .375), assigns to F_Dmin
    return F_DMin, F_DMax

def main():
    """
    Simulates a gravel production process where rocks are sieved between two screens,
    generating random samples from a truncated log-normal distribution and computing
    their mean and variance based on the user input called from each of the functions above.
    """
    mean_ln, sig_ln = getPreSievedParameters() #the variables from lines 105-108 set the user inputs for main
    D_Min, D_Max = getSieveParameters()   #by calling from the functions that ask for user input ^^
    N_samples, N_sampleSize = getSampleParameters()
    F_DMin, F_DMax = getFDMaxFDMin(mean_ln, sig_ln, D_Min, D_Max) #gets our truncated

    Samples = [] #initial list of samples set at zero
    Means = [] #initial means set at zero
    for _ in range(N_samples):
        sample = makeSample(mean_ln, sig_ln, D_Min, D_Max, F_DMax, F_DMin, N=N_sampleSize) #generates sample
        Samples.append(sample) #uses append to update samples (until Nsample)
        mean, var = sampleStats(sample)
        Means.append(mean) #updates mean by appending
        print(f"Sample: mean = {mean:.3f}, variance = {var:.3f}")

    stat_of_Means = sampleStats(Means) #statistics of all means from the loop that generates samples and calculates means
    print(f"Mean of the sampling mean:  {stat_of_Means[0]:.3f}") #output calculated mean/variance of means
    print(f"Variance of the sampling mean:  {stat_of_Means[1]:.6f}")

if __name__ == '__main__':
    main()
