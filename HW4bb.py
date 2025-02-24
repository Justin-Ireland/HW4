import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import lognorm
from scipy.integrate import quad
from HW4a import tln_PDF
#endregionimports


def tln_pdf(D, mu, sig, F_DMin, F_DMax):
    """
    Calls tln_PDF.
    """
    return tln_PDF(D, mu, sig, F_DMin, F_DMax) #called from HW4a


def tln_cdf(D, mu, sig, D_Min, D_Max, F_DMin, F_DMax):
    """
    Finds (tln_cdf) truncated log-normal cumulative distribution function.
    """
    if D < D_Min: #if below min sieve size return 0
        return 0
    if D > D_Max: #upper bound if above sieve size return 1
        return 1
    cdf_val, _ = quad(tln_PDF, D_Min, D, args=(mu, sig, F_DMin, F_DMax)) #solves integration of tln_PDF and assigns to cdf_val.
    return (cdf_val - quad(tln_PDF, D_Min, D_Min, args=(mu, sig, F_DMin, F_DMax))[0]) / (F_DMax - F_DMin)
    #this truncates cdf_val subtracted by the integration of tln_PDF to find our final CDF


def main():
    """
    This main function receives user input (with defaults) to generate two plots
    to view the results of our truncated log-normal distribution.
    """

    mu = float(input("Enter the mean of ln(D): ") or np.log(2)) #user inputs for mean, stdev, min/max sieve size
    sig = float(input("Enter the standard deviation of ln(D): ") or 1.0)
    D_Min = float(input("Enter the lower sieve size (D_Min): ") or 0.375)
    D_Max = float(input("Enter the upper sieve size (D_Max): ") or 1.0)

    # integration of tln_pdf for area under curve within range from Dmin to Dmax
    F_DMin, _ = quad(lambda D: tln_PDF(D, mu, sig, 0, 1), 0, D_Min)
    F_DMax, _ = quad(lambda D: tln_PDF(D, mu, sig, 0, 1), 0, D_Max)

    # Generates 500 x-values for better plot resolution
    x = np.linspace(D_Min, D_Max, 500)
    pdf_vals = np.array([tln_pdf(d, mu, sig, F_DMin, F_DMax) for d in x]) #takes tln_pdf at each x-value
    cdf_vals = np.cumsum(pdf_vals) / np.sum(pdf_vals)  # makes sure CDF increases so we don't have a decreasing graph issue

    # Define truncation limit at 0.75 (75 percent) between D_Min and D_Max
    D_trunc = D_Min + (D_Max - D_Min) * 0.75
    P_trunc, _ = quad(tln_PDF, D_Min, D_trunc, args=(mu, sig, F_DMin, F_DMax)) #probability of any sample being < D_trunc by quad integration

    # Creates our subplots
    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(7, 8))

    # pdf plot and our settings for the plot
    axs[0].plot(x, pdf_vals, label='Truncated Log-Normal PDF') #x-axis label
    axs[0].fill_between(x[x <= D_trunc], pdf_vals[x <= D_trunc], color='grey', alpha=0.3) #fill area under curve
    axs[0].set_ylabel('f(D)') #y label
    axs[0].set_title('Truncated Log-Normal Distribution') #title
    axs[0].legend()

    # Adds annotation so we can see our 'truncation point'
    axs[0].annotate(f'P(D<{D_trunc:.2f})={P_trunc:.2f}', #displays the equation
                    xy=(D_trunc, tln_pdf(D_trunc, mu, sig, F_DMin, F_DMax) / 2),
                    xytext=(D_Min + (D_Max - D_Min) * 0.3, max(pdf_vals) * 0.7),
                    arrowprops=dict(arrowstyle='->'))

    # cdf plot
    axs[1].plot(x, cdf_vals, label='Truncated Log-Normal CDF')
    axs[1].scatter([D_trunc], [P_trunc], color='red', edgecolor='black', zorder=3)
    axs[1].hlines(P_trunc, D_Min, D_trunc, color='black', linewidth=1, linestyle='dashed') #sets settings for lines
    axs[1].vlines(D_trunc, 0, P_trunc, color='black', linewidth=1, linestyle='dashed') #black dashed 1unit wide lines for graph
    axs[1].set_xlabel('D')
    axs[1].set_ylabel('F(D)')
    axs[1].legend()

    plt.show() #displays plot when main is run


if __name__ == '__main__':
    main()
