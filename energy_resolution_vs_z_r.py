import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from iminuit     import Minuit, describe
from probfit     import BinnedChi2, Extended

from invisible_cities.core.core_functions import in_range
from plotting_functions import plot_residuals_E_reso_gaussC
from control_plots      import labels, hist
from plotting_functions import gaussC

def select_dst_ring_r_z(dst, bins_r, bins_z, index_r, index_z):
    """
    Returns a ring/slice in r and z
    """    
    #print(bins_r[index_r], bins_z[index_z])

    print(f'Cut in R range    = {bins_r[index_r]}, {bins_r[index_r+1]} \
          and Cut in Z range = {bins_z[index_z]}, {bins_z[index_z+1]}')
    
    region = '${} < r < {}$\n${} < z < {}$'.format(bins_r[index_r], bins_r[index_r+1], bins_z[index_z], bins_z[index_z+1])

    sel_r = in_range(dst.R, bins_r[index_r], bins_r[index_r+1] )
    sel_z = in_range(dst.Z, bins_z[index_z], bins_z[index_z+1])
    sel   = sel_r & sel_z
    #print(sel)
    return dst[sel], region

def select_dst_disk_r_z(dst, bins_r, bins_z, index_r, index_z):
    """
    Returns a disk (inclusive) in r and z
    """  
    #print(bins_r[index_r], bins_z[index_z])

    print(f'Cut in R range    = {bins_r[0]}, {bins_r[index_r+1]} \
          and Cut in Z range = {bins_z[0]}, {bins_z[index_z+1]}')
    
    region = '${} < r < {}$\n${} < z < {}$'.format(bins_r[0], bins_r[index_r+1], bins_r[0], bins_z[index_z+1])

    sel_r = in_range(dst.R, bins_r[0], bins_r[index_r+1] )
    sel_z = in_range(dst.Z, bins_z[0], bins_z[index_z+1])
    sel   = sel_r & sel_z
    return dst[sel], region


def plot_fits(dst_inrange, corr, fit_erange, m, chi2_val, bins, region):
    
    
    mean     = m.values[0]
    mean_u   = m.errors[0]

    sigma    = m.values[1]
    sigma_u  = m.errors[1]

    N       = m.values[2]
    N_u     = m.errors[2]

    N2        = m.values[3]
    N2_u      = m.errors[3]
    
    chi2_1    = chi2_val
    chi2_2    = m.fval   # degrees of freedom

    print(f'Mean:  {mean:.2f}         +/- {mean_u:.2f} ')
    print(f'Sigma: {sigma:.2f}        +/- {sigma_u:.2f} ')
    print(f'N:     {N:.1f}            +/- {N_u:.1f} ')
    print(f'N2:    {N2:.1f}           +/- {N2_u:.1f} ')


    plt.style.use('classic')
    reso, fig = plot_residuals_E_reso_gaussC('', '', dst_inrange.S2e*corr, bins, fit_erange, mean, mean_u, sigma, sigma_u, N, N_u, N2, N2_u, chi2_1, chi2_2, region)

    return reso, fig


def plot_e_resolution_vs_z_r(reso_list, file_plot, ring):
    """
    Returns several plots in rings and/or disks in r and z
    """  
    
    bins_r = (0,30,35,40,45,50,55,60)
    bins_z = (0,70,120,170,220,270,320)
    
    same_R_30 =  reso_list [0:6]
    same_R_35 =  reso_list [6:12]
    same_R_40 =  reso_list [12:18]
    same_R_45 =  reso_list [18:24]
    same_R_50 =  reso_list [24:30]
    same_R_55 =  reso_list [30:36]
    same_R_60 =  reso_list [36:42]
    

    same_Z_70  = [reso_list[i] for i in [0,6,12,18,24,30,36]]
    same_Z_120 = [reso_list[i] for i in [1,7,13,19,25,31,37]]
    same_Z_170 = [reso_list[i] for i in [2,8,14,20,26,32,38]]
    same_Z_220 = [reso_list[i] for i in [3,9,15,21,27,33,39]]
    same_Z_270 = [reso_list[i] for i in [4,10,16,22,28,34,40]]
    same_Z_320 = [reso_list[i] for i in [5,11,17,23,29,35,41]]

    r = (30,35,40,45,50,55,60)
    z = (70,120,170,220,270,320)

    pp = PdfPages(file_plot)
    fig_1 = plt.figure(figsize=(9,7))

    #fig = plt.figure(figsize=(13,10))
    #ax      = fig.add_subplot(2, 2, 1)
    
    if ring == "yes":
        plt.plot(z, same_R_30, color='black', marker='o', markeredgecolor='white', linestyle='dotted', label='R [0,30]')
        plt.plot(z, same_R_35, color='blue', marker='o', markeredgecolor='white', linestyle='dotted', label='R [30,35]')
        plt.plot(z, same_R_40, color='crimson', marker='o', markeredgecolor='white', linestyle='dotted', label='R [35,40]')
        plt.plot(z, same_R_45, color='yellow', marker='o', markeredgecolor='white', linestyle='dotted', label='R [40,45]')
        plt.plot(z, same_R_50, color='brown', marker='o', markeredgecolor='white', linestyle='dotted', label='R [40,50]')
        plt.plot(z, same_R_55, color='pink', marker='o', markeredgecolor='white', linestyle='dotted', label='R [50,55]')
        plt.plot(z, same_R_60, color='royalblue', marker='o', markeredgecolor='white', linestyle='dotted', label='R [55,60]')
        
    if ring == "no":
        plt.plot(z, same_R_30, color='black', marker='o', markeredgecolor='white', linestyle='dotted', label='R [0,30]')
        plt.plot(z, same_R_35, color='blue', marker='o', markeredgecolor='white', linestyle='dotted', label='R [0,35]')
        plt.plot(z, same_R_40, color='crimson', marker='o', markeredgecolor='white', linestyle='dotted', label='R [0,40]')
        plt.plot(z, same_R_45, color='yellow', marker='o', markeredgecolor='white', linestyle='dotted', label='R [0,45]')
        plt.plot(z, same_R_50, color='brown', marker='o', markeredgecolor='white', linestyle='dotted', label='R [0,50]')
        plt.plot(z, same_R_55, color='pink', marker='o', markeredgecolor='white', linestyle='dotted', label='R [0,55]')
        plt.plot(z, same_R_60, color='royalblue', marker='o', markeredgecolor='white', linestyle='dotted', label='R [0,60]')


    labels('Z (mm)','Resolution FWHM (%)','')
    plt.legend(loc='upper right', ncol=3)
    plt.xlim(0,350)
    plt.ylim(3,6)
    pp.savefig(fig_1)

    fig_2 = plt.figure(figsize=(9,7))
    if ring == "yes":
        plt.plot(r, same_Z_70, color='blue', marker='o', markeredgecolor='white', linestyle='dotted', label='Z [0,70]')
        plt.plot(r, same_Z_120, color='gold', marker='o', markeredgecolor='white', linestyle='dotted', label='Z [70,120]')
        plt.plot(r, same_Z_170, color='orange', marker='o', markeredgecolor='white', linestyle='dotted', label='Z [120,170]')
        plt.plot(r, same_Z_220, color='lightgreen', marker='o', markeredgecolor='white', linestyle='dotted', label='Z [170,220]')
        plt.plot(r, same_Z_270, color='yellowgreen', marker='o', markeredgecolor='white', linestyle='dotted', label='R [220,270]')
        plt.plot(r, same_Z_320, color='olive', marker='o', markeredgecolor='white', linestyle='dotted', label='R [270,320]')
        
    if ring == "no":
        plt.plot(r, same_Z_70, color='blue', marker='o', markeredgecolor='white', linestyle='dotted', label='Z [0,70]')
        plt.plot(r, same_Z_120, color='gold', marker='o', markeredgecolor='white', linestyle='dotted', label='Z [0,120]')
        plt.plot(r, same_Z_170, color='orange', marker='o', markeredgecolor='white', linestyle='dotted', label='Z [0,170]')
        plt.plot(r, same_Z_220, color='lightgreen', marker='o', markeredgecolor='white', linestyle='dotted', label='Z [0,220]')
        plt.plot(r, same_Z_270, color='yellowgreen', marker='o', markeredgecolor='white', linestyle='dotted', label='R [0,270]')
        plt.plot(r, same_Z_320, color='olive', marker='o', markeredgecolor='white', linestyle='dotted', label='R [0,320]')

    labels('R (mm)','Resolution FWHM (%)','')
    plt.legend(loc='upper right', ncol=3)
    plt.xlim(20,75)
    plt.ylim(3,6)
    pp.savefig(fig_2)
    pp.close()

    print(f'-----> Plot of energy resolution vs z and r saved in {file_plot}\n')




def energy_reso_vs_z_r(dst, corr, file_fits, file_plot, opt_dict, ring = 'yes'):
    """
    Plots the resolution in several regions of the corrected kdst
    """

    pp = PdfPages(file_fits)

    bins_res    = int(opt_dict["bins_res"])
    
    fit_min_res = int(opt_dict["fit_min_res"])
    fit_max_res = int(opt_dict["fit_max_res"])

    bins_r = (0,30,35,40,45,50,55,60)
    bins_z = (0,70,120,170,220,270,320)
    
    dst_list = []
    reso_list = []
    num = 0
    
    print(f'-----> Start fits to dst selected by r and z ranges\n')

    for i in range(len(bins_r)-1):
        for j in range(len(bins_z)-1):
            if ring == 'yes':
                dst_inrange, region = select_dst_ring_r_z(dst, bins_r, bins_z, i, j)
            elif ring == 'no':
                dst_inrange, region = select_dst_disk_r_z(dst, bins_r, bins_z, i, j)
            print(f'Region id = {num}, i index = {i}, j index = {j}')
            dst_list.append(dst_inrange)
            chi2 = BinnedChi2(gaussC, dst_inrange.S2e*corr, bins = bins_res, bound = (fit_min_res, fit_max_res))
            chi2_1 = chi2.ndof
            print('Chi2.ndof: ', chi2_1)
            m = Minuit(chi2, mu = 10500, sigma = 150, N = 200, Ny = 10)
            m.migrad()
            reso, fig = plot_fits(dst_inrange, corr, (fit_min_res, fit_max_res), m, chi2_1, bins_res, region)
            reso_list.append(reso)
            pp.savefig(fig)
            num+=1
    print(f'-----> Fits saved in {file_fits}---->\n')
    pp.close()
    
    plot_e_resolution_vs_z_r(reso_list, file_plot, ring)
    
    return reso_list
