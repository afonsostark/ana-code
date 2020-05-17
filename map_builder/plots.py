import numpy  as np
import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.colors

from matplotlib.ticker                        import LogFormatter 
from mpl_toolkits.axes_grid.inset_locator     import inset_axes

from invisible_cities.core.fit_functions      import profileX

from krcal.core.kr_types                      import FitType
from krcal.core.selection_functions           import select_xy_sectors_df, event_map_df
from krcal.NB_utils.xy_maps_functions         import draw_xy_maps

from invisible_cities.core.core_functions     import shift_to_bin_centers

from iminuit                                  import Minuit, describe
from probfit                                  import Extended, BinnedChi2
from probfit                                  import gaussian, linear, poly2, Chi2Regression

from matplotlib.backends.backend_pdf          import PdfPages

from plotting_functions                       import gaussC, plot_residuals_E_reso_gaussC             
 
    
def plots_before_r_cut(dst):
    
    fig = plt.figure(figsize=(15,25))
    
    plt.subplot(5,2,1)
    a = plt.hist(dst.time, 500, color='indianred', histtype = 'stepfilled');
    plt.xlabel('Time of the run (s)')
    plt.ylabel('Entries')
    
    plt.tight_layout()
    
    plt.subplot(5,2,2)
    a = plt.hist2d(dst.X, dst.Y, 200, cmap = 'coolwarm');
    plt.xlabel('X (mm)')
    plt.ylabel('Y (mm)')
    cbar = plt.colorbar(a[3])
    cbar.set_label('Number of events')

    plt.axvline(x = -60, color = 'yellow', linewidth = 2);
    plt.axvline(x = 60, color = 'yellow', linewidth = 2);
    plt.axhline(y = -60, color = 'yellow', linewidth = 2);
    plt.axhline(y = 60, color = 'yellow', linewidth = 2);
    
    plt.tight_layout()
    
    plt.subplot(5,2,3)
    plt.hist(dst.X, 200, color = 'indianred', histtype = 'stepfilled');
    plt.xlabel('X (mm)'); 
    plt.ylabel('Entries'); 
    plt.axvline(x = -60, color = 'yellow', linewidth = 2);
    plt.axvline(x = 60, color = 'yellow', linewidth = 2);

    plt.tight_layout()

    plt.subplot(5,2,4)
    plt.hist(dst.Y, 200, color = 'indianred', histtype = 'stepfilled');
    plt.xlabel('Y (mm)'); 
    plt.ylabel('Entries'); 
    plt.axvline(x = -60, color = 'yellow', linewidth = 2);
    plt.axvline(x = 60, color = 'yellow', linewidth = 2);

    plt.tight_layout()

    plt.subplot(5,2,5)
    plt.hist(dst.Z, 200, (0,500), color = 'indianred', histtype = 'stepfilled');
    plt.xlabel('Z (mm)'); 
    plt.ylabel('Entries'); 

    plt.tight_layout()
    
    return fig


    
def plot_s2e_all_before(dst, opt_dict):
    
    s2e_clean_min  = int(opt_dict["s2e_clean_min"])
    s2e_clean_max  = int(opt_dict["s2e_clean_max"])
    
    s1e_clean_min  = int(opt_dict["s1e_clean_min"])
    s1e_clean_max  = int(opt_dict["s1e_clean_max"])

    s2w_clean_min  = int(opt_dict["s2w_clean_min"])
    s2w_clean_max  = int(opt_dict["s2w_clean_max"])

    s2q_clean_min  = int(opt_dict["s2q_clean_min"])
    s2q_clean_max  = int(opt_dict["s2q_clean_max"])

    nsipm_clean_min  = int(opt_dict["nsipm_clean_min"])
    nsipm_clean_max  = int(opt_dict["nsipm_clean_max"])
    
    zmap_clean_min  = int(opt_dict["zmap_clean_min"])
    zmap_clean_max  = int(opt_dict["zmap_clean_max"])
    
    xy_num_bins   = int(opt_dict['xy_num_bins'])
    nmin          = int(opt_dict['nmin'])
    
    fig_1 = plt.figure(figsize=(15,25))
    
    plt.subplot(5,2,1)
    plt.hist(dst.S2e, 200, (4000, 13000), color = 'indianred', label = 'All region', histtype = 'stepfilled');
    plt.xlabel('S2e (pes)')
    plt.ylabel('Entries')
    plt.legend()
    
    plt.tight_layout()
    
    plt.subplot(5,2,2)
    plt.hist(dst.S2e[(dst.Z > 20) & (dst.Z < 100)], 200, (4000, 13000), color = 'indianred', label = 'Z = [20, 100]', histtype = 'stepfilled');
    plt.xlabel('S2e (pes)')
    plt.ylabel('Entries')
    plt.legend()
    
    plt.tight_layout()

    plt.subplot(5,2,3)
    plt.hist(dst.S2e[(dst.Z > 20) & (dst.Z < 100) & (dst.R > 0) & (dst.R < 20)], 200, (4000, 13000), color = 'indianred', label = 'Z = [20, 100]\nR = [0, 20]', histtype = 'stepfilled');
    plt.xlabel('S2e (pes)')
    plt.ylabel('Entries')
    plt.legend()

    plt.tight_layout()
    
    plt.subplot(5,2,4)
    plt.hist(dst.S2e[(dst.Z > 20) & (dst.Z < 100) & (dst.R > 20) & (dst.R < 40)], 200, (4000, 13000), color = 'indianred', label = 'Z = [20, 100]\nR = [20, 40]', histtype = 'stepfilled');
    plt.xlabel('S2e (pes)')
    plt.ylabel('Entries')
    plt.legend()

    plt.tight_layout()

    plt.subplot(5,2,5)
    plt.hist(dst.S2e[(dst.Z > 20) & (dst.Z < 100) & (dst.R > 40) & (dst.R < 60)], 200, (4000, 13000), color = 'indianred', label = 'Z = [20, 100]\nR = [40, 60]', histtype = 'stepfilled');
    plt.xlabel('S2e (pes)')
    plt.ylabel('Entries')
    plt.legend()

    plt.tight_layout()
    
    plt.subplot(5,2,6)
    plt.hist(dst.S2e[(dst.Z > 100) & (dst.Z < 200)], 200, (4000, 13000), color = 'indianred', label = 'Z = [100, 200]', histtype = 'stepfilled');
    plt.xlabel('S2e (pes)')
    plt.ylabel('Entries')
    plt.legend()

    plt.tight_layout()
    
    plt.subplot(5,2,7)
    plt.hist(dst.S2e[(dst.Z > 100) & (dst.Z < 200) & (dst.R > 0) & (dst.R < 20)], 200, (4000, 13000), color = 'indianred', label = 'Z = [100, 200]\nR = [0, 20]', histtype = 'stepfilled');
    plt.xlabel('S2e (pes)')
    plt.ylabel('Entries')
    plt.legend()

    plt.tight_layout()
    
    plt.subplot(5,2,8)
    plt.hist(dst.S2e[(dst.Z > 100) & (dst.Z < 200) & (dst.R > 20) & (dst.R < 40)], 200, (4000, 13000), color = 'indianred', label = 'Z = [100, 200]\nR = [20, 40]', histtype = 'stepfilled');
    plt.xlabel('S2e (pes)')
    plt.ylabel('Entries')
    plt.legend()

    plt.tight_layout()
    
    plt.subplot(5,2,9)
    plt.hist(dst.S2e[(dst.Z > 100) & (dst.Z < 200) & (dst.R > 40) & (dst.R < 60)], 200, (4000, 13000), color = 'indianred', label = 'Z = [100, 200]\nR = [40, 60]', histtype = 'stepfilled');
    plt.xlabel('S2e (pes)')
    plt.ylabel('Entries')
    plt.legend()
    
    
    fig_2 = plt.figure(figsize=(15,25))

    plt.subplot(5,2,1)
    plt.hist(dst.S2e[(dst.Z > 200) & (dst.Z < 300)], 200, (4000, 13000), color = 'indianred', label = 'Z = [200, 300]', histtype = 'stepfilled');
    plt.xlabel('S2e (pes)')
    plt.ylabel('Entries')
    plt.legend()

    plt.tight_layout()

    plt.subplot(5,2,2)
    plt.hist(dst.S2e[(dst.Z > 200) & (dst.Z < 300) & (dst.R > 0) & (dst.R < 20)], 200, (4000, 13000), color = 'indianred', label = 'Z = [200, 300]\nR = [0, 20]', histtype = 'stepfilled');
    plt.xlabel('S2e (pes)')
    plt.ylabel('Entries')
    plt.legend()

    plt.tight_layout()
    
    plt.subplot(5,2,3)
    plt.hist(dst.S2e[(dst.Z > 200) & (dst.Z < 300) & (dst.R > 20) & (dst.R < 40)], 200, (4000, 13000), color = 'indianred', label = 'Z = [200, 300]\nR = [20, 40]', histtype = 'stepfilled');
    plt.xlabel('S2e (pes)')
    plt.ylabel('Entries')
    plt.legend()

    plt.tight_layout()

    plt.subplot(5,2,4)
    plt.hist(dst.S2e[(dst.Z > 200) & (dst.Z < 300) & (dst.R > 40) & (dst.R < 60)], 200, (4000, 13000), color = 'indianred', label = 'Z = [200, 300]\nR = [40, 60]', histtype = 'stepfilled');
    plt.xlabel('S2e (pes)')
    plt.ylabel('Entries')
    plt.legend()

    plt.tight_layout()
    
    plt.subplot(5,2,5)
    plt.hist(dst.S2e[(dst.Z > 100) & (dst.Z < 200) & (dst.R > 20) & (dst.R < 40)], 200, (8000, 12000), color = 'indianred', label = 'Z = [100, 200]\nR = [20, 40]', histtype = 'stepfilled');
    plt.xlabel('S2e (pes)')
    plt.ylabel('Entries')
    plt.legend()
    
    plt.axvline(x = s2e_clean_min, color = 'green')
    plt.axvline(x = s2e_clean_max, color = 'green')
    
    plt.tight_layout()
    
    plt.subplot(5,2,6)
    plt.hist(dst.S2e[(dst.Z > 100) & (dst.Z < 200) & (dst.R > 40) & (dst.R < 60)], 200, (8000, 12000), color = 'indianred', label = 'Z = [100, 200]\nR = [40, 60]', histtype = 'stepfilled');
    plt.xlabel('S2e (pes)')
    plt.ylabel('Entries')
    plt.legend()
    
    plt.axvline(x = s2e_clean_min, color = 'green')
    plt.axvline(x = s2e_clean_max, color = 'green')
    
    plt.tight_layout()
    

    fig_3 = plt.figure(figsize=(15,25))

    plt.subplot(5,2,1)
    a = plt.hist2d(dst.S2e, dst.S1e, 100, range = ((0, 12000), (0, 50)), cmap = 'coolwarm', norm = matplotlib.colors.LogNorm())
    plt.xlabel('S2e (pes)')
    plt.ylabel('S1e (pes)')
    plt.title('S1e = [{0}, {1}]'.format(s1e_clean_min, s1e_clean_max))
    cbar = plt.colorbar(a[3])
    cbar.ax.tick_params(labelsize=10)
    cbar.set_label('Number of events')
    
    plt.axvline(x = s2e_clean_min, color = 'red', linewidth = 2)
    plt.axvline(x = s2e_clean_max, color = 'red', linewidth = 2)
    
    plt.axhline(y = s1e_clean_min, color = 'yellow', linewidth = 2)
    plt.axhline(y = s1e_clean_max, color = 'yellow', linewidth = 2)
    
    plt.tight_layout()
    
    plt.subplot(5,2,2)
    a = plt.hist2d(dst.S2e, dst.S2w, 100, range = ((0, 12000), (0, 20)), cmap = 'coolwarm', norm = matplotlib.colors.LogNorm())
    plt.xlabel('S2e (pes)')
    plt.ylabel('S2w ($\mu$s)')
    plt.title('S2w = [{0}, {1}]'.format(s2w_clean_min, s2w_clean_max))
    cbar = plt.colorbar(a[3])    
    cbar.set_label('Number of events')
    
    plt.axvline(x = s2e_clean_min, color = 'red', linewidth = 2)
    plt.axvline(x = s2e_clean_max, color = 'red', linewidth = 2)
    
    plt.axhline(y = s2w_clean_min, color = 'yellow', linewidth = 2)
    plt.axhline(y = s2w_clean_max, color = 'yellow', linewidth = 2)
    
    plt.tight_layout()
    
    plt.subplot(5,2,3)
    a = plt.hist2d(dst.S2e, dst.S2q, 100, range = ((0, 12000), (0, 900)), cmap = 'coolwarm', norm = matplotlib.colors.LogNorm())
    plt.xlabel('S2e (pes)')
    plt.ylabel('S2q (pes)')
    plt.title('S2q = [{0}, {1}]'.format(s2q_clean_min, s2q_clean_max))
    cbar = plt.colorbar(a[3])    
    cbar.set_label('Number of events')
    
    plt.axvline(x = s2e_clean_min, color = 'red', linewidth = 2)
    plt.axvline(x = s2e_clean_max, color = 'red', linewidth = 2)
    
    plt.axhline(y = s2q_clean_min, color = 'yellow', linewidth = 2)
    plt.axhline(y = s2q_clean_max, color = 'yellow', linewidth = 2)
    
    plt.subplot(5,2,4)
    a = plt.hist2d(dst.S2e, dst.Nsipm, 40, range = ((0, 12000), (0, 40)), cmap = 'coolwarm', norm = matplotlib.colors.LogNorm())
    plt.xlabel('S2e (pes)')
    plt.ylabel('Nsipm')
    plt.title('Nsipm = [{0}, {1}]'.format(nsipm_clean_min, nsipm_clean_max))
    cbar = plt.colorbar(a[3])    
    cbar.set_label('Number of events')
    
    plt.axvline(x = s2e_clean_min, color = 'red', linewidth = 2)
    plt.axvline(x = s2e_clean_max, color = 'red', linewidth = 2)
    
    plt.axhline(y = nsipm_clean_min, color = 'yellow', linewidth = 2)
    plt.axhline(y = nsipm_clean_max, color = 'yellow', linewidth = 2)
    
    plt.tight_layout()
    
    return fig_1, fig_2, fig_3

def plot_after(dst, opt_dict):
    
    s2e_clean_min  = int(opt_dict["s2e_clean_min"])
    s2e_clean_max  = int(opt_dict["s2e_clean_max"])
    
    s1e_clean_min  = int(opt_dict["s1e_clean_min"])
    s1e_clean_max  = int(opt_dict["s1e_clean_max"])

    s2w_clean_min  = int(opt_dict["s2w_clean_min"])
    s2w_clean_max  = int(opt_dict["s2w_clean_max"])

    s2q_clean_min  = int(opt_dict["s2q_clean_min"])
    s2q_clean_max  = int(opt_dict["s2q_clean_max"])

    nsipm_clean_min  = int(opt_dict["nsipm_clean_min"])
    nsipm_clean_max  = int(opt_dict["nsipm_clean_max"])
    
    zmap_clean_min  = int(opt_dict["zmap_clean_min"])
    zmap_clean_max  = int(opt_dict["zmap_clean_max"])
    
    xy_num_bins   = int(opt_dict['xy_num_bins'])
    nmin          = int(opt_dict['nmin'])
    
    fig = plt.figure(figsize=(15,25))
    plt.subplot(5,2,1)
    plt.hist(dst.S2e[(dst.Z > 100) & (dst.Z < 200) & (dst.R > 20) & (dst.R < 40)], 200, (8000, 12000), color = 'indianred', label = 'Z = [100, 200]\nR = [20, 40]', histtype = 'stepfilled');
    plt.xlabel('S2e (pes)')
    plt.ylabel('Entries')
    plt.legend()
    
    plt.axvline(x = s2e_clean_min, color = 'green')
    plt.axvline(x = s2e_clean_max, color = 'green')
    
    plt.tight_layout()
    
    plt.subplot(5,2,2)    
    plt.hist(dst.S2e[(dst.Z > 100) & (dst.Z < 200) & (dst.R > 40) & (dst.R < 60)], 200, (8000, 12000), color = 'indianred', label = 'Z = [100, 200]\nR = [40, 60]', histtype = 'stepfilled');
    plt.xlabel('S2e (pes)')
    plt.ylabel('Entries')
    plt.legend()
    
    plt.axvline(x = s2e_clean_min, color = 'green')
    plt.axvline(x = s2e_clean_max, color = 'green')
    
    plt.tight_layout()
    
    x, y, yu  = profileX(dst.Z, dst.S2e, 30, (zmap_clean_min, zmap_clean_max), (s2e_clean_min, s2e_clean_max))
    
    plt.subplot(5,2,3)  
    a = plt.hist2d(dst.Z,dst.S2e, 80, range = ((0, 350), (4000, 12000)), cmap = 'coolwarm')
    plt.ylabel('S2e (pes)')
    plt.xlabel('Drift time ($\mu$s)')

    cbar = plt.colorbar(a[3])
    cbar.set_label('Number of events')

    plt.plot(x,y,yu)
    plt.errorbar(x, y, yu, fmt="kp")
    
    plt.tight_layout()
    
     
    XYbins = (xy_num_bins , xy_num_bins)
    xbins = np.linspace(*(-60,60), XYbins[0] + 1)
    ybins = np.linspace(*(-60,60), XYbins[1] + 1)

    KXY = select_xy_sectors_df(dst, xbins, ybins)
    nXY = event_map_df(KXY)
    
    plt.subplot(5,2,4) 
    plt.hist(nXY.values.flatten(), 100, (1,600), color = 'indianred', histtype = 'stepfilled')
    plt.xlabel('Number of events in each XY bin')
    plt.ylabel('Entries')

    plt.axvline(x = nmin, color = 'green')
    
    plt.tight_layout()

    plt.subplot(5,2,5) 
    sns.heatmap(nXY, vmin = 0, vmax = 150, cmap = 'coolwarm', cbar_kws = {'label': 'Number of events per XY bin'})

    plt.xlabel("X bins");
    plt.ylabel("Y bins");

    print('The total number of events is', np.sum(np.sum(nXY)), '\n')
    print('{:.2f} % of the events are within nXY > {} (nmin)\n'.format(100*np.sum(np.sum(nXY[nXY > nmin]))/np.sum(np.sum(nXY[nXY > 0])),nmin))
    
    return fig


def plot_after(dst, opt_dict):
    
    s2e_clean_min  = int(opt_dict["s2e_clean_min"])
    s2e_clean_max  = int(opt_dict["s2e_clean_max"])
    
    s1e_clean_min  = int(opt_dict["s1e_clean_min"])
    s1e_clean_max  = int(opt_dict["s1e_clean_max"])

    s2w_clean_min  = int(opt_dict["s2w_clean_min"])
    s2w_clean_max  = int(opt_dict["s2w_clean_max"])

    s2q_clean_min  = int(opt_dict["s2q_clean_min"])
    s2q_clean_max  = int(opt_dict["s2q_clean_max"])

    nsipm_clean_min  = int(opt_dict["nsipm_clean_min"])
    nsipm_clean_max  = int(opt_dict["nsipm_clean_max"])
    
    zmap_clean_min  = int(opt_dict["zmap_clean_min"])
    zmap_clean_max  = int(opt_dict["zmap_clean_max"])
    
    xy_num_bins   = int(opt_dict['xy_num_bins'])
    nmin          = int(opt_dict['nmin'])
    
    fig = plt.figure(figsize=(15,25))
    plt.subplot(5,2,1)
    plt.hist(dst.S2e[(dst.Z > 100) & (dst.Z < 200) & (dst.R > 20) & (dst.R < 40)], 200, (8000, 12000), color = 'indianred', label = 'Z = [100, 200]\nR = [20, 40]', histtype = 'stepfilled');
    plt.xlabel('S2e (pes)')
    plt.ylabel('Entries')
    plt.legend()
    
    plt.axvline(x = s2e_clean_min, color = 'green')
    plt.axvline(x = s2e_clean_max, color = 'green')
    
    plt.tight_layout()
    
    plt.subplot(5,2,2)    
    plt.hist(dst.S2e[(dst.Z > 100) & (dst.Z < 200) & (dst.R > 40) & (dst.R < 60)], 200, (8000, 12000), color = 'indianred', label = 'Z = [100, 200]\nR = [40, 60]', histtype = 'stepfilled');
    plt.xlabel('S2e (pes)')
    plt.ylabel('Entries')
    plt.legend()
    
    plt.axvline(x = s2e_clean_min, color = 'green')
    plt.axvline(x = s2e_clean_max, color = 'green')
    
    plt.tight_layout()
    
    x, y, yu  = profileX(dst.Z, dst.S2e, 30, (zmap_clean_min, zmap_clean_max), (s2e_clean_min, s2e_clean_max))
    
    plt.subplot(5,2,3)  
    a = plt.hist2d(dst.Z,dst.S2e, 80, range = ((0, 350), (4000, 12000)), cmap = 'coolwarm')
    plt.ylabel('S2e (pes)')
    plt.xlabel('Drift time ($\mu$s)')

    cbar = plt.colorbar(a[3])
    cbar.set_label('Number of events')

    plt.plot(x,y,yu)
    plt.errorbar(x, y, yu, fmt="kp")
    
    plt.tight_layout()
    
     
    XYbins = (xy_num_bins , xy_num_bins)
    xbins = np.linspace(*(-60,60), XYbins[0] + 1)
    ybins = np.linspace(*(-60,60), XYbins[1] + 1)

    KXY = select_xy_sectors_df(dst, xbins, ybins)
    nXY = event_map_df(KXY)
    
    plt.subplot(5,2,4) 
    plt.hist(nXY.values.flatten(), 100, (1,600), color = 'indianred', histtype = 'stepfilled')
    plt.xlabel('Number of events in each XY bin')
    plt.ylabel('Entries')

    plt.axvline(x = nmin, color = 'green')
    
    plt.tight_layout()

    plt.subplot(5,2,5) 
    sns.heatmap(nXY, vmin = 0, vmax = 600, cmap = 'coolwarm', cbar_kws = {'label': 'Number of events per XY bin'})

    plt.xlabel("X bins");
    plt.ylabel("Y bins");

    print('The total number of events is', np.sum(np.sum(nXY)), '\n')
    print('{:.2f} % of the events are within nXY > {} (nmin)\n'.format(100*np.sum(np.sum(nXY[nXY > nmin]))/np.sum(np.sum(nXY[nXY > 0])),nmin))
    
    return fig

    
       
def map_drawing(maps):
       
    fig = plt.figure(figsize = (14,10))
    
    plt.subplot(2,2,1)
    sns.heatmap(maps.e0.fillna(0), vmin = 8500, vmax = 10500, cmap = 'coolwarm', square = True)

    plt.subplot(2,2,2)
    sns.heatmap(maps.e0u.fillna(0), vmin = 0, vmax = 1.5, cmap = 'coolwarm', square = True)

    plt.subplot(2,2,3)
    sns.heatmap(maps.lt.fillna(0), vmin = 0, vmax = 80000, cmap = 'coolwarm', square = True)

    plt.subplot(2,2,4)
    sns.heatmap(maps.ltu.fillna(0), vmin = 0, vmax = 100, cmap = 'coolwarm', square = True)
    
    plt.tight_layout()

    return fig


def control_plots(maps):
    
    fig = plt.figure(figsize=(15,25))
    fig.tight_layout()
    plt.subplot(5,2,1)
    plt.hist(maps.chi2.values.flatten(), 80, (0, 10), color = 'indianred', histtype='stepfilled', label = 'Regularized map')
    plt.tight_layout()
    plt.xlabel('Chi2')
    plt.ylabel('Entries')
    plt.yscale('log')
    plt.legend()

  
    plt.subplot(5,2,2)
    plt.hist(maps.lt.values.flatten(), 80, color = 'indianred', range = (-200000, 200000), histtype='stepfilled', label = 'Regularized map')
    plt.tight_layout()
    plt.xlabel('Lifetime ($\mu$s) ')
    plt.ylabel('Entries')
    plt.yscale('log')
    plt.legend()

    
    plt.subplot(5,2,3)
    plt.hist(maps.lt.values.flatten()*maps.ltu.values.flatten()/100, 80, color = 'indianred', histtype='stepfilled', range = (0, 200000), label = 'Regularized map')
    plt.tight_layout()

    plt.xlabel('Lifetime uncertainty ($\mu$s) ')
    plt.ylabel('Entries')
    plt.yscale('log')
    plt.legend()

    
    plt.subplot(5,2,4)
    plt.hist(maps.e0.values.flatten(), 80, color = 'indianred', histtype='stepfilled', label = 'Regularized map')
    plt.tight_layout()

    plt.xlabel('e0 (pes)')
    plt.ylabel('Entries')
    plt.yscale('log')
    plt.legend()

    
    plt.subplot(5,2,5)
    plt.hist(maps.e0.values.flatten()*maps.e0u.values.flatten()/100, 80, range = (0,200), histtype='stepfilled', color = 'indianred', label = 'Regularized map')
    plt.tight_layout()

    plt.xlabel('e0 uncertainty (pes)')
    plt.ylabel('Entries')
    plt.yscale('log')
    plt.legend()


    plt.subplot(5,2,6)
    plt.hist(100/maps.ltu.values.flatten(), 80, range = (-5, 15), color = 'indianred', histtype='stepfilled', label = 'Regularized map')
    plt.tight_layout()

    plt.xlabel('lt / ltu')
    plt.ylabel('Entries')
    plt.yscale('log')
    plt.legend()

    
    plt.subplot(5,2,7)
    sns.heatmap(100/maps.ltu, vmin = -4, vmax = 4, cmap = 'coolwarm', cbar_kws = {'label': 'lt / ltu'})
    plt.tight_layout()

    plt.xlabel("X bins");
    plt.ylabel("Y bins");

    
    plt.subplot(5,2,8)
    a = plt.hist2d(100/maps.ltu.values.flatten(), maps.lt.values.flatten(), 80, range = ((-3, 3), (-200000, 200000)), cmap = 'coolwarm', norm = matplotlib.colors.LogNorm())

    plt.xlabel('lt / ltu regularized')
    plt.ylabel('lt regularized ($\mu$s)')
    cbar = plt.colorbar(a[3])
    cbar.set_label('Number of events')
    plt.tight_layout()

    
    plt.subplot(5,2,9)
    a = plt.hist2d(maps.e0.values.flatten(), maps.lt.values.flatten(), 80, range = ((8000, 11000), (-100000, 100000)), cmap = 'coolwarm', norm = matplotlib.colors.LogNorm())

    plt.xlabel('e0 regularized (pes)')
    plt.ylabel('lt regularized ($\mu$s)')
    cbar = plt.colorbar(a[3])
    cbar.ax.tick_params(labelsize=10)
    cbar.set_label('Number of events')

    plt.tight_layout()
    
    return fig
    
    
def time_evolution_plots(maps):
    
    fig = plt.figure(figsize=(15,25))
    fig.tight_layout()
    plt.subplot(5,2,1)
    
    plt.errorbar(maps.t_evol.values[:,0], maps.t_evol.values[:,1], maps.t_evol.values[:,2], fmt = '.k', ecolor = 'black', marker = 'o', markeredgecolor = 'black', markerfacecolor = 'black', label = "Data ({0} points)".format(len(maps.t_evol.values[:,0])))
    
    plt.ylabel('e0 (pes)'); 
    plt.xlabel('Time of the run (s)')
    plt.axhline(y=np.mean(maps.t_evol.values[:,1]),color='b',linestyle = 'dashed',label = "Mean = {:.2f} pes"
                .format(np.mean(np.mean(maps.t_evol.values[:,1]))))
    plt.axhline(y=np.mean(maps.t_evol.values[:,1])*1.005,color='r',linestyle = 'dashed',label = "+0.5 % of mean ({:.2f} pes)"
                .format(np.mean(maps.t_evol.values[:,1])*1.005))
    plt.axhline(y=np.mean(maps.t_evol.values[:,1])*0.995,color='r',linestyle = 'dashed',label = "-0.5 % of mean ({:.2f} pes)"
                .format(np.mean(maps.t_evol.values[:,1])*0.995))
    
    plt.legend()
      

    plt.ylim(10000,10400)

    plt.xlim(maps.t_evol.values[0,0],maps.t_evol.values[-1,0])
    
    plt.tight_layout()
    
    plt.subplot(5,2,2)
    plt.scatter(maps.t_evol.values[:,0],maps.t_evol.values[:,3],marker = 'o',color='black',label = "Data ({0} points)".format(len(maps.t_evol.values[:,0])))
    plt.ylabel('Lifetime ($\mu$s)'); 
    plt.ylim(0,200000)
    plt.xlabel('Time of the run (s)')

    plt.axhline(y=np.mean(maps.t_evol.values[:,3]),color='b',linestyle = 'dashed',label = "Mean = {:.2f} $\mu$s"
                .format(np.mean(np.mean(maps.t_evol.values[:,3]))))
    
    plt.legend()

    plt.xlim(maps.t_evol.values[0,0],maps.t_evol.values[-1,0])
    
    plt.tight_layout()
    
    plt.subplot(5,2,3)
    plt.errorbar(maps.t_evol.values[:,0],maps.t_evol.values[:,5],maps.t_evol.values[:,6],fmt='.k',ecolor='black',marker = 'o',markeredgecolor='black',
                 markerfacecolor='black',label = "Data ({0} points)".format(len(maps.t_evol.values[:,0])))
    plt.ylabel('Drift velocity (mm/$\mu$s)'); 
    plt.ylim(0.86,0.92)
    plt.xlabel('Time of the run (s)')
 
    plt.axhline(y=np.mean(maps.t_evol.values[:,5]),color='b',linestyle = 'dashed',label = "Mean = {:.4f} mm/$\mu$s"
                .format(np.mean(np.mean(maps.t_evol.values[:,5]))))
    plt.axhline(y=np.mean(maps.t_evol.values[:,5])*1.005,color='r',linestyle = 'dashed',label = "+0.5 % of mean ({:.4f} mm/$\mu$s)"
                .format(np.mean(maps.t_evol.values[:,5])*1.005))
    plt.axhline(y=np.mean(maps.t_evol.values[:,5])*0.995,color='r',linestyle = 'dashed',label = "-0.5 % of mean ({:.4f} mm/$\mu$s)"
                .format(np.mean(maps.t_evol.values[:,5])*0.995))
    
    plt.legend()

    plt.xlim(maps.t_evol.values[0,0],maps.t_evol.values[-1,0])
    
    plt.tight_layout()

    plt.subplot(5,2,4)
    plt.errorbar(maps.t_evol.values[:,0],maps.t_evol.values[:,7],maps.t_evol.values[:,8],fmt='.k',ecolor='black',marker = 'o',markeredgecolor='black',
                 markerfacecolor='black',label = "Data ({0} points)".format(len(maps.t_evol.values[:,0])))
    plt.ylabel('Energy resolution FWHM (%)'); 
    plt.ylim(3,6)
    plt.xlabel('Time of the run (s)')

    plt.axhline(y=np.mean(maps.t_evol.values[:,7]),color='b',linestyle = 'dashed',label = "Mean = {:.2f} %"
                .format(np.mean(np.mean(maps.t_evol.values[:,7]))))
    plt.axhline(y=np.mean(maps.t_evol.values[:,7])*1.1,color='r',linestyle = 'dashed',label = "+10 % of mean ({:.2f} %)"
                .format(np.mean(maps.t_evol.values[:,7])*1.1))
    plt.axhline(y=np.mean(maps.t_evol.values[:,7])*0.9,color='r',linestyle = 'dashed',label = "-10 % of mean ({:.2f} %)"
                .format(np.mean(maps.t_evol.values[:,7])*0.9))
    
    plt.legend()

    plt.xlim(maps.t_evol.values[0,0],maps.t_evol.values[-1,0])
    
    plt.tight_layout()
 
    plt.subplot(5,2,5)
    plt.errorbar(maps.t_evol.values[:,0],maps.t_evol.values[:,13],maps.t_evol.values[:,14],fmt='.k',ecolor='black',marker = 'o',markeredgecolor='black',
                 markerfacecolor='black',label = "Data ({0} points)".format(len(maps.t_evol.values[:,0])))
    plt.ylabel('S1e (pes)'); 
    plt.ylim(7,14)
    plt.xlabel('Time of the run (s)')
 
    plt.axhline(y=np.mean(maps.t_evol.values[:,13]),color='b',linestyle = 'dashed',label = "Mean = {:.2f} pes"
                .format(np.mean(np.mean(maps.t_evol.values[:,13]))))
    plt.axhline(y=np.mean(maps.t_evol.values[:,13])*1.05,color='r',linestyle = 'dashed',label = "+5 % of mean ({:.2f} pes)"
                .format(np.mean(maps.t_evol.values[:,13])*1.05))
    plt.axhline(y=np.mean(maps.t_evol.values[:,13])*0.95,color='r',linestyle = 'dashed',label = "-5 % of mean ({:.2f} pes)"
                .format(np.mean(maps.t_evol.values[:,13])*0.95))
    
    plt.legend()

    plt.xlim(maps.t_evol.values[0,0],maps.t_evol.values[-1,0])
    
    plt.tight_layout()
    
    plt.subplot(5,2,6)
    plt.errorbar(maps.t_evol.values[:,0],maps.t_evol.values[:,19],maps.t_evol.values[:,20],fmt='.k',ecolor='black',marker = 'o',markeredgecolor='black',
                 markerfacecolor='black',label = "Data ({0} points)".format(len(maps.t_evol.values[:,0])))
    plt.ylabel('S2e (pes)'); 
    plt.ylim(10000,10300)
    plt.xlabel('Time of the run (s)')

    plt.axhline(y=np.mean(maps.t_evol.values[:,19]),color='b',linestyle = 'dashed',label = "Mean = {:.2f} pes"
                .format(np.mean(np.mean(maps.t_evol.values[:,19]))))
    plt.axhline(y=np.mean(maps.t_evol.values[:,19])*1.005,color='r',linestyle = 'dashed',label = "+0.5 % of mean ({:.2f} pes)"
                .format(np.mean(maps.t_evol.values[:,19])*1.005))
    plt.axhline(y=np.mean(maps.t_evol.values[:,19])*0.995,color='r',linestyle = 'dashed',label = "-0.5 % of mean ({:.2f} pes)"
                .format(np.mean(maps.t_evol.values[:,19])*0.995))
    
    plt.legend()

    plt.xlim(maps.t_evol.values[0,0],maps.t_evol.values[-1,0])
    
    plt.tight_layout()

    
    return fig
    
    
def plot_dst_corrected_with_map(dst, corr_tot, opt_dict):
    
    zmap_clean_min = int(opt_dict['zmap_clean_min'])
    zmap_clean_max = int(opt_dict['zmap_clean_max'])
    
    s2e_clean_min  = int(opt_dict['s2e_clean_min'])
    s2e_clean_max  = int(opt_dict['s2e_clean_max'])
    
    x, y, yu  = profileX(dst.Z, dst.S2e, 30, (zmap_clean_min, zmap_clean_max),(s2e_clean_min, s2e_clean_max))
    
    fig = plt.figure(figsize=(15,25))
    
    plt.subplot(5,2,1)
    a = plt.hist2d(dst.Z, dst.S2e, 80, range = ((0, 350), (7000, 12000)), cmap = 'coolwarm')
    plt.ylabel('S2e (pes)')
    plt.xlabel('Drift time ($\mu$s)')
    plt.title('Raw energy')

    cbar = plt.colorbar(a[3])
    cbar.set_label('Number of events')

    plt.plot(x, y, yu)
    plt.errorbar(x, y, yu, fmt = "kp")

    plt.subplot(5,2,2)
    a = plt.hist2d(dst.Z, dst.S2e*corr_tot, 80, range = ((0, 350), (7000, 12000)), cmap = 'coolwarm')
    plt.ylabel('S2e (pes)')
    plt.xlabel('Drift time ($\mu$s)')
    plt.title('Corrected energy')

    cbar = plt.colorbar(a[3])
    cbar.set_label('Number of events')

    plt.plot(x, y, yu)
    plt.errorbar(x, y, yu, fmt = "kp")

    plt.subplot(5,2,5)
    a = plt.hist2d(dst.R, dst.S2e, 80, range= ((0, 60), (s2e_clean_min, s2e_clean_max)), cmap = 'coolwarm');

    plt.ylabel('S2e (pes)')
    plt.xlabel('R (mm)')
    plt.title('Raw energy')

    cbar = plt.colorbar(a[3])
    cbar.set_label('Number of events')
    
    plt.subplot(5,2,6)
    a = plt.hist2d(dst.R, dst.S2e*corr_tot, 80, range = ((0, 60), (8600, 12000)), cmap = 'coolwarm');

    plt.ylabel('S2e (pes)')
    plt.xlabel('R (mm)')
    plt.title('Corrected energy')

    cbar = plt.colorbar(a[3])
    cbar.set_label('Number of events')
    
    return fig
    

def control_plots_before_maps(dst1, dst2, dst3, plots_dir, opt_dict):
    
    fig_1 = plots_before_r_cut(dst1)
    fig_2, fig_3, fig_4 = plot_s2e_all_before(dst2, opt_dict)
    fig_5 = plot_after(dst3, opt_dict)
    
    
    pp = PdfPages(plots_dir + '/control_plots_before_map.pdf')

    pp.savefig(fig_1)
    pp.savefig(fig_2)
    pp.savefig(fig_3)
    pp.savefig(fig_4)
    pp.savefig(fig_5)
    
    pp.close()
    
    
    
def control_plots_after_map(dst, maps, corr_tot, plots_dir, opt_dict):
    
    fig_1 = map_drawing(maps)
    fig_2 = control_plots(maps)
    fig_3 = time_evolution_plots(maps)
    fig_4 = plot_dst_corrected_with_map(dst, corr_tot, opt_dict)
    
    pp = PdfPages(plots_dir + '/control_plots_after_map.pdf')

    pp.savefig(fig_1)
    pp.savefig(fig_2)
    pp.savefig(fig_3)
    pp.savefig(fig_4)
    
    pp.close()
    

    
def worse_best_res(reso_list_1_ring, reso_list_1_disk, reso_list_2_ring, reso_list_2_disk, reso_list_4_ring, reso_list_4_disk, reso_list_4_4_ring, reso_list_4_4_disk):
    
    r = [30,35,40,45,50,55,60]
    z = [70,120,170,220,270,320]
    
    same_Z_70_ring_1   = [reso_list_1_ring[i] for i in [0,6,12,18,24,30,36]]
    same_Z_120_ring_1  = [reso_list_1_ring[i] for i in [1,7,13,19,25,31,37]]
    same_Z_170_ring_1  = [reso_list_1_ring[i] for i in [2,8,14,20,26,32,38]]
    same_Z_220_ring_1  = [reso_list_1_ring[i] for i in [3,9,15,21,27,33,39]]
    same_Z_270_ring_1  = [reso_list_1_ring[i] for i in [4,10,16,22,28,34,40]]
    same_Z_320_ring_1  = [reso_list_1_ring[i] for i in [5,11,17,23,29,35,41]]
    
    same_Z_70_disk_1   = [reso_list_1_disk[i] for i in [0,6,12,18,24,30,36]]
    same_Z_120_disk_1  = [reso_list_1_disk[i] for i in [1,7,13,19,25,31,37]]
    same_Z_170_disk_1  = [reso_list_1_disk[i] for i in [2,8,14,20,26,32,38]]
    same_Z_220_disk_1  = [reso_list_1_disk[i] for i in [3,9,15,21,27,33,39]]
    same_Z_270_disk_1  = [reso_list_1_disk[i] for i in [4,10,16,22,28,34,40]]
    same_Z_320_disk_1  = [reso_list_1_disk[i] for i in [5,11,17,23,29,35,41]]
    
    
    
    same_Z_70_ring_2   = [reso_list_2_ring[i] for i in [0,6,12,18,24,30,36]]
    same_Z_120_ring_2  = [reso_list_2_ring[i] for i in [1,7,13,19,25,31,37]]
    same_Z_170_ring_2  = [reso_list_2_ring[i] for i in [2,8,14,20,26,32,38]]
    same_Z_220_ring_2  = [reso_list_2_ring[i] for i in [3,9,15,21,27,33,39]]
    same_Z_270_ring_2  = [reso_list_2_ring[i] for i in [4,10,16,22,28,34,40]]
    same_Z_320_ring_2  = [reso_list_2_ring[i] for i in [5,11,17,23,29,35,41]]
    
    same_Z_70_disk_2   = [reso_list_2_disk[i] for i in [0,6,12,18,24,30,36]]
    same_Z_120_disk_2  = [reso_list_2_disk[i] for i in [1,7,13,19,25,31,37]]
    same_Z_170_disk_2  = [reso_list_2_disk[i] for i in [2,8,14,20,26,32,38]]
    same_Z_220_disk_2  = [reso_list_2_disk[i] for i in [3,9,15,21,27,33,39]]
    same_Z_270_disk_2  = [reso_list_2_disk[i] for i in [4,10,16,22,28,34,40]]
    same_Z_320_disk_2  = [reso_list_2_disk[i] for i in [5,11,17,23,29,35,41]]
    
    
    
    same_Z_70_ring_4   = [reso_list_4_ring[i] for i in [0,6,12,18,24,30,36]]
    same_Z_120_ring_4  = [reso_list_4_ring[i] for i in [1,7,13,19,25,31,37]]
    same_Z_170_ring_4  = [reso_list_4_ring[i] for i in [2,8,14,20,26,32,38]]
    same_Z_220_ring_4  = [reso_list_4_ring[i] for i in [3,9,15,21,27,33,39]]
    same_Z_270_ring_4  = [reso_list_4_ring[i] for i in [4,10,16,22,28,34,40]]
    same_Z_320_ring_4  = [reso_list_4_ring[i] for i in [5,11,17,23,29,35,41]]
    
    same_Z_70_disk_4   = [reso_list_4_disk[i] for i in [0,6,12,18,24,30,36]]
    same_Z_120_disk_4  = [reso_list_4_disk[i] for i in [1,7,13,19,25,31,37]]
    same_Z_170_disk_4  = [reso_list_4_disk[i] for i in [2,8,14,20,26,32,38]]
    same_Z_220_disk_4  = [reso_list_4_disk[i] for i in [3,9,15,21,27,33,39]]
    same_Z_270_disk_4  = [reso_list_4_disk[i] for i in [4,10,16,22,28,34,40]]
    same_Z_320_disk_4  = [reso_list_4_disk[i] for i in [5,11,17,23,29,35,41]]
    
    
    
    
    same_Z_70_ring_4_4   = [reso_list_4_4_ring[i] for i in [0,6,12,18,24,30,36]]
    same_Z_120_ring_4_4  = [reso_list_4_4_ring[i] for i in [1,7,13,19,25,31,37]]
    same_Z_170_ring_4_4  = [reso_list_4_4_ring[i] for i in [2,8,14,20,26,32,38]]
    same_Z_220_ring_4_4  = [reso_list_4_4_ring[i] for i in [3,9,15,21,27,33,39]]
    same_Z_270_ring_4_4  = [reso_list_4_4_ring[i] for i in [4,10,16,22,28,34,40]]
    same_Z_320_ring_4_4  = [reso_list_4_4_ring[i] for i in [5,11,17,23,29,35,41]]
    
    same_Z_70_disk_4_4   = [reso_list_4_4_disk[i] for i in [0,6,12,18,24,30,36]]
    same_Z_120_disk_4_4  = [reso_list_4_4_disk[i] for i in [1,7,13,19,25,31,37]]
    same_Z_170_disk_4_4  = [reso_list_4_4_disk[i] for i in [2,8,14,20,26,32,38]]
    same_Z_220_disk_4_4  = [reso_list_4_4_disk[i] for i in [3,9,15,21,27,33,39]]
    same_Z_270_disk_4_4  = [reso_list_4_4_disk[i] for i in [4,10,16,22,28,34,40]]
    same_Z_320_disk_4_4  = [reso_list_4_4_disk[i] for i in [5,11,17,23,29,35,41]]
    
    
    
    
    same_R_30_ring_1   = reso_list_1_ring[0:6]
    same_R_35_ring_1   = reso_list_1_ring[6:12]
    same_R_40_ring_1   = reso_list_1_ring[12:18]
    same_R_45_ring_1   = reso_list_1_ring[18:24]
    same_R_50_ring_1   = reso_list_1_ring[24:30]
    same_R_55_ring_1   = reso_list_1_ring[30:36]
    same_R_60_ring_1   = reso_list_1_ring[36:42]
    
    same_R_30_disk_1   = reso_list_1_disk[0:6]
    same_R_35_disk_1   = reso_list_1_disk[6:12]
    same_R_40_disk_1   = reso_list_1_disk[12:18]
    same_R_45_disk_1   = reso_list_1_disk[18:24]
    same_R_50_disk_1   = reso_list_1_disk[24:30]
    same_R_55_disk_1   = reso_list_1_disk[30:36]
    same_R_60_disk_1   = reso_list_1_disk[36:42]
    
    
    
    same_R_30_ring_2   = reso_list_2_ring[0:6]
    same_R_35_ring_2   = reso_list_2_ring[6:12]
    same_R_40_ring_2   = reso_list_2_ring[12:18]
    same_R_45_ring_2   = reso_list_2_ring[18:24]
    same_R_50_ring_2   = reso_list_2_ring[24:30]
    same_R_55_ring_2   = reso_list_2_ring[30:36]
    same_R_60_ring_2   = reso_list_2_ring[36:42]
    
    same_R_30_disk_2   = reso_list_2_disk[0:6]
    same_R_35_disk_2   = reso_list_2_disk[6:12]
    same_R_40_disk_2   = reso_list_2_disk[12:18]
    same_R_45_disk_2   = reso_list_2_disk[18:24]
    same_R_50_disk_2   = reso_list_2_disk[24:30]
    same_R_55_disk_2   = reso_list_2_disk[30:36]
    same_R_60_disk_2   = reso_list_2_disk[36:42]
    
    
    
    same_R_30_ring_4   = reso_list_4_ring[0:6]
    same_R_35_ring_4   = reso_list_4_ring[6:12]
    same_R_40_ring_4   = reso_list_4_ring[12:18]
    same_R_45_ring_4   = reso_list_4_ring[18:24]
    same_R_50_ring_4   = reso_list_4_ring[24:30]
    same_R_55_ring_4   = reso_list_4_ring[30:36]
    same_R_60_ring_4   = reso_list_4_ring[36:42]
    
    
    same_R_30_disk_4   = reso_list_4_disk[0:6]
    same_R_35_disk_4   = reso_list_4_disk[6:12]
    same_R_40_disk_4   = reso_list_4_disk[12:18]
    same_R_45_disk_4   = reso_list_4_disk[18:24]
    same_R_50_disk_4   = reso_list_4_disk[24:30]
    same_R_55_disk_4   = reso_list_4_disk[30:36]
    same_R_60_disk_4   = reso_list_4_disk[36:42]
    
    
    
    
    same_R_30_ring_4_4   = reso_list_4_4_ring[0:6]
    same_R_35_ring_4_4   = reso_list_4_4_ring[6:12]
    same_R_40_ring_4_4   = reso_list_4_4_ring[12:18]
    same_R_45_ring_4_4   = reso_list_4_4_ring[18:24]
    same_R_50_ring_4_4   = reso_list_4_4_ring[24:30]
    same_R_55_ring_4_4   = reso_list_4_4_ring[30:36]
    same_R_60_ring_4_4   = reso_list_4_4_ring[36:42]
    
    
    same_R_30_disk_4_4   = reso_list_4_4_disk[0:6]
    same_R_35_disk_4_4   = reso_list_4_4_disk[6:12]
    same_R_40_disk_4_4   = reso_list_4_4_disk[12:18]
    same_R_45_disk_4_4   = reso_list_4_4_disk[18:24]
    same_R_50_disk_4_4   = reso_list_4_4_disk[24:30]
    same_R_55_disk_4_4   = reso_list_4_4_disk[30:36]
    same_R_60_disk_4_4   = reso_list_4_4_disk[36:42]
    
    
    
    plt.figure()
    
    plt.title('7949 corrected with 1 run')
    plt.plot(r, same_Z_70_ring_1, color='blue', marker='o', markeredgecolor='white', linestyle='dotted', label='Ring, Z [0,70]')
    
    plt.plot(r, same_Z_120_ring_1, color='orange', marker='o', markeredgecolor='white', linestyle='dotted', label='Ring, Z [70,120]')
    
    plt.plot(r, same_Z_170_ring_1, color='olive', marker='o', markeredgecolor='white', linestyle='dotted', label='Ring, Z [120,170]')
    
    plt.plot(r, same_Z_220_ring_1, color='red', marker='o', markeredgecolor='white', linestyle='dotted', label='Ring, Z [170,220]')
    
    plt.plot(r, same_Z_270_ring_1, color='brown', marker='o', markeredgecolor='white', linestyle='dotted', label='Ring, Z [220,270]')
        
    plt.plot(r, same_Z_320_ring_1, color='crimson', marker='o', markeredgecolor='white', linestyle='dotted', label='Ring, Z [270,320]')
    
    plt.plot(r, same_Z_70_disk_1, color='pink', marker='o', markeredgecolor='white', linestyle='dotted', label='Disk, Z [0,70]')
    
    plt.plot(r, same_Z_120_disk_1, color='cyan', marker='o', markeredgecolor='white', linestyle='dotted', label='Disk, Z [0,120]')
    
    plt.plot(r, same_Z_170_disk_1, color='gray', marker='o', markeredgecolor='white', linestyle='dotted', label='Disk, Z [0,170]')
    
    plt.plot(r, same_Z_220_disk_1, color='green', marker='o', markeredgecolor='white', linestyle='dotted', label='Disk, Z [0,220]')
    
    plt.plot(r, same_Z_270_disk_1, color='purple', marker='o', markeredgecolor='white', linestyle='dotted', label='Disk, Z [0,270]')
        
    plt.plot(r, same_Z_320_disk_1, color='goldenrod', marker='o', markeredgecolor='white', linestyle='dotted', label='Disk, Z [0,320]')
    
    
    plt.xlabel('R (mm)'),
    plt.ylabel('Resolution FWHM (%)')
    plt.legend(loc='upper right', ncol=4, fontsize=6)
    plt.xlim(20,75)
    plt.ylim(3,6)
    
    plt.savefig('/home/afonso/data/results/plot_1_r_all_all.png', dpi = 300)
 
    # plt.show()
    
    plt.close()
    
    
    
    plt.figure()
    
    plt.title('7949 corrected with 2 runs')
    plt.plot(r, same_Z_70_ring_2, color='blue', marker='o', markeredgecolor='white', linestyle='dotted', label='Ring, Z [0,70]')
    
    plt.plot(r, same_Z_120_ring_2, color='orange', marker='o', markeredgecolor='white', linestyle='dotted', label='Ring, Z [70,120]')
    
    plt.plot(r, same_Z_170_ring_2, color='olive', marker='o', markeredgecolor='white', linestyle='dotted', label='Ring, Z [120,170]')
    
    plt.plot(r, same_Z_220_ring_2, color='red', marker='o', markeredgecolor='white', linestyle='dotted', label='Ring, Z [170,220]')
    
    plt.plot(r, same_Z_270_ring_2, color='brown', marker='o', markeredgecolor='white', linestyle='dotted', label='Ring, Z [220,270]')
        
    plt.plot(r, same_Z_320_ring_2, color='crimson', marker='o', markeredgecolor='white', linestyle='dotted', label='Ring, Z [270,320]')
    
    plt.plot(r, same_Z_70_disk_2, color='pink', marker='o', markeredgecolor='white', linestyle='dotted', label='Disk, Z [0,70]')
    
    plt.plot(r, same_Z_120_disk_2, color='cyan', marker='o', markeredgecolor='white', linestyle='dotted', label='Disk, Z [0,120]')
    
    plt.plot(r, same_Z_170_disk_2, color='gray', marker='o', markeredgecolor='white', linestyle='dotted', label='Disk, Z [0,170]')
    
    plt.plot(r, same_Z_220_disk_2, color='green', marker='o', markeredgecolor='white', linestyle='dotted', label='Disk, Z [0,220]')
    
    plt.plot(r, same_Z_270_disk_2, color='purple', marker='o', markeredgecolor='white', linestyle='dotted', label='Disk, Z [0,270]')
        
    plt.plot(r, same_Z_320_disk_2, color='goldenrod', marker='o', markeredgecolor='white', linestyle='dotted', label='Disk, Z [0,320]')
    
    
    plt.xlabel('R (mm)'),
    plt.ylabel('Resolution FWHM (%)')
    plt.legend(loc='upper right', ncol=4, fontsize=6)
    plt.xlim(20,75)
    plt.ylim(3,6)
    
    plt.savefig('/home/afonso/data/results/plot_2_r_all_all.png', dpi = 300)
 
    # plt.show()
    
    plt.close()
    
    
    
    
    plt.figure()
    
    plt.title('7949 corrected with 4 runs')
    plt.plot(r, same_Z_70_ring_4, color='blue', marker='o', markeredgecolor='white', linestyle='dotted', label='Ring, Z [0,70]')
    
    plt.plot(r, same_Z_120_ring_4, color='orange', marker='o', markeredgecolor='white', linestyle='dotted', label='Ring, Z [70,120]')
    
    plt.plot(r, same_Z_170_ring_4, color='olive', marker='o', markeredgecolor='white', linestyle='dotted', label='Ring, Z [120,170]')
    
    plt.plot(r, same_Z_220_ring_4, color='red', marker='o', markeredgecolor='white', linestyle='dotted', label='Ring, Z [170,220]')
    
    plt.plot(r, same_Z_270_ring_4, color='brown', marker='o', markeredgecolor='white', linestyle='dotted', label='Ring, Z [220,270]')
        
    plt.plot(r, same_Z_320_ring_4, color='crimson', marker='o', markeredgecolor='white', linestyle='dotted', label='Ring, Z [270,320]')
    
    plt.plot(r, same_Z_70_disk_4, color='pink', marker='o', markeredgecolor='white', linestyle='dotted', label='Disk, Z [0,70]')
    
    plt.plot(r, same_Z_120_disk_4, color='cyan', marker='o', markeredgecolor='white', linestyle='dotted', label='Disk, Z [0,120]')
    
    plt.plot(r, same_Z_170_disk_4, color='gray', marker='o', markeredgecolor='white', linestyle='dotted', label='Disk, Z [0,170]')
    
    plt.plot(r, same_Z_220_disk_4, color='green', marker='o', markeredgecolor='white', linestyle='dotted', label='Disk, Z [0,220]')
    
    plt.plot(r, same_Z_270_disk_4, color='purple', marker='o', markeredgecolor='white', linestyle='dotted', label='Disk, Z [0,270]')
        
    plt.plot(r, same_Z_320_disk_4, color='goldenrod', marker='o', markeredgecolor='white', linestyle='dotted', label='Disk, Z [0,320]')
    
    
    plt.xlabel('R (mm)'),
    plt.ylabel('Resolution FWHM (%)')
    plt.legend(loc='upper right', ncol=4, fontsize=6)
    plt.xlim(20,75)
    plt.ylim(3,6)
    
    plt.savefig('/home/afonso/data/results/plot_4_r_all_all.png', dpi = 300)
 
    # plt.show()
    
    plt.close()
    
    
    plt.figure()
    
    plt.title('7949, 7950, 7951, 7952 corrected with 4 runs')
    plt.plot(r, same_Z_70_ring_4_4, color='blue', marker='o', markeredgecolor='white', linestyle='dotted', label='Ring, Z [0,70]')
    
    plt.plot(r, same_Z_120_ring_4_4, color='orange', marker='o', markeredgecolor='white', linestyle='dotted', label='Ring, Z [70,120]')
    
    plt.plot(r, same_Z_170_ring_4_4, color='olive', marker='o', markeredgecolor='white', linestyle='dotted', label='Ring, Z [120,170]')
    
    plt.plot(r, same_Z_220_ring_4_4, color='red', marker='o', markeredgecolor='white', linestyle='dotted', label='Ring, Z [170,220]')
    
    plt.plot(r, same_Z_270_ring_4_4, color='brown', marker='o', markeredgecolor='white', linestyle='dotted', label='Ring, Z [220,270]')
        
    plt.plot(r, same_Z_320_ring_4_4, color='crimson', marker='o', markeredgecolor='white', linestyle='dotted', label='Ring, Z [270,320]')
    
    plt.plot(r, same_Z_70_disk_4_4, color='pink', marker='o', markeredgecolor='white', linestyle='dotted', label='Disk, Z [0,70]')
    
    plt.plot(r, same_Z_120_disk_4_4, color='cyan', marker='o', markeredgecolor='white', linestyle='dotted', label='Disk, Z [0,120]')
    
    plt.plot(r, same_Z_170_disk_4_4, color='gray', marker='o', markeredgecolor='white', linestyle='dotted', label='Disk, Z [0,170]')
    
    plt.plot(r, same_Z_220_disk_4_4, color='green', marker='o', markeredgecolor='white', linestyle='dotted', label='Disk, Z [0,220]')
    
    plt.plot(r, same_Z_270_disk_4_4, color='purple', marker='o', markeredgecolor='white', linestyle='dotted', label='Disk, Z [0,270]')
        
    plt.plot(r, same_Z_320_disk_4_4, color='goldenrod', marker='o', markeredgecolor='white', linestyle='dotted', label='Disk, Z [0,320]')
    
    
    plt.xlabel('R (mm)'),
    plt.ylabel('Resolution FWHM (%)')
    plt.legend(loc='upper right', ncol=4, fontsize=6)
    plt.xlim(20,75)
    plt.ylim(3,6)
    
    plt.savefig('/home/afonso/data/results/plot_4_4_r_all_all.png', dpi = 300)
 
    # plt.show()
    
    plt.close()
    
    
    
    
    plt.figure()
    
    plt.title('7949 corrected with 1 run')
    plt.plot(r, same_Z_70_ring_1, color='blue', marker='o', markeredgecolor='white', linestyle='dotted', label='Ring, Z [0,70]')
        
    plt.plot(r, same_Z_320_ring_1, color='crimson', marker='o', markeredgecolor='white', linestyle='dotted', label='Ring, Z [270,320]')
    
    plt.plot(r, same_Z_70_disk_1, color='yellow', marker='o', markeredgecolor='white', linestyle='dotted', label='Disk, Z [0,70]')
        
    plt.plot(r, same_Z_320_disk_1, color='black', marker='o', markeredgecolor='white', linestyle='dotted', label='Disk, Z [0,320]')
    
    
    
    plt.xlabel('R (mm)'),
    plt.ylabel('Resolution FWHM (%)')
    plt.legend(loc='upper right', ncol=2, fontsize=6)
    plt.xlim(20,75)
    plt.ylim(3,6)
    
    plt.savefig('/home/afonso/data/results/plot_1_r_all.png', dpi = 300)
 
    # plt.show()
    
    plt.close()
    
    
    plt.figure()
    
    plt.title('7949 corrected with 2 runs')
    plt.plot(r, same_Z_70_ring_2, color='blue', marker='o', markeredgecolor='white', linestyle='dotted', label='Ring, Z [0,70]')
        
    plt.plot(r, same_Z_320_ring_2, color='crimson', marker='o', markeredgecolor='white', linestyle='dotted', label='Ring, Z [270,320]')
    
    plt.plot(r, same_Z_70_disk_2, color='yellow', marker='o', markeredgecolor='white', linestyle='dotted', label='Disk, Z [0,70]')
        
    plt.plot(r, same_Z_320_disk_2, color='black', marker='o', markeredgecolor='white', linestyle='dotted', label='Disk, Z [0,320]')
    
    
    
    plt.xlabel('R (mm)'),
    plt.ylabel('Resolution FWHM (%)')
    plt.legend(loc='upper right', ncol=2, fontsize=6)
    plt.xlim(20,75)
    plt.ylim(3,6)
    
    plt.savefig('/home/afonso/data/results/plot_2_r_all.png', dpi = 300)
 
    # plt.show()
    
    plt.close()
    
    
    
    plt.figure()
    
    plt.title('7949 corrected with 4 runs')
    plt.plot(r, same_Z_70_ring_4, color='blue', marker='o', markeredgecolor='white', linestyle='dotted', label='Ring, Z [0,70]')
        
    plt.plot(r, same_Z_320_ring_4, color='crimson', marker='o', markeredgecolor='white', linestyle='dotted', label='Ring, Z [270,320]')
    
    plt.plot(r, same_Z_70_disk_4, color='yellow', marker='o', markeredgecolor='white', linestyle='dotted', label='Disk, Z [0,70]')
        
    plt.plot(r, same_Z_320_disk_4, color='black', marker='o', markeredgecolor='white', linestyle='dotted', label='Disk, Z [0,320]')
    
    
    
    plt.xlabel('R (mm)'),
    plt.ylabel('Resolution FWHM (%)')
    plt.legend(loc='upper right', ncol=2, fontsize=6)
    plt.xlim(20,75)
    plt.ylim(3,6)
    
    plt.savefig('/home/afonso/data/results/plot_4_r_all.png', dpi = 300)
 
    # plt.show()
    
    plt.close()
    
    
    
    plt.figure()
    
    plt.title('7949, 7950, 7951, 7952 corrected with 4 runs')
    plt.plot(r, same_Z_70_ring_4_4, color='blue', marker='o', markeredgecolor='white', linestyle='dotted', label='Ring, Z [0,70]')
        
    plt.plot(r, same_Z_320_ring_4_4, color='crimson', marker='o', markeredgecolor='white', linestyle='dotted', label='Ring, Z [270,320]')
    
    plt.plot(r, same_Z_70_disk_4_4, color='yellow', marker='o', markeredgecolor='white', linestyle='dotted', label='Disk, Z [0,70]')
        
    plt.plot(r, same_Z_320_disk_4_4, color='black', marker='o', markeredgecolor='white', linestyle='dotted', label='Disk, Z [0,320]')
    
    
    
    plt.xlabel('R (mm)'),
    plt.ylabel('Resolution FWHM (%)')
    plt.legend(loc='upper right', ncol=2, fontsize=6)
    plt.xlim(20,75)
    plt.ylim(3,6)
    
    plt.savefig('/home/afonso/data/results/plot_4_4_r_all.png', dpi = 300)
 
    # plt.show()
    
    plt.close()
    
    
    
    
    ########################
    
    
    plt.figure()
    
    plt.title('7949 corrected with 1 run')
    plt.plot(z, same_R_30_ring_1, color='blue', marker='o', markeredgecolor='white', linestyle='dotted', label='Ring, R [0,30]')
        
    plt.plot(z, same_R_60_ring_1, color='crimson', marker='o', markeredgecolor='white', linestyle='dotted', label='Ring, R [55,60]')
    
    plt.plot(z, same_R_30_disk_1, color='yellow', marker='o', markeredgecolor='white', linestyle='dotted', label='Disk, R [0,30]')
        
    plt.plot(z, same_R_60_disk_1, color='black', marker='o', markeredgecolor='white', linestyle='dotted', label='Disk, R [0,60]')
    
    
    
    plt.xlabel('Z (mm)'),
    plt.ylabel('Resolution FWHM (%)')
    plt.legend(loc='upper right', ncol=2, fontsize=6)
    plt.xlim(0,350)
    plt.ylim(3,6)
    
    plt.savefig('/home/afonso/data/results/plot_1_z_all.png', dpi = 300)
 
    # plt.show()
    
    plt.close()
    
    
    plt.figure()
    
    plt.title('7949 corrected with 2 runs')
    plt.plot(z, same_R_30_ring_2, color='blue', marker='o', markeredgecolor='white', linestyle='dotted', label='Ring, R [0,30]')
        
    plt.plot(z, same_R_60_ring_2, color='crimson', marker='o', markeredgecolor='white', linestyle='dotted', label='Ring, R [55,60]')
    
    plt.plot(z, same_R_30_disk_2, color='yellow', marker='o', markeredgecolor='white', linestyle='dotted', label='Disk, R [0,30]')
        
    plt.plot(z, same_R_60_disk_2, color='black', marker='o', markeredgecolor='white', linestyle='dotted', label='Disk, R [0,60]')
    
    
    
    plt.xlabel('Z (mm)'),
    plt.ylabel('Resolution FWHM (%)')
    plt.legend(loc='upper right', ncol=2, fontsize=6)
    plt.xlim(0,350)
    plt.ylim(3,6)
    
    plt.savefig('/home/afonso/data/results/plot_2_z_all.png', dpi = 300)
 
    # plt.show()
    
    plt.close()
    
    
    
    plt.figure()
    
    plt.title('7949 corrected with 4 runs')
    plt.plot(z, same_R_30_ring_4, color='blue', marker='o', markeredgecolor='white', linestyle='dotted', label='Ring, R [0,30]')
        
    plt.plot(z, same_R_60_ring_4, color='crimson', marker='o', markeredgecolor='white', linestyle='dotted', label='Ring, R [55,60]')
    
    plt.plot(z, same_R_30_disk_4, color='yellow', marker='o', markeredgecolor='white', linestyle='dotted', label='Disk, R [0,30]')
        
    plt.plot(z, same_R_60_disk_4, color='black', marker='o', markeredgecolor='white', linestyle='dotted', label='Disk, R [0,60]')
    
    
    
    plt.xlabel('Z (mm)'),
    plt.ylabel('Resolution FWHM (%)')
    plt.legend(loc='upper right', ncol=2, fontsize=6)
    plt.xlim(0,350)
    plt.ylim(3,6)
    
    plt.savefig('/home/afonso/data/results/plot_4_z_all.png', dpi = 300)
 
    # plt.show()
    
    plt.close()
    
    
    
    plt.figure()
    
    plt.title('7949, 7950, 7951, 7952 corrected with 4 runs')
    plt.plot(z, same_R_30_ring_4_4, color='blue', marker='o', markeredgecolor='white', linestyle='dotted', label='Ring, R [0,30]')
        
    plt.plot(z, same_R_60_ring_4_4, color='crimson', marker='o', markeredgecolor='white', linestyle='dotted', label='Ring, R [55,60]')
    
    plt.plot(z, same_R_30_disk_4_4, color='yellow', marker='o', markeredgecolor='white', linestyle='dotted', label='Disk, R [0,30]')
        
    plt.plot(z, same_R_60_disk_4_4, color='black', marker='o', markeredgecolor='white', linestyle='dotted', label='Disk, R [0,60]')
    
    
    
    plt.xlabel('Z (mm)'),
    plt.ylabel('Resolution FWHM (%)')
    plt.legend(loc='upper right', ncol=2, fontsize=6)
    plt.xlim(0,350)
    plt.ylim(3,6)
    
    plt.savefig('/home/afonso/data/results/plot_4_4_z_all.png', dpi = 300)
 
    # plt.show()
    
    plt.close()
    
    
    
    
    
    
    plt.figure()
    
    plt.title('Ring plots')
    plt.plot(r, same_Z_70_ring_1, color='yellow', marker='o', markeredgecolor='white', linestyle='dotted', label='7949 corrected with 1 run, Z [0,70]')
    
    plt.plot(r, same_Z_70_ring_2, color='blue', marker='o', markeredgecolor='white', linestyle='dotted', label='7949 corrected with 2 runs, Z [0,70]')
    
    plt.plot(r, same_Z_70_ring_4, color='red', marker='o', markeredgecolor='white', linestyle='dotted', label='7949 corrected with 4 runs, Z [0,70]')
    
    plt.plot(r, same_Z_70_ring_4_4, color='brown', marker='o', markeredgecolor='white', linestyle='dotted', label='[7949,7950,7951,7952] corrected with 4 runs, Z [0,70]')
    
    plt.plot(r, same_Z_320_ring_1, color='pink', marker='o', markeredgecolor='white', linestyle='dotted', label='7949 corrected with 1 run, Z [270,320]')
    
    plt.plot(r, same_Z_320_ring_2, color='black', marker='o', markeredgecolor='white', linestyle='dotted', label='7949 corrected with 2 runs, Z [270,320]')
    
    plt.plot(r, same_Z_320_ring_4, color='olive', marker='o', markeredgecolor='white', linestyle='dotted', label='7949 corrected with 4 runs, Z [270,320]')
    
    plt.plot(r, same_Z_320_ring_4_4, color='crimson', marker='o', markeredgecolor='white', linestyle='dotted', label='[7949,7950,7951,7952] corrected with 4 runs, Z [270,320]')
    
    plt.xlabel('R (mm)'),
    plt.ylabel('Resolution FWHM (%)')
    plt.legend(loc='upper right', ncol=2, fontsize=6)
    plt.xlim(20,75)
    plt.ylim(3,6)
    
    plt.savefig('/home/afonso/data/results/plot_all_r_ring.png', dpi = 300)
 
    # plt.show()
    
    plt.close()
    
    
    
    
    plt.figure()
    
    plt.title('Disk plots')
    plt.plot(r, same_Z_70_disk_1, color='yellow', marker='o', markeredgecolor='white', linestyle='dotted', label='7949 corrected with 1 run, Z [0,70]')
    
    plt.plot(r, same_Z_70_disk_2, color='blue', marker='o', markeredgecolor='white', linestyle='dotted', label='7949 corrected with 2 runs, Z [0,70]')
    
    plt.plot(r, same_Z_70_disk_4, color='red', marker='o', markeredgecolor='white', linestyle='dotted', label='7949 corrected with 4 runs, Z [0,70]')
    
    plt.plot(r, same_Z_70_disk_4_4, color='brown', marker='o', markeredgecolor='white', linestyle='dotted', label='[7949,7950,7951,7952] corrected with 4 runs, Z [0,70]')
    
    plt.plot(r, same_Z_320_disk_1, color='pink', marker='o', markeredgecolor='white', linestyle='dotted', label='7949 corrected with 1 run, Z [0,320]')
    
    plt.plot(r, same_Z_320_disk_2, color='black', marker='o', markeredgecolor='white', linestyle='dotted', label='7949 corrected with 2 runs, Z [0,320]')
    
    plt.plot(r, same_Z_320_disk_4, color='olive', marker='o', markeredgecolor='white', linestyle='dotted', label='7949 corrected with 4 runs, Z [0,320]')
    
    plt.plot(r, same_Z_320_disk_4_4, color='crimson', marker='o', markeredgecolor='white', linestyle='dotted', label='[7949,7950,7951,7952] corrected with 4 runs, Z [0,320]')
    
    plt.xlabel('R (mm)'),
    plt.ylabel('Resolution FWHM (%)')
    plt.legend(loc='upper right', ncol=2, fontsize=6)
    plt.xlim(20,75)
    plt.ylim(3,6)
    
    plt.savefig('/home/afonso/data/results/plot_all_r_disk.png', dpi = 300)
 
    # plt.show()
    
    plt.close()

    
    
    plt.figure()
    
    plt.title('Ring plots')
    
    plt.plot(z, same_R_30_ring_1, color='yellow', marker='o', markeredgecolor='white', linestyle='dotted', label='7949 corrected with 1 run, R [0,30]')
    
    plt.plot(z, same_R_60_ring_1, color='blue', marker='o', markeredgecolor='white', linestyle='dotted', label='7949 corrected with 1 run, R [55,60]')
    
    plt.plot(z, same_R_30_ring_2, color='red', marker='o', markeredgecolor='white', linestyle='dotted', label='7949 corrected with 2 runs, R [0,30]')
    
    plt.plot(z, same_R_60_ring_2, color='pink', marker='o', markeredgecolor='white', linestyle='dotted', label='7949 corrected with 2 runs, R [55,60]')
    
    plt.plot(z, same_R_30_ring_4, color='black', marker='o', markeredgecolor='white', linestyle='dotted', label='7949 corrected with 4 runs, R [0,30]')
    
    plt.plot(z, same_R_60_ring_4, color='olive', marker='o', markeredgecolor='white', linestyle='dotted', label='7949 corrected with 4 runs, R [55,60]')
    
    plt.plot(z, same_R_30_ring_4_4, color='brown', marker='o', markeredgecolor='white', linestyle='dotted', label='[7949,7950,7951,7952] corrected with 4 runs, R [0,30]')
    
    plt.plot(z, same_R_60_ring_4_4, color='crimson', marker='o', markeredgecolor='white', linestyle='dotted', label='[7949,7950,7951,7952] corrected with 4 runs, R [55,60]')
    
    plt.xlabel('Z (mm)'),
    plt.ylabel('Resolution FWHM (%)')
    plt.legend(loc='upper right', ncol=2, fontsize=6)
    plt.xlim(0,350)
    plt.ylim(3,6)
    
    plt.savefig('/home/afonso/data/results/plot_all_z_ring.png', dpi = 300)
    
    # plt.show()
    
    plt.close()
    
    
    
    plt.figure()
    
    plt.title('Disk plots')
    plt.plot(z, same_R_30_disk_1, color='yellow', marker='o', markeredgecolor='white', linestyle='dotted', label='7949 corrected with 1 run, R [0,30]')
    
    plt.plot(z, same_R_60_disk_1, color='blue', marker='o', markeredgecolor='white', linestyle='dotted', label='7949 corrected with 1 run, R [0,60]')
    
    plt.plot(z, same_R_30_disk_2, color='red', marker='o', markeredgecolor='white', linestyle='dotted', label='7949 corrected with 2 runs, R [0,30]')
    
    plt.plot(z, same_R_60_disk_2, color='pink', marker='o', markeredgecolor='white', linestyle='dotted', label='7949 corrected with 2 runs, R [0,60]')
    
    plt.plot(z, same_R_30_disk_4, color='black', marker='o', markeredgecolor='white', linestyle='dotted', label='7949 corrected with 4 runs, R [0,30]')
    
    plt.plot(z, same_R_60_disk_4, color='olive', marker='o', markeredgecolor='white', linestyle='dotted', label='7949 corrected with 4 runs, R [0,60]')
    
    plt.plot(z, same_R_30_disk_4_4, color='brown', marker='o', markeredgecolor='white', linestyle='dotted', label='[7949,7950,7951,7952] corrected with 4 runs, R [0,30]')
    
    plt.plot(z, same_R_60_disk_4_4, color='crimson', marker='o', markeredgecolor='white', linestyle='dotted', label='[7949,7950,7951,7952] corrected with 4 runs, R [0,60]')
    
    plt.xlabel('Z (mm)'),
    plt.ylabel('Resolution FWHM (%)')
    plt.legend(loc='upper right', ncol=2, fontsize=6)
    plt.xlim(0,350)
    plt.ylim(3,6)
    
    plt.savefig('/home/afonso/data/results/plot_all_z_disk.png', dpi = 300)
    
#     plt.show()
    
    plt.close()
    
    
    
    plt.figure()
    
    plt.title('Ring plots')
    
    plt.plot(z, same_R_30_ring_1, color='yellow', marker='o', markeredgecolor='white', linestyle='dotted', label='7949 corrected with 1 run, R [0,30]')
        
    plt.plot(z, same_R_30_ring_2, color='blue', marker='o', markeredgecolor='white', linestyle='dotted', label='7949 corrected with 2 runs, R [0,30]')
        
    plt.plot(z, same_R_30_ring_4, color='red', marker='o', markeredgecolor='white', linestyle='dotted', label='7949 corrected with 4 runs, R [0,30]')
        
    plt.plot(z, same_R_30_ring_4_4, color='black', marker='o', markeredgecolor='white', linestyle='dotted', label='[7949,7950,7951,7952] corrected with 4 runs, R [0,30]')
        
    plt.xlabel('Z (mm)'),
    plt.ylabel('Resolution FWHM (%)')
    plt.legend(loc='upper right', ncol=2, fontsize=6)
    plt.xlim(0,350)
    plt.ylim(3,6)
    
    plt.savefig('/home/afonso/data/results/plot_r_0_30_ring.png', dpi = 300)
    
#     plt.show()
    
    plt.close()
    
    
    
    
    plt.figure()
    
    plt.title('Disk plots')
    plt.plot(z, same_R_30_disk_1, color='yellow', marker='o', markeredgecolor='white', linestyle='dotted', label='7949 corrected with 1 run, R [0,30]')
        
    plt.plot(z, same_R_30_disk_2, color='blue', marker='o', markeredgecolor='white', linestyle='dotted', label='7949 corrected with 2 runs, R [0,30]')
        
    plt.plot(z, same_R_30_disk_4, color='red', marker='o', markeredgecolor='white', linestyle='dotted', label='7949 corrected with 4 runs, R [0,30]')
        
    plt.plot(z, same_R_30_disk_4_4, color='black', marker='o', markeredgecolor='white', linestyle='dotted', label='[7949,7950,7951,7952] corrected with 4 runs, R [0,30]')
        
    plt.xlabel('Z (mm)'),
    plt.ylabel('Resolution FWHM (%)')
    plt.legend(loc='upper right', ncol=2, fontsize=6)
    plt.xlim(0,350)
    plt.ylim(3,6)
    
    plt.savefig('/home/afonso/data/results/plot_r_0_30_disk.png', dpi = 300)
    
#     plt.show()
    
    plt.close()
    
    
    plt.figure()
    
    plt.title('Ring plots')
        
    plt.plot(z, same_R_60_ring_1, color='yellow', marker='o', markeredgecolor='white', linestyle='dotted', label='7949 corrected with 1 run, R [55,60]')
        
    plt.plot(z, same_R_60_ring_2, color='blue', marker='o', markeredgecolor='white', linestyle='dotted', label='7949 corrected with 2 runs, R [55,60]')
        
    plt.plot(z, same_R_60_ring_4, color='red', marker='o', markeredgecolor='white', linestyle='dotted', label='7949 corrected with 4 runs, R [55,60]')
        
    plt.plot(z, same_R_60_ring_4_4, color='black', marker='o', markeredgecolor='white', linestyle='dotted', label='[7949,7950,7951,7952] corrected with 4 runs, R [55,60]')
    
    plt.xlabel('Z (mm)'),
    plt.ylabel('Resolution FWHM (%)')
    plt.legend(loc='upper right', ncol=2, fontsize=6)
    plt.xlim(0,350)
    plt.ylim(3,6)
    
    plt.savefig('/home/afonso/data/results/plot_r_55_60_ring.png', dpi = 300)
    
#     plt.show()
    
    plt.close()
    
    
    
    plt.figure()
    
    plt.title('Disk plots')
    
    plt.plot(z, same_R_60_disk_1, color='yellow', marker='o', markeredgecolor='white', linestyle='dotted', label='7949 corrected with 1 run, R [0,60]')
        
    plt.plot(z, same_R_60_disk_2, color='blue', marker='o', markeredgecolor='white', linestyle='dotted', label='7949 corrected with 2 runs, R [0,60]')
        
    plt.plot(z, same_R_60_disk_4, color='red', marker='o', markeredgecolor='white', linestyle='dotted', label='7949 corrected with 4 runs, R [0,60]')
        
    plt.plot(z, same_R_60_disk_4_4, color='black', marker='o', markeredgecolor='white', linestyle='dotted', label='[7949,7950,7951,7952] corrected with 4 runs, R [0,60]')
    
    plt.xlabel('Z (mm)'),
    plt.ylabel('Resolution FWHM (%)')
    plt.legend(loc='upper right', ncol=2, fontsize=6)
    plt.xlim(0,350)
    plt.ylim(3,6)
    
    plt.savefig('/home/afonso/data/results/plot_r_0_60_disk.png', dpi = 300)
    
#     plt.show()
    
    plt.close()
    
    
    
    
    
    
    plt.figure()
    
    plt.title('Rins plots')
    plt.plot(r, same_Z_70_ring_1, color='yellow', marker='o', markeredgecolor='white', linestyle='dotted', label='7949 corrected with 1 run, Z [0,70]')
    
    plt.plot(r, same_Z_70_ring_2, color='blue', marker='o', markeredgecolor='white', linestyle='dotted', label='7949 corrected with 2 runs, Z [0,70]')
    
    plt.plot(r, same_Z_70_ring_4, color='red', marker='o', markeredgecolor='white', linestyle='dotted', label='7949 corrected with 4 runs, Z [0,70]')
    
    plt.plot(r, same_Z_70_ring_4_4, color='black', marker='o', markeredgecolor='white', linestyle='dotted', label='[7949,7950,7951,7952] corrected with 4 runs, Z [0,70]')
    
    
    plt.xlabel('R (mm)'),
    plt.ylabel('Resolution FWHM (%)')
    plt.legend(loc='upper right', ncol=2, fontsize=6)
    plt.xlim(20,75)
    plt.ylim(3,6)
    
    plt.savefig('/home/afonso/data/results/plot_z_0_70_ring.png', dpi = 300)
 
    # plt.show()
    
    plt.close()
    
    
    plt.figure()
    
    plt.title('Ring plots')
    plt.plot(r, same_Z_320_ring_1, color='yellow', marker='o', markeredgecolor='white', linestyle='dotted', label='7949 corrected with 1 run, Z [270,320]')
    
    plt.plot(r, same_Z_320_ring_2, color='blue', marker='o', markeredgecolor='white', linestyle='dotted', label='7949 corrected with 2 runs, Z [270,320]')
    
    plt.plot(r, same_Z_320_ring_4, color='red', marker='o', markeredgecolor='white', linestyle='dotted', label='7949 corrected with 4 runs, Z [270,320]')
    
    plt.plot(r, same_Z_320_ring_4_4, color='black', marker='o', markeredgecolor='white', linestyle='dotted', label='[7949,7950,7951,7952] corrected with 4 runs, Z [270,320]')
    
    plt.xlabel('R (mm)'),
    plt.ylabel('Resolution FWHM (%)')
    plt.legend(loc='upper right', ncol=2, fontsize=6)
    plt.xlim(20,75)
    plt.ylim(3,6)
    
    plt.savefig('/home/afonso/data/results/plot_z_270_320_ring.png', dpi = 300)
 
    # plt.show()
    
    plt.close()
    
    
    
    plt.figure()
    
    plt.title('Disk plots')
    plt.plot(r, same_Z_70_disk_1, color='yellow', marker='o', markeredgecolor='white', linestyle='dotted', label='7949 corrected with 1 run, Z [0,70]')
    
    plt.plot(r, same_Z_70_disk_2, color='blue', marker='o', markeredgecolor='white', linestyle='dotted', label='7949 corrected with 2 runs, Z [0,70]')
    
    plt.plot(r, same_Z_70_disk_4, color='red', marker='o', markeredgecolor='white', linestyle='dotted', label='7949 corrected with 4 runs, Z [0,70]')
    
    plt.plot(r, same_Z_70_disk_4_4, color='black', marker='o', markeredgecolor='white', linestyle='dotted', label='[7949,7950,7951,7952] corrected with 4 runs, Z [0,70]')
    
    
    plt.xlabel('R (mm)'),
    plt.ylabel('Resolution FWHM (%)')
    plt.legend(loc='upper right', ncol=2, fontsize=6)
    plt.xlim(20,75)
    plt.ylim(3,6)
    
    plt.savefig('/home/afonso/data/results/plot_z_0_70_disk.png', dpi = 300)
 
    # plt.show()
    
    plt.close()
    
    
    plt.figure()
    
    plt.title('Disk plots')
    plt.plot(r, same_Z_320_disk_1, color='yellow', marker='o', markeredgecolor='white', linestyle='dotted', label='7949 corrected with 1 run, Z [0,320]')
    
    plt.plot(r, same_Z_320_disk_2, color='blue', marker='o', markeredgecolor='white', linestyle='dotted', label='7949 corrected with 2 runs, Z [0,320]')
    
    plt.plot(r, same_Z_320_disk_4, color='red', marker='o', markeredgecolor='white', linestyle='dotted', label='7949 corrected with 4 runs, Z [0,320]')
    
    plt.plot(r, same_Z_320_disk_4_4, color='black', marker='o', markeredgecolor='white', linestyle='dotted', label='[7949,7950,7951,7952] corrected with 4 runs, Z [0,320]')
    
    plt.xlabel('R (mm)'),
    plt.ylabel('Resolution FWHM (%)')
    plt.legend(loc='upper right', ncol=2, fontsize=6)
    plt.xlim(20,75)
    plt.ylim(3,6)
    
    plt.savefig('/home/afonso/data/results/plot_z_270_320_disk.png', dpi = 300)
 
    # plt.show()
    
    plt.close()
    
    
    
    
    
def energy_fits(gaussC, dst, corr, opt_dict):
    
    bins_res = int(opt_dict["bins_res"])
    
    fit_min_res = int(opt_dict["fit_min_res"])
    fit_max_res = int(opt_dict["fit_max_res"])
    
    chi2 = BinnedChi2(gaussC, dst.S2e*corr, bins = bins_res, bound = (fit_min_res, fit_max_res))
    chi2_1 = chi2.ndof
    m = Minuit(chi2, mu = 10500, sigma = 150, N = 200, Ny = 10)
    m.migrad()
    
    mean     = m.values[0]
    mean_u   = m.errors[0]

    sigma    = m.values[1]
    sigma_u  = m.errors[1]

    N       = m.values[2]
    N_u     = m.errors[2]

    N2        = m.values[3]
    N2_u      = m.errors[3]
    
    chi2_2    = m.fval
    
    resolution, fig = plot_residuals_E_reso_gaussC('', '', dst.S2e*corr, bins_res, (fit_min_res, fit_max_res), mean, mean_u , sigma, sigma_u, N, N_u, N2, N2_u, chi2_1, chi2_2, 'All region')
    
    return fig
   
    






