import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from krcal.map_builder.map_builder_functions  import calculate_map
from krcal.map_builder.map_builder_functions  import check_failed_fits
from krcal.map_builder.map_builder_functions  import regularize_map
from krcal.map_builder.map_builder_functions  import remove_peripheral

from krcal.map_builder.map_builder_functions  import add_krevol
from krcal.core.io_functions                  import write_complete_maps
from krcal.core.map_functions                 import add_mapinfo

from invisible_cities.core. core_functions    import in_range
from map_builder.plots                        import control_plots, time_evolution_plots

from manage_data                              import load_runs_parameter


def map_creation(dst, opt_dict):
    """
    Input a dst and config file for map creation
    Returns the regularized map for after-time evolution computation
    """
    

    runs          = load_runs_parameter(opt_dict)
    run           = runs[0]
    rmax          = int(opt_dict['rmax'])
    xy_num_bins   = int(opt_dict['xy_num_bins'])
    
    z_bin         = int(opt_dict['z_bin'])
    e_bin         = int(opt_dict['e_bin'])
    
    chi2_min      = int(opt_dict['chi2_min'])
    chi2_max      = int(opt_dict['chi2_max'])
    
    fit_type      = opt_dict['fit_type']
    
    nmin          = int(opt_dict['nmin'])
    time_evo      = int(opt_dict['time_evo'])
    
    print('Beginning map creation...\n')

    maps = calculate_map(dst        = dst             ,
                         XYbins     = (xy_num_bins    ,
                                       xy_num_bins)   ,
                         nbins_z    = z_bin           ,
                         nbins_e    = e_bin           ,
                         z_range    = (np.min(dst.Z)  ,
                                       np.max(dst.Z) ),
                         e_range    = (np.min(dst.S2e),  
                                       np.max(dst.S2e)),
                         chi2_range = (chi2_min       ,
                                       chi2_max)      ,
                         lt_range   = (0, 1)          , 
                         fit_type   = fit_type        ,
                         nmin       = nmin             ,
                         x_range    = (-60, 60)       ,
                         y_range    = (-60, 60)       )
    
    print('Map created successfully!\n')
    
    check_failed_fits(maps      = maps                ,
                      maxFailed = 200                 ,
                      nbins     = xy_num_bins         ,
                      rmax      = 60                  ,
                      rfid      = 60                  )
    
    print('\nProceeding to regularization...\n')

    
    regularized_maps = regularize_map(maps    = maps                ,
                                      x2range = (chi2_min, chi2_max))
        

    regularized_maps = remove_peripheral(map   = regularized_maps ,
                                         nbins = xy_num_bins      ,
                                         rmax  = 60               ,
                                         rfid  = 60               )
    
    print('Map regularization successful!\n')

    
    regularized_maps = add_mapinfo(asm        = regularized_maps       ,
                       xr         = (-60, 60)  ,
                       yr         = (-60, 60)  ,
                       nx         = xy_num_bins,
                       ny         = xy_num_bins,
                       run_number = int(run)        )
    
    print(regularized_maps.mapinfo, '\n')
    
    return regularized_maps


def time_evolution_computation(dst, maps, dir_out, opt_dict):
    """
    Input a dst, a map, a directory path and config file for time evolution computation
    Returns the map with time evolution
    """
    
    runs          = load_runs_parameter(opt_dict)
    
    label         = ''
    
    for run in runs:
        label     += str(run) + '_'
    
    time_evo      = int(opt_dict['time_evo'])
    xy_num_bins   = int(opt_dict['xy_num_bins'])
    
    nbins_dv      = int(opt_dict['nbins_dv'])
    
    fit_type      = opt_dict['fit_type']
    
    
    print('\nProceeding to adding time evolution...\n')

    mask = pd.DataFrame({'s1': dst.nS1 == 1, 's2': dst.nS2 == 1, 'band': in_range(dst.S2e, np.min(dst.S2e), np.max(dst.S2e))})

    add_krevol(maps          = maps                 ,
               masks_cuts    = mask                 ,
               dst           = dst                  ,
               r_fid         = 59                   ,
               nStimeprofile = time_evo             ,
               x_range       = (-60, 60)            ,
               y_range       = (-60, 60)            ,
               XYbins        = (xy_num_bins         ,
                                xy_num_bins        ),
               detector      = 'demopp'             ,
               zrange_lt     = (np.min(dst.Z)       , 
                                np.max(dst.Z))      ,
               zslices_lt    = 15                   ,
               zrange_dv     = (300, 380)           ,
               nbins_dv      = nbins_dv             )
        
    map_file_out = dir_out + '/kr_map_{}{}bins_{}.h5'.format(label,xy_num_bins, fit_type.split('.')[-1])
    

    write_complete_maps(asm      = maps        ,
                        filename = map_file_out)
    
    
    print('\nTime evolution added and final map saved in {}\n'.format(map_file_out))
    
    return maps
    
    
    


    




    


