import os
from manage_data               import load_dst_for_map, dst_cleaning, load_runs_parameter, merge_dst_for_map_production, check_if_dst_exists
import pandas as pd


def ana_create_kdst_map(dir_in, opt_dict):
    """
    Input the directory path where dsts are (example, if the dsts are in /home/afonso/data/{run_number}/kdst/{files}), dir_in should be /home/afonso/data
    Returns the dst for control plots, map creation and time evolution computation
    """
    
    check = check_if_dst_exists(dir_in, opt_dict)
        
    if all(check) == True:
        dst = merge_dst_for_map_production(dir_in, opt_dict)
        
    else:
        load_dst_for_map(dir_in, opt_dict, save=True) 
        dst = merge_dst_for_map_production(dir_in, opt_dict)
    
    
    dst_dv, dst_map = dst_cleaning(dst, opt_dict)
    
    return dst, dst_dv, dst_map
        
    