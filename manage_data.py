import os
import shutil
import numpy  as np
import pandas as pd

from ana_stat import error_eff

from  invisible_cities.io.dst_io import load_dst
from  invisible_cities.io.dst_io import load_dsts
from  invisible_cities.core .core_functions import in_range

from invisible_cities.reco.corrections       import apply_all_correction

from krcal.map_builder.map_builder_functions import e0_xy_correction

def save_dst_to_file(dst, dir_file):
    """
    Input path and file name to write a dst
    Output dst written to disk
    """
    dst = dst.sort_values('event')
    store = pd.HDFStore(dir_file, "w", complib=str("zlib"), complevel=4)
    store.put('dataframe', dst, format='table', data_columns=True)
    store.close()


def create_dirs(dir):
    """
    Input directory name
    Output creates directory
    """
    #  to do: si escribimos ENTER que sea equivalente a NO

    '''
    print('Do you want to overwrite? :' + dir)
    answer = input()
    if answer == 'y':
        overwrite = True
    else:
        overwrite = False
    print(overwrite)
    '''

    # big production
    overwrite = 'n'

#     if os.path.exists(dir) 
#     and overwrite:
#         shutil.rmtree(dir)
    try:
        os.makedirs(dir)
    except OSError:
        print (f'Creation of dir: {dir} failed')
    else:
        print (f'Successfully created dir: {dir}')


def load_data(fout, dir_in, run):
    """
    Input path to all dst files
    Return a merged dst
    ------
    Note: In the kdst there are duplicated events because of the
    multiplicity from both s1 and s2
    Note two: dst.event.nunique returns int. (Panda.Series.nunique)
    if used in Panda.Dataframe.nunique returns Series with unique entries
    """

    path        = dir_in + '/' + run + '/kdst/'
    files_all   = [path + f for f in os.listdir(path) \
                  if os.path.isfile( os.path.join(path, f) )]
    dst         = load_dsts(files_all, "DST", "Events")
    time_run    = dst.time.mean()

    # count number of number of unique entries
    unique_events = ~dst.event.duplicated()
    #unique_events = dst.event.nunique()
    nunique_events = dst.event.nunique()

    #print(nunique_events)

    num_of_S2s  = np.size         (unique_events)
    num_of_evts = np.count_nonzero(unique_events)

    print(num_of_evts)
    fout.write(f"dst_entries {str(len(dst))}\n")
    fout.write(f"time_run {time_run}\n")
    fout.write(f"s2_tot {num_of_S2s}\n")
    fout.write(f"evt_tot {num_of_evts}\n")

    # compute number of s1 and s2
    df = dst[~dst.time.duplicated()]
    tot_ev = df.event.nunique()
    s1_num = df.nS1.values
    s2_num = df.nS2.values
    fout.write(f"num_of_ev_check {tot_ev}\n")

    s1_0 = np.count_nonzero(s1_num == 0)
    s1_1 = np.count_nonzero(s1_num == 1)
    s1_2 = np.count_nonzero(s1_num == 2)
    s1_3 = np.count_nonzero(s1_num == 3)
    s1_4 = np.count_nonzero(s1_num == 4)
    s1_5 = np.count_nonzero(s1_num == 5)
    s1_6 = np.count_nonzero(s1_num == 6)

    s2_0 = np.count_nonzero(s2_num == 0)
    s2_1 = np.count_nonzero(s2_num == 1)
    s2_2 = np.count_nonzero(s2_num == 2)
    s2_3 = np.count_nonzero(s2_num == 3)
    s2_4 = np.count_nonzero(s2_num == 4)
    s2_5 = np.count_nonzero(s2_num == 5)
    s2_6 = np.count_nonzero(s2_num == 6)
    s2_7 = np.count_nonzero(s2_num == 7)
    s2_8 = np.count_nonzero(s2_num == 8)

    fout.write(f'eff_0s1  {s1_0 /tot_ev*100:.5f}\n')
    fout.write(f'eff_0s1_u  {error_eff(tot_ev, s1_0 /tot_ev)*100:.5f}\n')

    fout.write(f'eff_1s1  {s1_1 /tot_ev*100:.5f}\n')
    fout.write(f'eff_1s1_u  {error_eff(tot_ev, s1_1 /tot_ev)*100:.5f}\n')

    fout.write(f'eff_2s1  {s1_2 /tot_ev*100:.5f}\n')
    fout.write(f'eff_2s1_u  {error_eff(tot_ev, s1_2 /tot_ev)*100:.5f}\n')

    fout.write(f'eff_3s1  {s1_3 /tot_ev*100:.5f}\n')
    fout.write(f'eff_3s1_u  {error_eff(tot_ev, s1_3 /tot_ev)*100:.5f}\n')

    fout.write(f'eff_4s1  {s1_4 /tot_ev*100:.5f}\n')
    fout.write(f'eff_4s1_u  {error_eff(tot_ev, s1_4 /tot_ev)*100:.5f}\n')

    fout.write(f'eff_5s1  {s1_5 /tot_ev*100:.5f}\n')
    fout.write(f'eff_5s1_u  {error_eff(tot_ev, s1_5 /tot_ev)*100:.5f}\n')

    fout.write(f'eff_6s1  {s1_6 /tot_ev*100:.5f}\n')
    fout.write(f'eff_6s1_u  {error_eff(tot_ev, s1_6 /tot_ev)*100:.5f}\n')

# s2 eff
    fout.write(f'eff_0s2  {s2_0 /tot_ev*100:.5f}\n')
    fout.write(f'eff_0s2_u  {error_eff(tot_ev, s2_0/tot_ev)*100:.5f}\n')

    fout.write(f'eff_1s2  {s2_1 /tot_ev*100:.5f}\n')
    fout.write(f'eff_1s2_u  {error_eff(tot_ev, s2_1/tot_ev)*100:.5f}\n')

    fout.write(f'eff_2s2  {s2_2 /tot_ev*100:.5f}\n')
    fout.write(f'eff_2s2_u  {error_eff(tot_ev, s2_2/tot_ev)*100:.5f}\n')

    fout.write(f'eff_3s2  {s2_3 /tot_ev*100:.5f}\n')
    fout.write(f'eff_3s2_u  {error_eff(tot_ev, s2_3/tot_ev)*100:.5f}\n')

    fout.write(f'eff_4s2  {s2_4 /tot_ev*100:.5f}\n')
    fout.write(f'eff_4s2_u  {error_eff(tot_ev, s2_4/tot_ev)*100:.5f}\n')

    fout.write(f'eff_5s2  {s2_5 /tot_ev*100:.5f}\n')
    fout.write(f'eff_5s2_u  {error_eff(tot_ev, s2_5/tot_ev)*100:.5f}\n')

    fout.write(f'eff_6s2  {s2_6 /tot_ev*100:.5f}\n')
    fout.write(f'eff_6s2_u  {error_eff(tot_ev, s2_6/tot_ev)*100:.5f}\n')

    fout.write(f'eff_7s2  {s2_7 /tot_ev*100:.5f}\n')
    fout.write(f'eff_7s2_u  {error_eff(tot_ev, s2_7/tot_ev)*100:.5f}\n')

    fout.write(f'eff_8s2  {s2_8 /tot_ev*100:.5f}\n')
    fout.write(f'eff_8s2_u  {error_eff(tot_ev, s2_8/tot_ev)*100:.5f}\n')


    return dst

def s1s2_selection(dst, fout, dst_out_dir, run, rmax, save=False):
    """
    Input one file with a dst dataframe and run number
    Returns dst and writes to disk a reduced dst with events that pass
    the 1s1 and 1s2 selection criteria
    """

    dst_s1   = dst     [in_range(dst.nS1,    1,2)]
    dst_s2   = dst_s1  [in_range(dst_s1.nS2, 1,2)]


    tot_ev  = dst.   event.nunique()
    s1_ev   = dst_s1.event.nunique()
    s1s2_ev = dst_s2.event.nunique()

    eff_s1      = s1_ev   / tot_ev
    eff_s2      = s1s2_ev / tot_ev
    eff_s1s2    = s1s2_ev / s1_ev

    fout.write(f'ev_1s1 {s1_ev }\n')
    fout.write(f'ev_1s1s2 {s1s2_ev }\n')

    fout.write(f'eff_1s1_check {eff_s1*100:.5f}\n')
    fout.write(f'eff_1s1_u_check {error_eff(tot_ev, s1_ev/tot_ev)*100:.5f}\n')

    fout.write(f'eff_1s1s2 {eff_s2*100:.5f}\n')
    fout.write(f'eff_1s1s2_u {error_eff(tot_ev, s1s2_ev/tot_ev)*100:.5f}\n')

    fout.write(f'rel_eff_1s2_from_1s1  {eff_s1s2*100:.5f}\n')
    fout.write(f'rel_eff_1s2_from_1s1_u {error_eff(s1_ev, s1s2_ev/s1_ev)*100:.5f}\n')


    if save:
        dir_file_name = f'{dst_out_dir}/reduced_{run}_kdst_{rmax}.h5'
        save_dst_to_file(dst_s2, dir_file_name)
        print(f'Save reduced kdst with 1s1 and 1s2 in: {dir_file_name}')

    return dst_s2

def radial_selection(dst, fout, dst_out_dir, run, rfid , save=False):
    """
    Input a dst
    Return dst with radial requirement applied
    """

    #rfid     = int(rfid)
    dst_rfid = dst[in_range(dst.R, 0, rfid)]
    tot_ev   = dst.event.nunique()
    rfid_ev  = dst_rfid.event.nunique()
    eff      = rfid_ev/tot_ev

    s1e_mean = dst_rfid.S1e.mean()
    s1e_median = dst_rfid.S1e.median()

    #print(f'Rel_eff_R_{rfid} = {np.round(eff*100,2)}%  ({rfid_ev} / {tot_ev})\n')
    fout.write(f'ev_r_fid {rfid_ev}\n')
    fout.write(f'rel_eff_r_{rfid} {eff*100:.5f}\n')
    fout.write(f'rel_eff_r_{rfid}_u {error_eff(tot_ev, eff)*100:.5f}\n')

    fout.write(f's1e_mean {s1e_mean:3f}\n')
    fout.write(f's1e_median {s1e_median:3f}\n')

    if save:
        dir_file_name = f'{dst_out_dir}/reduced_{run}_kdst_{rfid}.h5'
        save_dst_to_file(dst_rfid, dir_file_name)
        print(f'Save reduced kdst with R < {rfid} : {dir_file_name}')

    return dst_rfid

def energy_selection(dst, opt_dict, fout, dst_out_dir, run, save=False):
    """
    Input a dst
    Return dst with energy range requirement applied
    """

    emin   = float(opt_dict["s2e_sig_min"])
    emax   = float(opt_dict["s2e_sig_max"])
    dst_e = dst[in_range(dst.S2e, emin, emax)]

    tot_ev     = dst.event.nunique()
    energy_ev  = dst_e.event.nunique()
    eff        = energy_ev/tot_ev

    print(f'rel_eff_e {np.round(eff*100,2)}\n')

    fout.write(f'rel_eff_e {eff*100:.5f}\n')
    fout.write(f'rel_eff_e_u {error_eff(tot_ev, eff)*100:.5f}\n')

    if save:
        
        dir_file_name = f'{dst_out_dir}/reduced_{run}_kdst_emin{emin}_emax{emax}.h5'
        save_dst_to_file(dst_e, dir_file_name)
        print(f'Save reduced kdst with e = [{emin,emax}]: {dir_file_name}')

    return dst_e


def load_runs_parameter(opt_dict):
    """
    Input the config file
    Returns dst a list of the runs for later directory creation, dst loading, ...
    """
    
    
    runs_to_load = opt_dict["runs"] # if we load like this, it will load character by character
                                    # which means, for example, if in the config file runs = [7949, 7950],
                                    # print(runs_to_load) = ['[', 7', '9', '4', '9', ',', ' ', '7', '9', '5', '0', ']']
    runs = []
    
    for i in runs_to_load:
        if i.isnumeric():
            runs.append(i)         # to remove the commas and the brackets 
            
    runs = ''.join(runs)
    runs = [runs[i : i + 4] for i in range(0, len(runs), 4)] # to join the individual strings (we suppose that the runs are in format xxxx)
       
    return runs
        


def load_dst_for_map(dir_in, opt_dict, save):
    """
    Input a directory where dsts are stored, a config file and a boolean to choose if the merged dst are saved
    """
    
    runs = load_runs_parameter(opt_dict)
    
    print('\nRuns to load:', runs, '\n')

    for run in runs:
        
        frames = []

        path = dir_in + run + '/kdst/'
            
        for ifile in range(0, 10000):
            file = path + 'kdst_{0000:04n}'.format(ifile) + '_{}_trigger1_v1.2.0_20181011-19-g25d838b_demo-kdst.h5'.format(run)
            if os.path.exists(file): 
                dst = load_dst(file, 'DST', 'Events')  
                frames.append(dst[(dst.R < 70) & (dst.nS1 == 1) & (dst.nS2 == 1)])

        dst = pd.concat(frames, ignore_index = True)
    
        dst_out_dir = dir_in + 'results/' + run + '/kdst-reduced'
        
        if save:
            create_dirs(dir_in + 'results/' + run)
            create_dirs(dst_out_dir)
            dir_file_name = f'{dst_out_dir}/kdst_tot_{run}.h5'
            save_dst_to_file(dst, dir_file_name)
            print(f'Run {run} with nS1 ans nS2 = 1 saved in: {dir_file_name}')
            
        del dst, frames


def check_if_dst_exists(dir_in, opt_dict):
    """
    Input a directory where dst is and a config file
    Checks if the merged dst already exists, to save map creation time
    """
    
    runs = load_runs_parameter(opt_dict)
    
    check = []
    
    for run in runs:
        file = dir_in + 'results/' + run + '/kdst-reduced/kdst_tot_' + run + '.h5'
        if os.path.exists(file):
            check.append(True)
            print('\n Run {} merged file already exists'.format(run))
        else:
            check.append(False)
            print('\nRun {} merged file does not exists'.format(run))
            
    print('')
    
    return check



def merge_dst_for_map_production(dir_in, opt_dict):
    """
    Input a directory where dst is and a config file
    Returns the dsts for the desired runs
    """
    
    frames_kdst = []
    max_evt = 0
    
    runs = load_runs_parameter(opt_dict)

    for index, run in enumerate(runs):
        file = dir_in + 'results/' + run + '/kdst-reduced/kdst_tot_' + run + '.h5'
        if os.path.exists(file):
            kdst = load_dst(file, 'dataframe', 'table')
            kdst['eventID'] = kdst.event
            kdst.event = kdst.event + max_evt
            run_column = [run] * len(kdst)
            kdst['run_number'] = run_column
            print(f'Events in {file} = {kdst.event.nunique()}')
            frames_kdst.append(kdst)
            max_evt = kdst.event.max() + 1
            kdst_all = pd.concat(frames_kdst, ignore_index=True)
            print(f'Events in the merge of {run} = {kdst_all.event.nunique()}')
            
    return kdst_all
    

def dst_cleaning(dst, opt_dict):
    """
    Input the dst and a config file
    Returns the dst cleaned in several variables for map production and time evolution computation
    """
    
    r_clean        = int(opt_dict["r_clean"])
    
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

    zdv_clean_min  = int(opt_dict["zdv_clean_min"])
    zdv_clean_max  = int(opt_dict["zdv_clean_max"])

    zmap_clean_min  = int(opt_dict["zmap_clean_min"])
    zmap_clean_max  = int(opt_dict["zmap_clean_max"])
    
    
    print('Data cleaning begins:\n\n     R     < {:.2f} mm\n     S2e   = [{:.2f}, {:.2f}] pes\n     S1e   = [{:.2f}, {:.2f}] pes\n     S2w   = [{:.2f}, {:.2f}] \u03BCs\n     S2q   = [{:.2f}, {:.2f}] pes\n     Nsipm = [{:.2f}, {:.2f}]\n     Z_map = [{:.2f}, {:.2f}] mm\n     Z_dv  = [{:.2f}, {:.2f}] mm\n'
          .format(r_clean, s2e_clean_min, s2e_clean_max, s1e_clean_min, s1e_clean_max, s2w_clean_min, s2w_clean_max, s2q_clean_min, s2q_clean_max, nsipm_clean_min, nsipm_clean_max, 
                 zmap_clean_min, zmap_clean_max, zdv_clean_min, zdv_clean_max))
    
    mask_r     = in_range(dst.R, 0, r_clean)
    
    mask_s2e   = in_range(dst.S2e, s2e_clean_min, s2e_clean_max)
    mask_s1e   = in_range(dst.S1e, s1e_clean_min, s1e_clean_max)
    mask_s2w   = in_range(dst.S2w, s2w_clean_min, s2w_clean_max)
    mask_s2q   = in_range(dst.S2q, s2q_clean_min, s2q_clean_max)
    mask_nsipm = in_range(dst.Nsipm, nsipm_clean_min + 1, nsipm_clean_max + 1)
    mask_zdv   = in_range(dst.Z, zdv_clean_min, zdv_clean_max)
    mask_zmap  = in_range(dst.Z, zmap_clean_min, zmap_clean_max)

    dst_dv     = dst[(mask_r) & (mask_s2e) & (mask_s1e) & (mask_s2w) & (mask_s2q) & (mask_nsipm) & (mask_zdv)]
    dst_map    = dst[(mask_r) & (mask_s2e) & (mask_s1e) & (mask_s2w) & (mask_s2q) & (mask_nsipm) & (mask_zmap)]
    
    print('Data cleaning ended:\n\n     R     < {:.2f} mm\n     S2e   = [{:.2f}, {:.2f}] pes\n     S1e   = [{:.2f}, {:.2f}] pes\n     S2w   = [{:.2f}, {:.2f}] \u03BCs\n     S2q   = [{:.2f}, {:.2f}] pes\n     Nsipm = [{:.2f}, {:.2f}]\n     Z_map = [{:.2f}, {:.2f}] mm\n     Z_dv  = [{:.2f}, {:.2f}] mm\n'
          .format(np.max(dst_map.R), np.min(dst_map.S2e), np.max(dst_map.S2e), np.min(dst_map.S1e), np.max(dst_map.S1e), np.min(dst_map.S2w), np.max(dst_map.S2w), np.min(dst_map.S2q), np.max(dst_map.S2q), np.min(dst_map.Nsipm), np.max(dst_map.Nsipm), np.min(dst_map.Z), np.max(dst_map.Z), np.min(dst_dv.Z), np.max(dst_dv.Z)))
  
    
    return dst_dv, dst_map


def dst_correction_with_map(dst, maps):
    """
    Input a dst and a map
    Returns the correction factor
    """
    
    geom_corr = e0_xy_correction(maps)

    total_correction = apply_all_correction(maps, apply_temp = True)

    corr_geo = geom_corr(dst.X, dst.Y)
    
    corr_tot = total_correction(dst.X, dst.Y, dst.Z, dst.time)
    
    return corr_tot
