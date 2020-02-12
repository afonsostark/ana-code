import os
import sys
import time
import pandas as pd
from argparse_configuration import get_parser
import pandas as pd

from manage_data          import load_data, create_dirs, s1s2_selection
from manage_data          import radial_selection, energy_selection
from control_plots        import s1_1D_control_plots, s2_1D_control_plots
from detector_properties  import drift_velocity, energy_resolution
from detector_corrections import apply_corrections

print("Last updated on ", time.asctime())

def main(args = None):
    print('-------- Hi there! Let\'s look at control plots,\
    energy resolution, drift velocity and lifetime!')
    print("Last updated on ", time.asctime())

    # to do: comprobar que el config existe

    args = get_parser()
    opt_dict = vars(args)

    dir_input = opt_dict["dir_in"]
    run       = opt_dict["run"]
    rfid      = opt_dict["rfid"]

    #rfid      = opt_dict["rfid"]

    dst_out_dir = opt_dict['dir_out'] + '/'+ opt_dict["run"] +'/kdst-reduced'
    plots_dir   = opt_dict['dir_out'] + '/'+ opt_dict["run"] + '/plots'

    create_dirs(plots_dir)
    create_dirs(dst_out_dir)

    # To do pass by config file the name of the file
    #fout_name = plots_dir+'summary.txt'
    fout_name = plots_dir+'energy.txt'

    fout = open(fout_name,'w')
    fout.write(f"----------  Summary of run {run}  ----------\n\n")

    #dst_full = load_data(fout, dir_input, run)
    #dst_s1s2 = s1s2_selection(dst_full,          fout, dst_out_dir, run, save=True)
    #dst_r    = radial_selection(dst_s1s2, rfid,  fout, dst_out_dir, run, save=True)
    #dst_e    = energy_selection(dst_r, opt_dict, fout, dst_out_dir, run, save=True)

    file_in = opt_dict["file_in"]

    dst = pd.read_hdf(file_in)


    #s1_1D_control_plots(plots_dir, dst_full, opt_dict, 'all')
    #s2_1D_control_plots(plots_dir, dst_full, opt_dict, 'all')
    #s1_1D_control_plots(plots_dir, dst_s1s2, opt_dict, 's1s2')
    #s2_1D_control_plots(plots_dir, dst_s1s2, opt_dict, 's1s2')
    #s1_1D_control_plots(plots_dir, dst_r, opt_dict, f'r_{rfid}')
    #s2_1D_control_plots(plots_dir, dst_r, opt_dict, f'r_{rfid}')

    drift_velocity(plots_dir, dst, 'test')
    #energy_resolution(plots_dir, dst, 'test')



    print(f'closing output summary file in: {fout_name}')
    fout.close()


if __name__ == "__main__":
        main()
