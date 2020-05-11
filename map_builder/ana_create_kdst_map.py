from manage_data               import load_dst_for_map, dst_cleaning


def ana_create_kdst_map(fout, dst_out_dir, plots_dir, dir_input, run, opt_dict):
    """
    """
    rmax = int(opt_dict['rmax'])
    rfid = int(opt_dict['rfid'])

    #fout_name = plots_dir+'summary.txt'
    #fout = open(fout_name,'w')
    #fout.write(f"----------  Summary of run {run}  ----------\n")
    
    dst = load_dst_for_map(fout, dir_input, run)

    dst_dv, dst_map = dst_cleaning(dst, opt_dict, fout, dst_out_dir, run, save=False)

    #print(f'-----> Closing output summary file in: {fout_name}\n')
    #fout.close()
    return dst_dv, dst_map