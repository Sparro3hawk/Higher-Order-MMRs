# Part of Population_Synthesis_Analysis at the following repository: https://github.com/Sparro3hawk/Higher-Order-MMRs.git
# Written by Finnegan Keller 05/28/25
import numpy as np
import pandas as pd
import fractions

# Some functions to add additional information to the spreadsheet of all simulations.
def get_initial_periods_and_ratios(dataframe, period = False, ratio = False):
    # to get the initial period ratios, we need to compute the initial periods from the initial semi-major axes.
    dataframe_pl_initial_smaxes = np.transpose(np.vstack([dataframe["pl_orbsmax_0"].to_numpy(), dataframe["pl_orbsmax_1"].to_numpy(), dataframe["pl_orbsmax_2"].to_numpy(), dataframe["pl_orbsmax_3"].to_numpy(), dataframe["pl_orbsmax_4"].to_numpy(), dataframe["pl_orbsmax_5"].to_numpy(), dataframe["pl_orbsmax_6"].to_numpy()]))
    dataframe_pl_masses = np.transpose(np.vstack([dataframe["pl_mass_0"].to_numpy(), dataframe["pl_mass_1"].to_numpy(), dataframe["pl_mass_2"].to_numpy(), dataframe["pl_mass_3"].to_numpy(), dataframe["pl_mass_4"].to_numpy(), dataframe["pl_mass_5"].to_numpy(), dataframe["pl_mass_6"].to_numpy()]))
    dataframe_st_masses = dataframe['st_mass'].to_numpy()
    dataframe_pnums = dataframe["pnum"].to_numpy()
    dataframe_pl_initial_periods = pd.DataFrame(columns = ['p_i_0', 'p_i_1', 'p_i_2', 'p_i_3', 'p_i_4', 'p_i_5', 'p_i_6']) 
    for row in range(len(dataframe_pl_initial_smaxes)): 
        st_mass = dataframe_st_masses[row]
        pnum = dataframe_pnums[row]
        sy_pl_masses = dataframe_pl_masses[row][0:pnum]
        sy_smaxes = dataframe_pl_initial_smaxes[row][0:pnum]
        sy_orbpers = np.sqrt((sy_smaxes**3)/(st_mass+sy_pl_masses))
        for i in range(7-pnum):
            sy_orbpers = np.append(sy_orbpers, np.nan)
        dataframe_pl_initial_periods.loc[row] = sy_orbpers

    if period == True:
        dataframe_pl_initial_periods.reset_index(inplace = True)
        return dataframe_pl_initial_periods
    
    elif ratio == True and period == False:
        dataframe_initial_period_ratios = pd.DataFrame(columns = ['p_i_01', 'p_i_12', 'p_i_23', 'p_i_34', 'p_i_45', 'p_i_56'])
        for idx in range(len(dataframe_pnums)):
            pnum = dataframe_pnums[idx]
            sy_periods = dataframe_pl_initial_periods.iloc[idx][:pnum]
            sy_period_ratios = np.array([])
            for p in range(pnum-1):
                sy_period_ratios = np.append(sy_period_ratios, sy_periods[p+1]/sy_periods[p])
            for p in range(7-pnum):
                sy_period_ratios = np.append(sy_period_ratios, np.nan)
            dataframe_initial_period_ratios.loc[idx] = sy_period_ratios
        dataframe_initial_period_ratios.reset_index(inplace = True)
        return dataframe_initial_period_ratios
    
    elif period == True and ratio == True:
        dataframe_initial_period_ratios = pd.DataFrame(columns = ['p_i_01', 'p_i_12', 'p_i_23', 'p_i_34', 'p_i_45', 'p_i_56'])
        for idx in range(len(dataframe_pnums)):
            pnum = dataframe_pnums[idx]
            sy_periods = dataframe_pl_initial_periods.iloc[idx][:pnum]
            sy_period_ratios = np.array([])
            for p in range(pnum-1):
                sy_period_ratios = np.append(sy_period_ratios, sy_periods[p+1]/sy_periods[p])
            for p in range(7-pnum):
                sy_period_ratios = np.append(sy_period_ratios, np.nan)
            dataframe_initial_period_ratios.loc[idx] = sy_period_ratios
        dataframe_pl_initial_periods.reset_index(inplace = True)
        dataframe_initial_period_ratios.reset_index(inplace = True)
        return dataframe_pl_initial_periods, dataframe_initial_period_ratios
    elif period == False and ratio == False:
        dataframe_initial_period_ratios = pd.DataFrame(columns = ['p_i_01', 'p_i_12', 'p_i_23', 'p_i_34', 'p_i_45', 'p_i_56'])
        for idx in range(len(dataframe_pnums)):
            pnum = dataframe_pnums[idx]
            sy_periods = dataframe_pl_initial_periods.iloc[idx][:pnum]
            sy_period_ratios = np.array([])
            for p in range(pnum-1):
                try:
                    sy_period_ratios = np.append(sy_period_ratios, sy_periods[p+1]/sy_periods[p])
                except RuntimeWarning:
                    print(sy_periods[p+1])
                    print(sy_periods[p])
                    print(sy_periods[p+1]/sy_periods[p])
                    sy_period_ratios = np.append(sy_period_ratios, sy_periods[p+1]/sy_periods[p])
            for p in range(7-pnum):
                sy_period_ratios = np.append(sy_period_ratios, np.nan)
            dataframe_initial_period_ratios.loc[idx] = sy_period_ratios   
        dataframe_pl_initial_periods.reset_index(inplace = True)
        dataframe_initial_period_ratios.reset_index(inplace = True) 
        # Let's make a copy of the input dataframe with these new rows.
        #mod_dataframe = dataframe.join(dataframe_pl_initial_periods, on = 'level_0')
        mod_dataframe = pd.concat([dataframe, dataframe_pl_initial_periods], axis = 1)
        mod_dataframe = pd.concat([dataframe, dataframe_initial_period_ratios], axis = 1)
        return mod_dataframe
    
def get_mass_ratios(dataframe, ratio = False): 
    dataframe_pl_masses = np.transpose(np.vstack([dataframe["pl_mass_0"].to_numpy(), dataframe["pl_mass_1"].to_numpy(), dataframe["pl_mass_2"].to_numpy(), dataframe["pl_mass_3"].to_numpy(), dataframe["pl_mass_4"].to_numpy(), dataframe["pl_mass_5"].to_numpy(), dataframe["pl_mass_6"].to_numpy()]))
    dataframe_pnums = dataframe["pnum"].to_numpy()
    dataframe_mass_ratios = pd.DataFrame(columns = ['m_01', 'm_12', 'm_23', 'm_34', 'm_45', 'm_56'])
    for idx in range(len(dataframe_pnums)):
        pnum = dataframe_pnums[idx]
        sy_mass_ratios = np.array([])
        for p in range(pnum - 1):
            sy_mass_ratios = np.append(sy_mass_ratios, dataframe_pl_masses[idx][p+1]/dataframe_pl_masses[idx][p])
        for p in range(7-pnum): 
            sy_mass_ratios = np.append(sy_mass_ratios, np.nan)
        dataframe_mass_ratios.loc[idx] = sy_mass_ratios
    if ratio == True:
        return dataframe_mass_ratios
    elif ratio == False:
        dataframe_mass_ratios.reset_index(inplace = True)
        mod_dataframe = pd.concat([dataframe, dataframe_mass_ratios], axis = 1)
        return mod_dataframe
    
# Initially we used Delta = P_out/P_in - p/q to define resonance. We have decided to update this condition to (P_out/P_in)/(p/q) - 1.
def get_newly_defined_deltas(dataframe, ratio = False): 
    dataframe_period_ratios = np.transpose(np.vstack([dataframe["p_ratio_01"].to_numpy(), dataframe["p_ratio_12"].to_numpy(), dataframe["p_ratio_23"].to_numpy(), dataframe["p_ratio_34"].to_numpy(), dataframe["p_ratio_45"].to_numpy(), dataframe["p_ratio_56"].to_numpy()]))
    dataframe_proximal_resonance = np.transpose(np.vstack([dataframe["prox_res_01"].to_numpy(), dataframe["prox_res_12"].to_numpy(), dataframe["prox_res_23"].to_numpy(), dataframe["prox_res_34"].to_numpy(), dataframe["prox_res_45"].to_numpy(), dataframe["prox_res_56"].to_numpy()]))
    dataframe_pnums = dataframe["pnum"].to_numpy()
    dataframe_new_deltas = pd.DataFrame(columns = ['delta_01', 'delta_12', 'delta_23', 'delta_34', 'delta_45', 'delta_56'])
    for idx in range(len(dataframe_pnums)):
        sy_deltas = np.array([])
        if np.isnan(dataframe_proximal_resonance[idx]).all() == True:
            for ppair in range(6): 
                sy_deltas = np.append(sy_deltas, np.nan)
        else:
            pnum = dataframe_pnums[idx]
            for ppair in range(pnum - 1):
                period_ratio = dataframe_period_ratios[idx][ppair]
                proximal_resonance = dataframe_proximal_resonance[idx][ppair]
                proximal_resonance_fraction = fractions.Fraction(proximal_resonance).limit_denominator(1000)
                p = proximal_resonance_fraction.numerator
                q = proximal_resonance_fraction.denominator
                sy_deltas = np.append(sy_deltas, period_ratio/(p/q) - 1)
            for ppair in range(7-pnum): 
                sy_deltas = np.append(sy_deltas, np.nan)
        dataframe_new_deltas.loc[idx] = sy_deltas
    if ratio == True:
        return dataframe_new_deltas
    elif ratio == False:
        dataframe_new_deltas.reset_index(inplace = True)
        mod_dataframe = pd.concat([dataframe, dataframe_new_deltas], axis = 1)
        return mod_dataframe
    
# Defines the type of resonant chain we have.
def resonant_chain_classification(dataframe, max_libration_amp = 90):
    predicted_2body_amplitudes = pd.DataFrame(np.transpose([dataframe["phi_amp_2_01"].to_numpy(), dataframe["phi_amp_2_12"].to_numpy(), dataframe["phi_amp_2_23"].to_numpy(), dataframe["phi_amp_2_34"].to_numpy(), dataframe["phi_amp_2_45"].to_numpy(), dataframe["phi_amp_2_56"].to_numpy()]), columns = ["phi_amp_2_01", "phi_amp_2_12", "phi_amp_2_23", "phi_amp_2_34", "phi_amp_2_45", "phi_amp_2_56"])
    predicted_3body_amplitudes = pd.DataFrame(np.transpose([dataframe["phi_amp_3_012"].to_numpy(), dataframe["phi_amp_3_123"].to_numpy(), dataframe["phi_amp_3_234"].to_numpy(), dataframe["phi_amp_3_345"].to_numpy(), dataframe["phi_amp_3_456"].to_numpy()]), columns = ["phi_amp_3_012", "phi_amp_3_123", "phi_amp_3_234", "phi_amp_3_345", "phi_amp_3_456"])
    all_three_body_librate_mask = np.array([], dtype = 'bool')
    # Let's see what chains have all three-body MMRs librating, some librating, and none librating.
    for row in range(len(dataframe)):
        pnum = dataframe["pnum"].iloc[row]
        predicted_resonant_amplitudes_of_row = predicted_3body_amplitudes.iloc[row].to_numpy()[0:pnum-2] # remove NaNs at the end of systems with fewer than the maximum planet count
        truth_table_of_row = predicted_resonant_amplitudes_of_row <= max_libration_amp #(max_libration_amp*u.deg).to(u.rad)
        # all resonant case
        if np.all(truth_table_of_row) == True:
            all_three_body_librate_mask = np.append(all_three_body_librate_mask, True)
        else:
            all_three_body_librate_mask = np.append(all_three_body_librate_mask, False)

    # Now, let's see which of the ones where all three bodies librate also have all librating two bodies
    all_two_body_librate_mask = np.array([], dtype = 'bool')
    no_two_body_librate_mask = np.array([], dtype = 'bool')
    # Let's see what chains have all two-body MMRs librating, some librating, and none librating.
    for row in range(len(dataframe)):
        pnum = dataframe["pnum"].iloc[row]
        predicted_resonant_amplitudes_of_row = predicted_2body_amplitudes.iloc[row].to_numpy()[0:pnum-1] # remove NaNs at the end of systems with fewer than the maximum planet count
        truth_table_of_row = predicted_resonant_amplitudes_of_row <= max_libration_amp #(max_libration_amp*u.deg).to(u.rad)
        # all resonant case
        if np.all(truth_table_of_row) == True:
            all_two_body_librate_mask = np.append(all_two_body_librate_mask, True)
        else:
            all_two_body_librate_mask = np.append(all_two_body_librate_mask, False)
        # no resonant case
        if np.all(truth_table_of_row) == False and np.any(truth_table_of_row) == False:
            no_two_body_librate_mask = np.append(no_two_body_librate_mask, True)
        else:
            no_two_body_librate_mask = np.append(no_two_body_librate_mask, False)

     # Separately, it would also be cool to collect the systems that feature at least one triplet with a librating three-body resonance where both two-body resonances do not librate. These are so called "pure" three-body resonances.
    pure_three_body_mask = np.array([], dtype = 'bool')
    for row in range(len(dataframe)):
        pnum = dataframe["pnum"].iloc[row]
        three_body_predicted_resonant_amplitudes_of_row = predicted_3body_amplitudes.iloc[row].to_numpy()[0:pnum-2] # remove NaNs at the end of systems with fewer than the maximum planet count
        three_body_truth_table_of_row = three_body_predicted_resonant_amplitudes_of_row <= max_libration_amp #(max_libration_amp*u.deg).to(u.rad)
        two_body_predicted_resonant_amplitudes_of_row = predicted_2body_amplitudes.iloc[row].to_numpy()[0:pnum-1] # remove NaNs at the end of systems with fewer than the maximum planet count
        two_body_truth_table_of_row = two_body_predicted_resonant_amplitudes_of_row <= max_libration_amp #(max_libration_amp*u.deg).to(u.rad)
        p3b_indicator = False
        for idx in range(len(three_body_truth_table_of_row)):
            if ((three_body_truth_table_of_row[idx] == True) and (two_body_truth_table_of_row[idx] == False) and (two_body_truth_table_of_row[idx+1] == False)) == True:
                p3b_indicator = True
                break
        pure_three_body_mask = np.append(pure_three_body_mask, p3b_indicator)
    
    # Let's get our combined masks for three-body resonant chains (all two and three body angles librate), two-body resonant chains (all two-body angles librate but not all three-body angles librate), and partial resonant chains (some two-body resonant angles librate)
    three_body_resonant_chain_mask = (all_three_body_librate_mask) & (all_two_body_librate_mask)
    two_body_resonant_chain_mask = (~all_three_body_librate_mask) & (all_two_body_librate_mask)
    partial_resonant_chain_mask = (~all_two_body_librate_mask) & (~no_two_body_librate_mask)
    # Let's apply the masks
    three_body_resonant_chains = dataframe[three_body_resonant_chain_mask]
    two_body_resonant_chains = dataframe[two_body_resonant_chain_mask]
    partial_resonant_chains = dataframe[partial_resonant_chain_mask]
    no_two_body_resonances = dataframe[no_two_body_librate_mask]
    # Let's also return our separate pure three-body case
    systems_with_pure_three_body = dataframe[pure_three_body_mask]
    return three_body_resonant_chains, two_body_resonant_chains, partial_resonant_chains, no_two_body_resonances, systems_with_pure_three_body    

# Find name of plot files and .bin file. Note that the plot files have the type of plot (e.g. a_ratio) after the last underscore while the .bin have the index again after the last underscore.
def plot_finder(index, K_outermost, tau_a): # called plot_finder as initially used function just to identify output plots. Keeping same name for consistency with old code.
    version = str(index)+'_inner_edge_' +'{0:1.1e}'.format(K_outermost)+'_taua_' +'{0:1.1e}'.format(tau_a)+'_'
    return version

# Add file names to the dataframe. 
def get_file_names(dataframe, file_name_only = False):
    dataframe_file_names = pd.DataFrame(columns = ['File Name'])
    indexes_to_remove = []
    for idx in range(len(dataframe)):
        K_outermost = dataframe.iloc[idx]['K_outermost']
        taua = dataframe.iloc[idx]['taua']
        try: 
            try: 
                index = int(dataframe.iloc[idx]['index'][0]) # other indexes come from way I set up the arrays
            except IndexError:
                index = int(dataframe.iloc[idx]['index'])
        except ValueError: # Weirdly, some indexes do not save when do this process. 
            indexes_to_remove.append(idx)
            continue
        file_name = plot_finder(index, K_outermost, taua)
        dataframe_file_names.loc[idx] = file_name
    if file_name_only == True:
        return dataframe_file_names
    elif file_name_only == False:
        #dataframe_file_names.reset_index(inplace = True)
        dataframe = dataframe.drop(indexes_to_remove)
        mod_dataframe = pd.concat([dataframe, dataframe_file_names], axis = 1)
        return mod_dataframe, dataframe_file_names, indexes_to_remove
    
# Let's cleanup the Dataframes. They all have some columns that are unnecessary and are in the wrong order.
def dataframe_column_cleanup(dataframe):
    dataframe = dataframe.iloc[:,1:]
    dataframe = dataframe.drop(['level_0', 'parameter_method', 'tmp'], axis = 1)
    # They also have lots of repitions of the index column annoyingly. More frustratingly, they all have slightly different names (index.+repeted time)
    repeated_index_columns = []
    for column in dataframe.columns:
        if column[:6] == 'index.':
            repeated_index_columns.append(column)
    dataframe = dataframe.drop(repeated_index_columns, axis = 1)
    # Finally, I'd ideally want the directory column and the file name at the end of the dataframe, but, as it was easier to add directory before other appends were done, they are separated.
    # To rectify this issue, we need to reorder the columns.
    adjusted_column_order = dataframe.columns.drop(['Directory', 'File Name'])
    adjusted_column_order = adjusted_column_order.to_list()
    adjusted_column_order.append('Directory')
    adjusted_column_order.append('File Name')
    dataframe = dataframe.loc[:, adjusted_column_order]
    return dataframe