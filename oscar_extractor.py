# Part of Population_Synthesis_Analysis at the following repository: https://github.com/Sparro3hawk/Higher-Order-MMRs.git
# Written by Finnegan Keller 05/28/25
import numpy as np
import pandas as pd
import astropy.units as u
import os
os.chdir("/Users/finnkeller/Desktop/Keller, Finnegan University of Hawaii Insitute for Astronomy Summer Internship Formation of Exosolar Resonant Systems 2024/Population Synthesis/")
from oscar_extractor_setup_func import *

# The first step here is to setup a dataframe of only the systems run on Oscar and partition it by the different classifications of resonance.

# Folder names for runs on Oscar.
oscar_run_names = []
for data, subdirs, files in os.walk('CCV_Outputs/'):
    for i in subdirs:
        oscar_run_names.append("Migration Runs "+str(i[11:]))

# Let's extract only the Oscar runs. Also append the directories they come from.
sim_results = pd.DataFrame()
sim_sets_with_oscar_info = []
for data, subdirs, files in os.walk('results/'):
    for i in files:
        if i == ".DS_Store":
            continue
        if i[:-4] in oscar_run_names: # note: not all oscar runs appear to be saved on computer. One known issue is the first set of oscar runs does not have an associated date. I could look to more properly mask that in.
            sim_sets_with_oscar_info.append(i[:-4])
            filename=os.path.join(data, i)
            sim_results_local = pd.read_csv(filename, index_col = False)
            # Let's add a column filled with just the directory.
            dataframe_directory = pd.DataFrame(columns = ['Directory'])
            for j in range(len(sim_results_local)):
                dataframe_directory.loc[j] = i[:-4]
            sim_results_local = pd.concat([sim_results_local, dataframe_directory], axis = 1)            
            # Concatenate to the larger simulation results set.
            sim_results = pd.concat([sim_results, sim_results_local])
        else: 
            continue
print("Simulation Sets that we have required information for: "+str(sim_sets_with_oscar_info))

# A handful of systems were entered weirdly and are classified with a bunch of missing data. I just remove them because I can't do any analysis with them.
nan_mask = np.array([], dtype = 'bool')
for row in range(len(sim_results)):
    if sim_results['close_encounter_flag'].iloc[row]!=True and sim_results['close_encounter_flag'].iloc[row]!=False:
        nan_mask = np.append(nan_mask, False)
    else: 
        nan_mask = np.append(nan_mask, True)
sim_results = sim_results[nan_mask]

sim_results.reset_index(inplace = True)
sim_results = get_initial_periods_and_ratios(sim_results)
sim_results = get_mass_ratios(sim_results)
sim_results = get_newly_defined_deltas(sim_results)
print("Number of Simulations from Oscar: "+str(len(sim_results))) # may need more as might not be enough converged ones...could get far more if don't need outputs

# We have a few systems with very large negative deltas for a planet pair that ends in resonance. These are removed now.
readin_negative_delta_tauas = np.load('output_variables/negative_delta_tauas.npy')
readin_negative_delta_K_outermosts = np.load('output_variables/negative_delta_K_outermosts.npy')
relevant_idxs = np.array([])
readin_overall_counter = 0
# Let's figure out which indexes they correspond to
for i in range(len(readin_negative_delta_tauas)):
    counter = 0
    taua = round(readin_negative_delta_tauas[i],8)
    K_outermost = round(readin_negative_delta_K_outermosts[i],8)
    for idx in range(len(sim_results)):
        if taua == round(sim_results.iloc[idx]['taua'], 8) and K_outermost == round(sim_results.iloc[idx]['K_outermost'], 8):
            if counter == 1: 
                print("Multiple systems have the identified taua and K factor values. Pull more conditions to isolate the system.")
            relevant_idxs = np.append(relevant_idxs, idx)
            counter+=1
    readin_overall_counter+=1
assert readin_overall_counter == len(readin_negative_delta_tauas)

# Let's mask out those indexes.
readin_mask = np.array([], dtype = 'bool')
for idx in range(len(sim_results)):
    if np.isin(idx, relevant_idxs):
        readin_mask = np.append(readin_mask, False)
    else:
        readin_mask = np.append(readin_mask, True)
sim_results_stage0 = sim_results[readin_mask]
neg_delta_sims = sim_results[~readin_mask]
# And return results.
print("Number of Systems with Resonant Pairs with Large Negative Deltas: "+str(len(neg_delta_sims)))
print("Number of Remaining Systems: "+str(len(sim_results_stage0)))
assert len(sim_results) == len(neg_delta_sims)+len(sim_results_stage0)

# It is worth noting that a small selection of integrations are sucessful but are flagged by rebound as having a convergence issue. The observable sign of these systems is a planet period less than 1.
# Weirdly, we do not see these getting flagged when we separate by encounter flag first even though these systems have negative close encounter flags.
# Now, fixing our close encounter threshold, we get no so called convergence errors. 
convergence_errors = np.array([], dtype = 'bool')
for row in range(len(sim_results_stage0)):
    p_ratios = np.hstack([sim_results_stage0["p_ratio_01"].iloc[row], sim_results_stage0["p_ratio_12"].iloc[row], sim_results_stage0["p_ratio_23"].iloc[row], sim_results_stage0["p_ratio_34"].iloc[row], sim_results_stage0["p_ratio_45"].iloc[row], sim_results_stage0["p_ratio_56"].iloc[row]])
    pnums = sim_results_stage0['pnum'].iloc[row]
    p_ratios = p_ratios[0:pnums-1]
    if np.any(p_ratios<1):
        convergence_errors = np.append(convergence_errors, False)
    else:
        convergence_errors = np.append(convergence_errors, True)
conv_results = sim_results_stage0[~convergence_errors]
print("Number of Systems That Had Convergence Errors: "+str(len(conv_results)))
sim_results_stage1 = sim_results_stage0[convergence_errors]
print("Number of Remaining Systems: "+str(len(sim_results_stage1)))
assert len(sim_results) == len(neg_delta_sims)+len(conv_results)+len(sim_results_stage1)

# We have a few systems that have masses greater than 30Mearth due to an improper while loop that did not actually resample mass if the masses were too high. Thus, we remove them. 
mass_flag = np.array([], dtype = 'bool')
for idx in range(len(sim_results_stage1)):
    pl_masses = np.hstack([sim_results_stage1['pl_mass_0'].iloc[idx], sim_results_stage1['pl_mass_1'].iloc[idx], sim_results_stage1['pl_mass_2'].iloc[idx], sim_results_stage1['pl_mass_3'].iloc[idx], sim_results_stage1['pl_mass_4'].iloc[idx], sim_results_stage1['pl_mass_5'].iloc[idx], sim_results_stage1['pl_mass_6'].iloc[idx]])[:int(sim_results_stage1['pnum'].iloc[idx])]
    if (pl_masses>30*u.Mearth.to(u.Msun)).any() == True:
        mass_flag = np.append(mass_flag, True)
    else:
        mass_flag = np.append(mass_flag, False)
large_mass_sims = sim_results_stage1[mass_flag]
sim_results_stage2 = sim_results_stage1[~mass_flag]
print("Number of Systems with Masses Larger than 30Mearth: "+str(len(large_mass_sims)))
print("Number of Remaining Systems: "+str(len(sim_results_stage2)))
assert len(sim_results) == len(neg_delta_sims)+len(conv_results)+len(large_mass_sims)+len(sim_results_stage2)

# Removing systems that had close encounters is easy, we just use the flag in the CSVs.
failed_sims = sim_results_stage2[sim_results_stage2["close_encounter_flag"]==True] 
sucessful_sims = sim_results_stage2[sim_results_stage2["close_encounter_flag"]==False]
print("Number of Systems Terminated Early Due To Close Encounters: "+str(len(failed_sims)))
print("Number of Systems that Sucessfully Integrated: "+str(len(sucessful_sims)))
all_sims = pd.concat([sucessful_sims, failed_sims])
assert len(all_sims) == len(sim_results_stage2)
assert len(sim_results) == len(neg_delta_sims)+len(large_mass_sims)+len(conv_results)+len(sucessful_sims)+len(failed_sims)

# Divide up by type of resonant chain.
three_body_resonant_chains, two_body_resonant_chains, partial_resonant_chains, no_two_body_resonances, systems_with_pure_three_body = resonant_chain_classification(sucessful_sims, 90)
# make sure I'm not double counting or missing any systems
assert len(three_body_resonant_chains)+len(two_body_resonant_chains)+len(partial_resonant_chains)+len(no_two_body_resonances) == len(sucessful_sims)
print("Number of Systems in Three-Body Resonant Chains: "+str(len(three_body_resonant_chains)))
print("Number of Systems in Two-Body Resonant Chains: "+str(len(two_body_resonant_chains)))
print("Number of Systems with Some Two-Body Resonant Pairs: "+str(len(partial_resonant_chains)))
# Save intermediate result. Helps with formatting for next step.
three_body_resonant_chains.to_csv("output_variables/intermediate_result_three_body_resonant_chains.csv")
two_body_resonant_chains.to_csv("output_variables/intermediate_result_two_body_resonant_chains.csv")
partial_resonant_chains.to_csv("output_variables/intermediate_result_partial_resonant_chains.csv")

# Get a list of file names for all of the systems. A small set of systems are saved without index names. Again, I could probably fix this at some point.
three_body_resonant_chains = pd.read_csv('output_variables/intermediate_result_three_body_resonant_chains.csv', index_col = False)
two_body_resonant_chains = pd.read_csv('output_variables/intermediate_result_two_body_resonant_chains.csv', index_col = False)
partial_resonant_chains = pd.read_csv('output_variables/intermediate_result_partial_resonant_chains.csv', index_col = False)

three_body_resonant_chains, three_body_resonant_chains_files, three_body_resonant_chains_removed_indexes = get_file_names(three_body_resonant_chains)
two_body_resonant_chains, two_body_resonant_chains_files, two_body_resonant_chains_removed_indexes = get_file_names(two_body_resonant_chains)
partial_resonant_chains, partial_resonant_chains_files, partial_resonant_chains_removed_indexes = get_file_names(partial_resonant_chains)
print("Number of Systems in Three-Body Resonant Chains with File Names: "+str(len(three_body_resonant_chains)))
print("Number of Systems in Two-Body Resonant Chains with File Names: "+str(len(two_body_resonant_chains)))
print("Number of Systems with Some Two-Body Resonant Pairs with File Names: "+str(len(partial_resonant_chains)))

# Let's cleanup the Dataframes. They all have some columns that are unnecessary and are in the wrong order.
three_body_resonant_chains = dataframe_column_cleanup(three_body_resonant_chains)
two_body_resonant_chains = dataframe_column_cleanup(two_body_resonant_chains)
partial_resonant_chains = dataframe_column_cleanup(partial_resonant_chains)
three_body_resonant_chains.to_csv("output_variables/oscar_three_body_resonant_chains.csv", index = False)
print("Oscar Three-Body Resonant Chains CSV saved.")
two_body_resonant_chains.to_csv("output_variables/oscar_two_body_resonant_chains.csv", index = False)
print("Oscar Two-Body Resonant Chains CSV saved.")
partial_resonant_chains.to_csv("output_variables/oscar_partial_resonant_chains.csv", index = False)
print("Oscar Partial Resonant Chains CSV saved.")
