# Hi! Here are a few Python functions that I wrote to best examine the Exoplanet Archive's list of exoplanetary systems.

import numpy as np
import pandas as pd 


# This function will make a data frame of only the desired parameters and optionally removes planets with missing data 
# Note: the order the parameters are inserted determines the order that they sit in the final array
# Note: we could still add functionality for the data to only store planets that are within a certain range of values
# Note: we might be able to simplify this code as .loc and dropnan can be combined helpfully, see for example: https://natashabatalha.github.io/picaso/notebooks/workshops/ESO2021/ESO_Tutorial.html
def makeframe(desired_parameters, full_data,ranges=False,drop_nans=True):
    # To create this dataframe, we will use the dict() type constructor to set up a series of keys paired to arrays inside a long list
    # first, we define the list we will add all of our key value pairs to
    all_pairs = []
    # In the loop, we add the values for all the system parameters we want 
    for parameter in desired_parameters:
        parameter_column = full_data['{}'.format(parameter)].to_numpy()
        dictionary_pair = (parameter, parameter_column.T)
        all_pairs.append(dictionary_pair)
    # finally, we then make a dataframe of the dictionary of our large list
    desired_data = pd.DataFrame(dict(all_pairs))
    
    # Finally, we allow for ranges within the data
    # These ranges have to be carefully specified using the following format: [[parameter1, parameter1_lower bound, parameter1_upperbound],[parameter2, parameter2_lower bound, parameter2_upperbound],...]
    if ranges !=False:
        all_not_allowed = []
        for parameter_range in ranges:
            parameter_to_restrict = parameter_range[0]
            lower_bound = parameter_range[1]
            upper_bound = parameter_range[2]

            truth_lower_bound = (lower_bound <= desired_data[parameter_to_restrict]).to_numpy()
            truth_upper_bound = (desired_data[parameter_to_restrict] <= upper_bound).to_numpy()
            for idx in range(len(truth_lower_bound)):
                if truth_lower_bound[idx]==False:
                    all_not_allowed.append(idx)
            for idx in range(len(truth_upper_bound)):
                if truth_upper_bound[idx]==False:
                    all_not_allowed.append(idx)
        all_not_allowed = np.sort(np.unique(np.array(all_not_allowed)))
        desired_data = desired_data.drop(all_not_allowed)
            
    # We have implemented the capacity to remove rows from this dataframe where nans are present. 
    if drop_nans == True:
        desired_data = desired_data.dropna(axis=0)
    
    return desired_data

# This function can be employed to return a specified parameter from any planet system by using all_systems_all_parameters above
# At the moment, this function pulls a value for a specific parameter for a specific planet. Thus, if we want all of one parameter, it's better to rerun makeframe with that parameter included.
# This may yield to multiple rows if the dataset is the all observations data not the comparative parameters dataset.
def retrieve(planet_name,desired_parameter,full_data):
    # get the row of the planet_name, in other words all the system info
    # this is a tricky line that required I look up a lot of documentation
    # Using the query method, we look for values in the frame where pl_name == planet name and save the row. I found out about this method from: https://saturncloud.io/blog/how-to-get-the-index-of-a-row-in-a-pandas-dataframe-as-an-integer/#:~:text=You%20can%20use%20the%20.,value%20in%20the%20Name%20column.
    # Using the {} and string format method, we can stick our variable name into the string properly. I learned about this technique from: https://realpython.com/python-f-strings/
    system_row = full_data.query("pl_name == '{}'".format(planet_name)).index[0] #--adding this last bit saves the row # instead of the row. 
    return full_data.loc[system_row]['{}'.format(desired_parameter)]

def loc_retrieve(planet_name,desired_parameter,full_data):
    # get the row of the planet_name, in other words all the system info
    # this is a tricky line that required I look up a lot of documentation
    # Using the query method, we look for values in the frame where pl_name == planet name and save the row. I found out about this method from: https://saturncloud.io/blog/how-to-get-the-index-of-a-row-in-a-pandas-dataframe-as-an-integer/#:~:text=You%20can%20use%20the%20.,value%20in%20the%20Name%20column.
    # Using the {} and string format method, we can stick our variable name into the string properly. I learned about this technique from: https://realpython.com/python-f-strings/
    system_rows = full_data.query("pl_name == '{}'".format(planet_name)).index #--adding this last bit saves the row # instead of the row. 
    param_values = []
    for idx in system_rows:
        param_values.append(full_data.loc[idx]['{}'.format(desired_parameter)])
    return param_values

def system_parameters(sys_data, hostname):
    mask = sys_data["hostname"].isin([hostname]) # learned about isin from: https://saturncloud.io/blog/how-to-select-rows-from-a-dataframe-based-on-list-values-in-a-column-in-pandas/#:~:text=To%20select%20rows%20from%20a%20DataFrame%20based%20on%20a%20list,to%20select%20the%20desired%20rows.
    parameters = sys_data[mask].to_numpy()
    return parameters

# Sometimes, we have systems where one planet has defined parameter values and other planets do not. 
# This function drops those systems through masking. 
# To find the complete systems, we need to specify where the "sy_pnum" is in the array of parameters inputted into makeframe.
# If you don't want to specify it manually, you can use the following code:
#params_to_study = ["pl_name", "hostname", "sy_pnum", "pl_rade", "st_mass"]
#for param_idx in range(len(params_to_study)):
#    if params_to_study[param_idx] == "sy_pnum":
#        pnum_location = param_idx
#        break
def drop_partial_systems(sys_data, pnum_location):
    mask = []
    for idx in range(len(sys_data)):
        planet = sys_data.iloc[idx]
        details = system_parameters(sys_data, planet["hostname"])
        if len(details) == 0:
            mask.append(False)
        else:
            actual_planet_number = details[0][pnum_location] # planet number is the same for all planets in the system, so we could choose any value of the first index. 
            planet_number_with_parameters = len(details)
        if actual_planet_number==planet_number_with_parameters:
            mask.append(True)   
        elif actual_planet_number!=planet_number_with_parameters:
            mask.append(False)
    return mask

# Similar to the above function except that we keep partial systems with equal to or more than the minimum number of planets.
# In other words, systems where not all planets have measurements are considered a new system with only the planets with measurements. 
# we also save an array of updated counts of all the planets in a system with the appropriate parameters for those systems that fit the criteria.
def drop_systems_below_pnum(sys_data, pnum_location, planet_count_min = 3):
    mask = []
    new_planet_count = []
    for idx in range(len(sys_data)):
        planet = sys_data.iloc[idx]
        details = system_parameters(sys_data, planet["hostname"])
        if len(details) == 0:
            mask.append(False)
        else:
            actual_planet_number = details[0][pnum_location] # planet number is the same for all planets in the system, so we could choose any value of the first index. 
            planet_number_with_parameters = len(details)
        if actual_planet_number==planet_number_with_parameters:
            mask.append(True)
            new_planet_count.append(actual_planet_number)   
        elif actual_planet_number!=planet_number_with_parameters & planet_number_with_parameters>=planet_count_min:
            mask.append(True)
            new_planet_count.append(planet_number_with_parameters)
        elif actual_planet_number!=planet_number_with_parameters & planet_number_with_parameters<planet_count_min:
            mask.append(False)
    assert len(mask)==len(sys_data)
    return mask, new_planet_count
    
