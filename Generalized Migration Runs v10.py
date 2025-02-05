#Stage 0: Package Imports
# 0.0: import standard packages
import fractions
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] == True
import pandas as pd
from itertools import combinations
import astropy.units as u
import astropy.constants as const
from scipy.stats import lognorm
from collections import OrderedDict
import os

# 0.1: set working directory. Must do before importing mr_forecast and Exoplanet_Archive Examiner
os.chdir("/Users/finnkeller/Desktop/Keller, Finnegan University of Hawaii Insitute for Astronomy Summer Internship Formation of Exosolar Resonant Systems 2024/Population Synthesis/")

# 0.2: import migration codes, Exoplanet archive homebrew functions, and mass radius code
import rebound # nbody code
import reboundx # extra code for migration
import celmech.disturbing_function as disturb # extra code for celestial mechanics (getting f and g coefficients)
import mr_forecast as mr # code to predict masses from planetary radii measurements
from Exoplanet_Archive_Examiner import makeframe, drop_systems_below_pnum # functions to pull from the Exoplanet Archive

# Let's clean up this code for future use. Currently, I'm going to make 1 dataset. We could call this every time for variation or just once at reread it in.
# Stage 1: Data Imports and Radius/Mass Thresholding
# 1.0: Create radius dataset. We only want to run this if the radius dataset does not already exist.
def setup_radius_dataset(planet_count_min = 3, planet_count_max = 7):
    # 1.0.0: Import the Exoplanet Archive Planetary Systems Composite (PSC)
    PSC = pd.read_csv("Exoplanet Datasets/PSCompPars_2024.08.01_14.31.11.csv", header = 88)

    # 1.0.1: Gather all planets with stellar masses and radius measurements<4Rearth. We remove all systems with fewer than the minimum number of planets of interest.
    min_radius = 0.5
    max_radius = 4 #Mpost2R([30], unit='Earth')[0]
    potential_systems = makeframe(["pl_name", "hostname", "sy_pnum", "pl_rade", "st_mass"], PSC, ranges = [["pl_rade", min_radius, max_radius], ["sy_pnum", planet_count_min, 8]])

    # 1.0.2: Keep systems where more than 3 planets have radius measurements under 4Rearth.
    # Note: It was important above not to cut our maximum planet count at 7 as we do add an 8 planet system with this line where 6 planets have radius measurements under 4Rearth.
    mask, updated_planet_counts = drop_systems_below_pnum(potential_systems, pnum_location = 2, planet_count_min = planet_count_min)  
    potential_systems = potential_systems[mask]

    # 1.0.3: Adjust planet counts to match the number of planets with radii measurements instead of the number of planets in the real system.
    potential_systems = potential_systems.drop(labels = "sy_pnum", axis = "columns")
    potential_systems = potential_systems.assign(sy_pnum = updated_planet_counts)

    # 1.0.4: Create a radius distribution to pull from to create a uniform input distribution of systems.
    # 1.0.4.0: Initialize radius distribution and find all potential systems by starname.
    radius_ratios = np.array([])
    stars = np.unique(potential_systems["hostname"])
    # 1.0.4.1: Create a distribution of radius ratios by finding the planet radii corresponding to a given system and computing the radius ratios between each planet pair.
    for system_idx in range(len(stars)):
        system = potential_systems[potential_systems["hostname"] == stars[system_idx]]
        radii = system["pl_rade"].to_numpy()
        planet_number = system["sy_pnum"].iloc[0]
        for planet_idx in range(planet_number-1):
            radius_ratios = np.append(radius_ratios, (radii[planet_idx+1]/radii[planet_idx]))
    # 1.0.4.2: Sort the radius ratios (not strictly necessary, but it does make the array pretty).
    radius_ratios = np.sort(radius_ratios)
    # 1.0.4.3: Fit a lognormal distribution to the radius ratios for later sampling from.
    pl_rade_loc, pl_rade_scale, pl_rade_shape = lognorm.fit(radius_ratios)

    # 1.0.5: Determine the maximum number of systems corresponding to a given planet multiplicity in the dataset. Run some assertion tests to double check it.
    if planet_count_min!=3 or planet_count_max!=7:
        print("Double check that the max number of systems for a given multiplicity is correct as the following are currently somewhat hardcoded for a planet count range of 3-7.")
    planet_range = np.arange(planet_count_min,planet_count_max+1)
    max_num_planets_for_pnum_pnum = planet_range[np.argmax(np.array([len(potential_systems[potential_systems['sy_pnum'] == 3]), len(potential_systems[potential_systems['sy_pnum'] == 4]), len(potential_systems[potential_systems['sy_pnum'] == 5]), len(potential_systems[potential_systems['sy_pnum'] == 6]), len(potential_systems[potential_systems['sy_pnum'] == 7])]))]
    assert max_num_planets_for_pnum_pnum == planet_count_min
    max_num_systems_for_pnum = int(np.max(np.array([len(potential_systems[potential_systems['sy_pnum'] == 3]), len(potential_systems[potential_systems['sy_pnum'] == 4]), len(potential_systems[potential_systems['sy_pnum'] == 5]), len(potential_systems[potential_systems['sy_pnum'] == 6]), len(potential_systems[potential_systems['sy_pnum'] == 7])]))/max_num_planets_for_pnum_pnum)
    assert np.max(np.array([len(potential_systems[potential_systems['sy_pnum'] == 3]), len(potential_systems[potential_systems['sy_pnum'] == 4]), len(potential_systems[potential_systems['sy_pnum'] == 5]), len(potential_systems[potential_systems['sy_pnum'] == 6]), len(potential_systems[potential_systems['sy_pnum'] == 7])])) % max_num_planets_for_pnum_pnum == 0
   
    # 1.0.6: For a given planet number above 3, draw from all planet numbers below 3 and append appropriately to create a number of systems equal to the number with 3 planets.
    # 1.0.6.0: Initialize the list, which will contain all of the simulated rows corresponding to new planets to be added to the dataframe. Initialize a counter to determine unique stellar names.
    new_rows = []
    star_name_counter = 0
    for i in range(planet_count_min+1,planet_count_max+1):
        # 1.0.6.1: Determine the number of known systems at a planet number by dividing the number of rows of the dataframe with that many planets by the number of planets.
        num_systems = int(len(potential_systems[potential_systems['sy_pnum'] == i])/i)
        # 1.0.6.2: As we want the number of systems to be an integer value, double check that it indeed is so.
        assert len(potential_systems[potential_systems['sy_pnum'] == i]) % i == 0
        # 1.0.6.3: Determine the number of systems we need to create to create a uniform input distribution.
        number_of_systems_to_generate = max_num_systems_for_pnum - num_systems
        # 1.0.6.4: Determine if we have enough systems to draw from to create a uniform input distribution that we do not need to repeat inputs for generation.
        if number_of_systems_to_generate <= len(potential_systems[potential_systems['sy_pnum'] < i]):
            replace_bool = False
        elif number_of_systems_to_generate > len(potential_systems[potential_systems['sy_pnum'] < i]):
            replace_bool = True

        # 1.0.6.5: Use the radius ratio distribution to create systems of length equal to i from randomly selected input systems of length less than i.
        for system_idx in range(number_of_systems_to_generate):
            # 1.0.6.5.0: Determine the system to copy and append to by drawing a random star.
            stars = np.unique(potential_systems[potential_systems["sy_pnum"] < i]["hostname"])
            random_star = np.random.choice(stars, replace = replace_bool)
            random_system = potential_systems[(potential_systems["hostname"]==random_star) & (potential_systems["sy_pnum"] < i)]
            # 1.0.6.5.1: Determine the radii of that system, how many radii we need to append to it, and the radius ratios we will use to do so.
            sy_radii = random_system["pl_rade"].to_numpy()
            number_of_planets_to_generate = i - len(sy_radii)
            random_r_ratio_lognormal = lognorm.rvs(pl_rade_loc, pl_rade_scale, pl_rade_shape, size = number_of_planets_to_generate)
            # 1.0.6.5.2. Add rows to the new rows dataset for each pre-existing radius measurement.
            for radius in sy_radii:
                new_rows.append([random_star+" simulated "+str(star_name_counter), random_star+" simulated "+str(star_name_counter), i, radius, np.median(random_system["st_mass"].to_numpy())])
            # 1.0.6.5.3: Use the radius ratios to add simulated rows to the new rows dataset. Threshold these based on radius limits.
            for planet_idx in range(number_of_planets_to_generate):
                # r_new = r_ratio * r_old
                radius = random_r_ratio_lognormal[planet_idx]*sy_radii[-1]
                # Note: Forecaster cannot solve for radii below 0.1 Rearth and we want a maximum radii of max_radius = 4. These radius ratios can inflate our masses and radii, so we set hard caps below.
                if radius>max_radius:
                    radius = max_radius
                elif radius<min_radius:
                    radius = min_radius
                sy_radii = np.append(sy_radii, radius)
                new_rows.append([random_star+" simulated "+str(planet_idx), random_star+" simulated "+str(star_name_counter), i, radius, np.median(random_system["st_mass"].to_numpy())])
            # 1.0.6.5.4: Modify the following counter so that the star name is unique for every simulated system but preserves the star system name it was drawn from.
            star_name_counter+=1

    # 1.0.7: Colate all of the new row into a dataframe and combine with our original dataframe.
    added_systems = pd.DataFrame(new_rows, columns = ["pl_name", "hostname", "sy_pnum", "pl_rade", "st_mass"])
    uniform_distribution = pd.concat([potential_systems, added_systems])
   
    # 1.0.8: Return dataframe.
    return uniform_distribution

# 1.1: Write this dataset to a CSV if it does not exist.
if os.path.exists("Exoplanet Datasets/Final Radius Dataset.csv")==False:
    # 1.1.0: Create the dataset,
    potential_systems = setup_radius_dataset(planet_count_min = 3)
    # 1.1.1: Write this dataset to a CSV.
    potential_systems.to_csv("Exoplanet Datasets/Final Radius Dataset.csv")
# 1.2: Open a CSV of this dataset if it has previously been run.
# Note: This is important as mr_forecaster will produce different radii every time the code is run, so it is useful to be consistent with our results.
elif os.path.exists("Exoplanet Datasets/Final Radius Dataset.csv")==True:
    # 1.2.0: Import pre-existing mass dataset.  
    potential_systems = pd.read_csv("Exoplanet Datasets/Final Radius Dataset.csv")

# Stage 2: Write functions to establish the simulation initial conditions.
# 2.0: Import Kepler period ratios from Fabrycky et al. (2014) https://iopscience.iop.org/article/10.1088/0004-637X/790/2/146#apj497647t1 table 1
kepler = pd.read_table("Exoplanet Datasets/Kepler Multiplanet Data.txt", header = 32, names = ["KOI", "Period", "T0", "Tdur", "Rp", "S/N", "M*", "R*", "P/P-", "Delta-"], sep=r"\s+")

# 2.1: Write a function to initialize a planetary system's (nonangular) parameters, planet number, planet mass, star mass, and planet semi-major axis.
def Keplerian_parameters(potential_systems = potential_systems, kepler = kepler, minimum_mass = 0.1, maximum_mass = 30, inner_disk_edge = 0.05, displacement_from_inner_disk_edge = 0.05, planet_count_min = 3, planet_count_max = 7, input_pnum = "N/A"):
    # 2.1.0: Draw a planet number randomly from a uniform distribution of planet_count_min to planet_count_max. If a planet number is specified, choose that instead.
    if input_pnum=="N/A":
        random_pnum = np.random.randint(planet_count_min, planet_count_max+1) # randint is a discrete uniform distribution. It is [inclusive, exclusive), which is why we add one to our max planet count.
    else:
        random_pnum = input_pnum

    # 2.1.1: Find all of the systems with that planet number (keep in mind that some of these systems actually have more planets without radius measurements).
    stars = np.unique(potential_systems[potential_systems["sy_pnum"]==random_pnum]["hostname"])
   
    # 2.1.2: Select one of those systems and get the stellar mass in that system.
    # Note: we take the median star mass as sometimes systems have slightly different reported star masses for different planets (see KOI-351 for example).
    random_system = np.random.choice(stars)
    random_st_mass = np.median(potential_systems[(potential_systems["hostname"]==random_system) & (potential_systems["sy_pnum"]==random_pnum)]["st_mass"])
   
    # 2.1.3: Get the planetary radii of the system.
    random_pl_radii = potential_systems[(potential_systems["hostname"]==random_system) & (potential_systems["sy_pnum"]==random_pnum)]["pl_rade"]
    random_pl_radii = random_pl_radii.to_numpy()
    # 2.1.4: Determine the masses of each planet using Forecaster (Chen and Kipping 2017) and convert to solar mass units.
    # Note: The same draw from the radius distribution will yield a different mass each time the code is run as Forecaster is probabilistic (see earlier versions of the code).
    # Note: This can lead to masses above 30Mearth, so we tell the system to find new masses if that happens.
    # Note: mr.Rpost2M can take an array of masses, but we want to keep all masses that are under the threshold and only grab a new mass for a planet that is outside of the range.
    random_pl_masses = np.array([])
    for planet_idx in range(random_pnum):
        # 2.1.4.1: Determine the mass for a given radii.
        random_pl_mass = mr.Rpost2M([random_pl_radii[planet_idx]], unit = 'Earth', grid_size = int(1e5))
        counter = 0
        # 2.1.4.2: Redraw the mass of the planet if it outside of a range of 0.1-30Mearth. If the code fails to find a mass in range after 100 iterations, set the mass to 0.1 or 30Mearth, whichever is closer.
        while (minimum_mass<random_pl_mass[0] and random_pl_mass[0]<maximum_mass) == False: # had previously written minimum_mass<random_pl_mass<maximum_mass == False for all simulations before 12/02/24.
            counter+=1
            random_pl_mass = mr.Rpost2M([random_pl_radii[planet_idx]], unit = 'Earth', grid_size = int(1e5))
            if counter == 100:
                differencemaxmass = random_pl_mass-maximum_mass
                differenceminmass = random_pl_mass-minimum_mass
                if np.min([differencemaxmass, differenceminmass]) == differencemaxmass:
                    random_pl_mass = maximum_mass
                elif np.min([differencemaxmass, differenceminmass]) == differenceminmass:
                    random_pl_mass = minimum_mass
                break
        random_pl_masses = np.append(random_pl_masses, random_pl_mass)
    # 2.1.4.3: Convert mass to solar units.
    random_pl_masses = (random_pl_masses*(u.M_earth)).to(u.Msun).value

    # 2.1.5: Set the innermost planet semi-major axis
    innermost_orbsmax = inner_disk_edge+displacement_from_inner_disk_edge
    innermost_orbper = np.sqrt((innermost_orbsmax**3)/(random_st_mass+random_pl_masses[0]))
   
    # 2.1.6: Determine the period ratios of each planet in the system by drawing from a fitted lognormal distribution of the period ratios less than 5.
    P_ratios = kepler["P/P-"].dropna()  
    P_ratios_under_5 = P_ratios[P_ratios<5]
    shape, loc, scale = lognorm.fit(P_ratios_under_5)
    random_p_ratio_lognormal = lognorm.rvs(shape, loc, scale, size = random_pnum-1)
   
    # 2.1.5: Determine the periods of each planet through the period ratios and innermost period.
    # 2.1.6: determine the corresponding semi-major axes from Kepler's third law.
    random_pl_orbpers = [innermost_orbper]
    random_pl_orbsmaxes = [innermost_orbsmax]
    for planet_idx in range(random_pnum-1):
        # P_new = P/P_ * P_old
        period = random_p_ratio_lognormal[planet_idx]*random_pl_orbpers[-1]
        random_pl_orbpers.append(period)
        semi_major = np.cbrt((random_st_mass+random_pl_masses[planet_idx+1])*(period)**2)
        random_pl_orbsmaxes.append(semi_major)
   
    # 2.1.7: Return nonangular orbital parameters.
    return random_pnum, random_pl_masses, random_st_mass, random_pl_orbsmaxes

# 2.2: Write a function to initialize a planetary system's angular parameters, eccentricity, inclination, longitude of the ascending node, argument of the pericenters, and mean anomalies.
def Keplerian_angle_parameters(pnum):
    # 2.2.0: Write, e, i, and Omega as 0.
    random_pl_orbeccens = np.zeros(pnum)
    random_pl_orbincs = np.zeros(pnum)
    random_pl_long_ascs = np.zeros(pnum)
 
    # 2.2.1: Pull omega and M randomly from distributions of 0-2pi.
    random_pl_arg_pericenters = np.random.uniform(0,1, pnum)*2*np.pi
    random_pl_mean_anoms = np.random.uniform(0,1, pnum)*2*np.pi
 
    # 2.2.2: Return angular orbital parameters.
    return random_pl_orbeccens, random_pl_orbincs, random_pl_long_ascs, random_pl_arg_pericenters, random_pl_mean_anoms

# 2.3: Write a function to initialize the migration force parameters, inner disk edge, inner disk edge with, disk aspect ratio at 1 AU, gas disk surface density, disk surface density exponent, and flaring index.
def migration_parameters():
    # 2.3.0: Set inner disk edge and width values.
    inner_edge = 0.05
    inner_edge_width = 0.01
    
    # 2.3.1: Draw K factor randomly from a uniform distribution of 10-1000.
    K = 10**np.random.uniform(1,3)
 
    # 2.3.2: Draw disk surface density randomly from a uniform distribution of 10-10000 g/cm^2.
    sigma_norm = 10**(np.random.uniform(1,4))
    sd_0 = (sigma_norm*u.g/u.cm**2).to(u.Msun/u.AU**2).value
 
     # 2.3.3: Set relevant exponents.
    alpha = 1.5
    beta = 0.0
 
    # 2.3.4: Return migration parameters.
    return inner_edge, inner_edge_width, K, sd_0, alpha, beta

# 2.4: Write a function to establish the integration timeframe of each simulation, minimum integration time, maximum integration time, and timestep.
def integration_timeframe():
    # 2.4.0: Set itime range and timestep.
    minimum_itime = 30e3
    maximum_itime = 10e6
    timestep = 0.05 # times the innermost orbital period
   
    # 2.4.1: Return integration timeframe parameters.
    return minimum_itime, maximum_itime, timestep

# 2.5: Write a function to colate all of the system and integration parameters. These functions were written modularly so that other perscriptions could be swapped in and out. Those others are currently excluded from this file and may require adjustment to be compatible.
def general_simulation_parameters_v10(input_pnum = "N/A"):
    # 2.5.0: Colate all simulation parameters.
    random_pnum, random_pl_masses, random_st_mass, random_pl_orbsmaxes = Keplerian_parameters(input_pnum = input_pnum)
    random_pl_orbeccens, random_pl_orbincs, random_pl_long_ascs, random_pl_arg_pericenters, random_pl_mean_anoms = Keplerian_angle_parameters(random_pnum)
    inner_edge, inner_edge_width, K, sd_0, alpha, beta = migration_parameters()
    minimum_itime, maximum_itime, timestep = integration_timeframe()
   
    # 2.5.1: Return all simulation parameters.
    return random_pnum, random_pl_masses, random_st_mass, random_pl_orbsmaxes, random_pl_orbeccens, random_pl_orbincs, random_pl_long_ascs, random_pl_arg_pericenters, random_pl_mean_anoms, inner_edge, inner_edge_width, K, sd_0, alpha, beta, minimum_itime, maximum_itime, timestep

# 2.6: Write a function that can format a set of system and integration parameters for reintegration from the results of a previous simulation.
# Note: currently, some csvs has strange spacing in the keys when it is read in. That spacing is left in right now.
def find_simulation_parameters(sim_results, run):
    # 2.6.0: Confirm that the simulation used a known method for establishing initial conditions. These parameter methods are alternatives to general_simulation_parameters_v3() above.
    parameter_method = sim_results['parameter_method'].iloc[run]
    assert parameter_method == 'general' or 'general_v10' or 'general_v9' or 'general_v8' or 'general_v7' or 'general_v6' or 'general_v5' or 'general_v4' or 'general_v3' or 'general_v2' or 'general_v1' or 'TOI_1136'
    
    # 2.6.1: Unpack all values from the dataframe.
    random_pnum = sim_results['pnum'].iloc[run]
    pl_orbsmax_0 = sim_results['pl_orbsmax_0'].iloc[run]
    pl_orbsmax_1 = sim_results['pl_orbsmax_1'].iloc[run]
    pl_orbsmax_2 = sim_results['pl_orbsmax_2'].iloc[run]
    pl_orbsmax_3 = sim_results['pl_orbsmax_3'].iloc[run] 
    pl_orbsmax_4 = sim_results['pl_orbsmax_4'].iloc[run]
    pl_orbsmax_5 = sim_results['pl_orbsmax_5'].iloc[run]
    pl_orbsmax_6 = sim_results['pl_orbsmax_6'].iloc[run]
    pl_mass_0 = sim_results['pl_mass_0'].iloc[run]
    pl_mass_1 = sim_results['pl_mass_1'].iloc[run]
    pl_mass_2 = sim_results['pl_mass_2'].iloc[run]
    pl_mass_3 = sim_results['pl_mass_3'].iloc[run]
    pl_mass_4 = sim_results['pl_mass_4'].iloc[run]
    pl_mass_5 = sim_results['pl_mass_5'].iloc[run]
    pl_mass_6 = sim_results['pl_mass_6'].iloc[run]
    random_st_mass = sim_results['st_mass'].iloc[run]
    kep_angles = sim_results['kep_angles'].iloc[run]
    arg_pericenter_0 = sim_results['arg_peri_0'].iloc[run] # really should be arg_pericenter. In old simulations, it is incorrectly labeled.
    arg_pericenter_1 = sim_results['arg_peri_1'].iloc[run]
    arg_pericenter_2 = sim_results['arg_peri_2'].iloc[run]
    arg_pericenter_3 = sim_results['arg_peri_3'].iloc[run]
    arg_pericenter_4 = sim_results['arg_peri_4'].iloc[run]
    arg_pericenter_5 = sim_results['arg_peri_5'].iloc[run]
    arg_pericenter_6 = sim_results['arg_peri_6'].iloc[run]
    mean_anom_0 = sim_results['mean_anom_0'].iloc[run]
    mean_anom_1 = sim_results['mean_anom_1'].iloc[run]
    mean_anom_2 = sim_results['mean_anom_2'].iloc[run]
    mean_anom_3 = sim_results['mean_anom_3'].iloc[run]
    mean_anom_4 = sim_results['mean_anom_4'].iloc[run]
    mean_anom_5 = sim_results['mean_anom_5'].iloc[run]
    mean_anom_6 = sim_results['mean_anom_6'].iloc[run]
    disk_inner_edge = sim_results['disk_inner_edge'].iloc[run]
    disk_inner_edge_width = sim_results['disk_inner_edge_width'].iloc[run]
    sd_0 = sim_results['Sigma'].iloc[run]*((1*u.g/u.cm**2).to(u.Msun/u.AU**2).value) # convert back to units used by the code
    K = sim_results['K'].iloc[run]
    alpha = sim_results['alpha'].iloc[run]
    beta = sim_results['beta'].iloc[run]

    # 2.6.2: Repack specific values into appropriate arrays
    random_pl_masses = np.array([pl_mass_0, pl_mass_1, pl_mass_2, pl_mass_3, pl_mass_4, pl_mass_5, pl_mass_6])
    random_pl_masses = random_pl_masses[~np.isnan(random_pl_masses)] # drop nans (represents the fact that there are fewer than 7 planets in the system)
    random_pl_orbsmaxes = np.array([pl_orbsmax_0, pl_orbsmax_1, pl_orbsmax_2, pl_orbsmax_3,	pl_orbsmax_4,	pl_orbsmax_5,	pl_orbsmax_6])
    random_pl_orbsmaxes = random_pl_orbsmaxes[~np.isnan(random_pl_orbsmaxes)] # drop nans 
    if kep_angles == 0.0:
        random_pl_orbeccens = np.zeros(random_pnum)
        random_pl_orbincs = np.zeros(random_pnum)
        random_pl_long_ascs = np.zeros(random_pnum)
    else: 
        print("The Keplerian angles are not all 0. Adjust code before saving as currently will save with a 0 value for all angles.")
        random_pl_orbeccens = np.zeros(random_pnum)
        random_pl_orbincs = np.zeros(random_pnum)
        random_pl_long_ascs = np.zeros(random_pnum)
    random_pl_arg_pericenters = np.array([arg_pericenter_0, arg_pericenter_1, arg_pericenter_2, arg_pericenter_3, arg_pericenter_4, arg_pericenter_5, arg_pericenter_6])
    random_pl_arg_pericenters = random_pl_arg_pericenters[~np.isnan(random_pl_arg_pericenters)]
    random_pl_mean_anoms = np.array([mean_anom_0, mean_anom_1, mean_anom_2, mean_anom_3, mean_anom_4, mean_anom_5, mean_anom_6])    
    random_pl_mean_anoms = random_pl_mean_anoms[~np.isnan(random_pl_mean_anoms)]
    print("Assuming a minimum and maximum integration time of 30kyr and 10Myr respectively alongside a timestep of 0.05 times the innermost orbital period.")
    minimum_itime = 30e3
    maximum_itime = 10e6
    timestep = 0.05 # times the innermost orbital period
    
    # 2.6.3: Return specific values. 
    return random_pnum, random_pl_masses, random_st_mass, random_pl_orbsmaxes, random_pl_orbeccens, random_pl_orbincs, random_pl_long_ascs, random_pl_arg_pericenters, random_pl_mean_anoms, disk_inner_edge, disk_inner_edge_width, K, sd_0, alpha, beta, minimum_itime, maximum_itime, timestep 


# Stage 3: Write additional functions to compute the resonance angles.
# 3.0: Write all first, second, and third order resonances from 1.1-4. Format them as Python fractions.
global decimal_resonances, fractional_resonances
decimal_resonances = [11/10, 21/19, 10/9, 19/17, 28/25, 9/8, 26/23, 17/15, 25/22, 8/7, 15/13, 22/19, 7/6, 20/17, 13/11, 19/16, 6/5, 17/14, 11/9, 16/13, 5/4, 14/11, 9/7, 13/10, 4/3, 11/8, 7/5, 10/7, 3/2, 8/5, 5/3, 7/4, 2/1, 5/2, 3/1, 4/1]
fractional_resonances = [fractions.Fraction(11,10), fractions.Fraction(21,19), fractions.Fraction(10, 9), fractions.Fraction(19,17), fractions.Fraction(28,25), fractions.Fraction(9,8), fractions.Fraction(26,23), fractions.Fraction(17,15), fractions.Fraction(25,22), fractions.Fraction(8,7),
                         fractions.Fraction(15,13), fractions.Fraction(22,19), fractions.Fraction(7,6), fractions.Fraction(20,17), fractions.Fraction(13,11), fractions.Fraction(19,16), fractions.Fraction(6,5), fractions.Fraction(17,14), fractions.Fraction(11,9),
                         fractions.Fraction(16,13), fractions.Fraction(5,4), fractions.Fraction(14,11), fractions.Fraction(9,7), fractions.Fraction(13,10), fractions.Fraction(4,3), fractions.Fraction(11,8), fractions.Fraction(7,5), fractions.Fraction(10,7),
                         fractions.Fraction(3,2), fractions.Fraction(8,5), fractions.Fraction(5,3), fractions.Fraction(7,4), fractions.Fraction(2,1), fractions.Fraction(5,2), fractions.Fraction(3,1), fractions.Fraction(4,1)]

# 3.1: Write a function to define p, q, two coprime integers such that period ratio ~ p/q. Here, the order of resonance is (p-q). This is the definition presented in Dai et al. (2023).
def get_p_q(fractional_resonance:fractions.Fraction) -> tuple[int, int]:
    # 3.1.0: Define p as the numerator of a given resonance.
    p = fractional_resonance.numerator
    # 3.1.1: Define q as p-p-q or equivilently, the numerator-denominator.
    q = fractional_resonance.denominator
    # 3.1.2: Return p and q as a tuple.
    return (p, q)
# Note: The definition used in celmech (Hadden and Tamayo 2022) is period ratio ~ p/(p-q). The definition in Murray and Dermott (1999) is period ratio ~(p+q)/q. Both of these have resonance of order q.
# Note: We thus write a modified function for p and q in the Hadden and Tamayo (2022) notation.
def get_p_q_HT(fractional_resonance:fractions.Fraction) -> tuple[int, int]:
    # 3.1.0: Define p as the numerator of a given resonance.
    p = fractional_resonance.numerator
    # 3.1.1: Define q as p-p-q or equivilently, the numerator-denominator.
    q = p-fractional_resonance.denominator
    # 3.1.2: Return p and q as a tuple.
    return (p, q)

# 3.2: Write a function to compute the corresponding f and g coefficients for a set p/(p-q) resonances and return a dictionary of relevant f and g coefficients. Only run this function if the dictionary is not up to date in the function get_f_g.
# Note: this function takes in p, q in the form used by Hadden and Tamayo (2022).
def setup_f_g_coefficients(resonances = fractional_resonances):
    # 3.2.0: Initialize dictionary of (p,q): (f,g).
    pq_fg = {}
    # 3.2.1: For each (p,q), find (f,g) and add to the dictionary.
    for frac_res in resonances:
        # 3.2.1.0: Get p, q for a particular fractional resonance.
        pq = get_p_q_HT(frac_res)
        p, q = pq[0], pq[1]
        # 3.2.1.1: Get (f,g) for a particular p, q.
        fg = disturb.get_fg_coefficients(p, q)
        # 3.2.1.2: Update the dictionary to include this pair of (p, q): (f, g).
        pq_fg.update({pq: fg})
    # 3.2.2: Return the dictionary.
    return pq_fg

# 3.3: Write and the dictionary if there have been modifications to the number of resonances (i.e. there are not 36 anymore) and save it in the function get_fg.
if len(fractional_resonances) != 36:
    # 3.3.0: Get the f and g coefficient for each resonance.
    pq_fg = setup_f_g_coefficients(fractional_resonances)
    # 3.3.1: Define get_fg as a function to recall (f,g) corresponding to a given p, q.
    def get_fg(p: int, q: int) -> tuple[float, float]:
        """Compute the f and g coefficients for the p:p-q resonance"""
        # 3.3.1.0: If p, q is in pq_fg, return the f, g of p, q.
        try:
            return pq_fg[(p, q)]
        # 3.3.1.1: If p, q is not in pq_fg, return NaN for f and g.
        except KeyError:
            return np.nan, np.nan
# 3.4: Write the function of the f and g coefficients for all 36 possible resonances between 1.1-4.
elif len(fractional_resonances) == 36:
    # 3.4.0: Define get_fg as a function to recall (f,g) corresponding to a given p, q.
    def get_f_g(p: int, q: int) -> tuple[float, float]:
        """Compute the f and g coefficients for the p:p-q resonance"""
        # 3.4.0.0: If p, q is in pq_fg, return the f, g of p, q.
        try:
            return {(11, 1): (-8.4757071483941, 8.892509247929354),
                    (21, 2): (-9.684217813918819, 10.22101857260467),
                    (10, 1): (-7.672611034804834, 8.090770597474132),
                    (19, 2): (-8.703813372755441, 9.241601431793642),
                    (28, 3): (-9.730043362557275, 10.354369559804102),
                    (9, 1): (-6.869251916654681, 7.289089771449579),
                    (26, 3): (-8.980284266983487, 9.605262981458454),
                    (17, 2): (-7.723158210197641, 8.262182273828888),
                    (25, 3): (-8.605352270373526, 9.230699657708843),
                    (8, 1): (-6.065523627134207, 6.487489727978886),
                    (15, 2): (-6.742144745278571, 7.282761214704821),
                    (22, 3): (-7.480284735162513, 8.106958574829655),
                    (7, 1): (-5.2612538318883395, 5.686007411509038),
                    (20, 3): (-6.729939531504269, 7.357742469634851),
                    (13, 2): (-5.760593404498293, 6.303339217551179),
                    (19, 3): (-6.354645757847441, 6.983112559869986),
                    (6, 1): (-4.456142785058929, 4.884706297500388),
                    (17, 3): (-5.603736926457859, 6.23379622068279),
                    (11, 2): (-4.77818004814446, 5.32391988128576),
                    (16, 3): (-5.228077049658948, 5.859102779106245),
                    (5, 1): (-3.6496182441157066, 4.08370537179621),
                    (14, 3): (-4.476179610886548, 5.109620500457167),
                    (9, 2): (-3.794252821934526, 4.344515739752699),
                    (13, 3): (-4.099839820750246, 4.734817303383065),
                    (4, 1): (-2.840431856721869, 3.2832567218222195),
                    (11, 3): (-3.3459565332539976, 3.9850325208989763),
                    (7, 2): (-2.8072722991054757, 3.365175666976672),
                    (10, 3): (-2.968121115950039, 3.610017440085913),
                    (3, 1): (-2.0252226899385946, 2.484005183303941),
                    (8, 3): (-2.2092216711500767, 2.859607186084824),
                    (5, 2): (-1.8124665351918696, 2.386158133409964),
                    (7, 3): (-1.826902781789326, 2.4841286121439907),
                    (2, 1): (-1.1904936978495033, 0.428389834143899),
                    (5, 3): (-1.0470767047914566, 1.7324826721000228),
                    (3, 2): (-0.9905647816074253, 0.9059256153690979),
                    (4, 3): (-0.7930604747813073, 1.0779956916619475)}[(p,q)]
        # 3.4.0.1: If p, q is not in pq_fg, return NaN for f and g.
        except KeyError:
            return np.nan, np.nan
       
# 3.5: Write a function to (a) reduce the 2 or 3 body resonant angle to a 0 to 360 or -180 to 180 range and (b) compute the resonant angle amplitude.
def report_phi(phi_deg, range_0_360= True):
    # 3.5.0: modulo the input angle to an appropriate range.
    if range_0_360 == True:
        # 3.5.0.0: assign the angles to be between 0 and 360.
        phi_hat = np.remainder(phi_deg, 360.)
    elif range_0_360 == False:
        # 3.5.0.1: assign the angles to be between -180 and 180.
        phi_hat = np.remainder(phi_deg+180., 360.)-180.
   
    # 3.5.1: Define the last set of values used when saving final conditions as the last 0.05% of values.
    # Note: This is also used to define the range we look at to compute the resonant amplitude.
    cutoff = int(len(phi_deg)/10)
    if cutoff<1:
    # 3.5.1.0: Set the cutoff to the last step if the total number of steps is less than 200.
        cutoff = 1
   
    # 3.5.2: Compute the libration amplitude via equation 8 of Mullholland et al. (2018) over the range after the cutoff.
    # Note: This amplitude physially corresponds to the amplitude of the sinsusoidal libration of the resonant angle (if it is actually sinsusoidal).
    phi_amp = np.sqrt(2/len(phi_hat[-cutoff:])*np.sum((phi_hat[-cutoff:]-np.nanmean(phi_hat[-cutoff:]))**2))
    return phi_hat, phi_amp

# Stage 4: Write a function that plots orbital parameters of the system over time, saves final values of those parameters, and prints final values of those parameters.
# 4.0: Write a function to plot semi-major axes(kyr), semi-major axis ratios(kyr), periods(kyr), period ratios(kyr), deltas(kyr), eccentricities(kyr), phi_2s(kyr), phi_3s(kyr).
def plot_save_print(times, a, omega, Omega, lam, per, e, random_pnum, version, date, txt_file, planet_count_max = 7):
    # 4.0.0: Define the last set of values used when saving and printing final conditions as the last 0.05% of values. Print a flag that we are displaying final conditions.
    print("\nFinal Conditions:")
    cutoff = int(len(times)/200)
    if cutoff<1:
        # 4.0.0.0: Set the cutoff to the last step if the total number of steps is less than 200.
        cutoff = 1

    # 4.0.1: Plot, save, and print the semi-major axis of each object.
    plt.subplots(1,1,figsize=(10,3))
    plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.15)
    for i in range(planet_count_max):
        # 4.0.1.0: When the index exceeds the actual planet number, skip plotting.
        if i>random_pnum-1: # -1 as we start at index 0.
            # 4.0.1.0.0: Instead, write a NaN in the position of the text file.
            txt_file.write(str(np.nan)+",")
            continue
        # 4.0.1.1: Plot each individual semi-major axis against kyr.
        plt.plot(times*0.001, a[i,:], label = '$a_'+str(i)+'$', linewidth = 4, alpha = 0.6)
        # 4.0.1.2: Save the final value of each semi-major axis averaged after the cutoff.
        txt_file.write(str(np.nanmean(a[i,-cutoff:]))+",")
        # 4.0.1.3: Print the closest value of each semi-major axis to the star. This is not necessarily identical to the saved value.
        print("\t* Closest Approach of Planet "+str(i)+": "+str(np.nanmin(a[i,:])))
    plt.xlabel('Time [kyr]')
    plt.ylabel('Semi-major axis [AU]')
    plt.legend()
    plt.savefig('plots/pdfs/Migration Runs '+date+"/"+version+'a.pdf')
    plt.savefig('plots/jpgs/Migration Runs '+date+"/"+version+'a.jpg')

    # 4.0.2: Plot, save, and print the semi-major axis ratio for each adjacent planet pair.
    plt.subplots(1,1,figsize=(10,3))
    plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.15)
    for i in range(planet_count_max-1):
        # 4.0.2.0: When the index exceeds the actual planet number, skip plotting.
        if i>random_pnum-2: # -1 as we start at index 0 and another -1 for the fact that there is one less semi-major axis ratio than there are semi-major axes.
            # 4.0.2.0.0: Instead, write a NaN in the position of the text file.
            txt_file.write(str(np.nan)+",")
            continue
        # 4.0.2.1: Plot each semi-major axis ratio against kyr.
        plt.plot(times*0.001, a[i+1,:]/a[i,:],label = "$ a_"+str(i+1)+"/a_"+str(i)+"$")
        # 4.0.2.2: Save the final value of each semi-major axis ratio averaged after the cutoff.
        txt_file.write(str(np.nanmean(a[i+1,-cutoff:]/a[i,-cutoff:]))+",")
        # 4.0.4.3: Print the final value of each semi-major axis ratio averaged after the cutoff.
        print("\t* Final Semi-Major Axis Ratio of Planet Pair "+str(i)+str(i+1)+": "+str(np.nanmean(a[i+1,-cutoff:]/a[i,-cutoff:])))
    plt.ylabel('a ratio')
    plt.legend()
    plt.xlabel('Time [kyr]')
    plt.savefig('plots/pdfs/Migration Runs '+date+"/"+version+'a_ratio.pdf')
    plt.savefig('plots/jpgs/Migration Runs '+date+"/"+version+'a_ratio.jpg')

    # 4.0.3: Plot, save, and print the period of each object.
    plt.subplots(1,1,figsize=(10,3))
    plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.15)
    for i in range(planet_count_max):
        # 4.0.3.0: When the index exceeds the actual planet number, skip plotting.
        if i>random_pnum-1: # -1 as we start at index 0.
            # 4.0.3.0.0: Instead, write a NaN in the position of the text file.
            txt_file.write(str(np.nan)+",")
            continue
        # 4.0.3.1: Plot each individual period against kyr.
        plt.plot(times*0.001, per[i,:], label = "$P_"+str(i)+"$", linewidth = 4, alpha = 0.6)
        # 4.0.3.2: Save the final value of each period averaged after the cutoff.
        txt_file.write(str(np.nanmean(per[i,-cutoff:]))+",")
        # 4.0.3.3: Print the final value of each period averaged after the cutoff.
        print("\t* Final Period of Planet "+str(i)+": "+str(np.nanmean(per[i,-cutoff:])))
    plt.ylabel('Period [yr]')
    plt.legend()
    plt.xlabel('Time [kyr]')
    plt.savefig('plots/pdfs/Migration Runs '+date+"/"+version+'P.pdf')
    plt.savefig('plots/jpgs/Migration Runs '+date+"/"+version+'P.jpg')

    # 4.0.4: Plot and save the period ratio for each adjacent planet pair.
    plt.subplots(1,1,figsize=(10,3))
    plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.15)
    for i in range(planet_count_max-1):
        # 4.0.4.0: When the index exceeds the actual planet number, skip plotting.
        if i>random_pnum-2: # -1 as we start at index 0 and another -1 for the fact that there is one less period ratio than there are periods.
            # 4.0.4.0.0: Instead, write a NaN in the position of the text file.
            txt_file.write(str(np.nan)+",")
            continue
        # 4.0.4.1: Plot each period ratio against kyr.
        plt.plot(times*0.001, per[i+1,:]/per[i,:],label = "$P_"+str(i+1)+"/P_"+str(i)+"$")
        # 4.0.4.2: Save the final value of each period ratio averaged after the cutoff.
        txt_file.write(str(np.nanmean(per[i+1,-cutoff:]/per[i,-cutoff:]))+",")
        # 4.0.4.3: Print the final value of each period ratio averaged after the cutoff.
        print("\t* Final Period Ratio of Planet Pair "+str(i)+str(i+1)+": "+str(np.nanmean(per[i+1,-cutoff:]/per[i,-cutoff:])))
    plt.ylabel('Period Ratio')
    plt.legend()
    plt.xlabel('Time [kyr]')
    plt.savefig('plots/pdfs/Migration Runs '+date+"/"+version+'P_ratio.pdf')
    plt.savefig('plots/jpgs/Migration Runs '+date+"/"+version+'P_ratio.jpg')

    # 4.0.5: Plot, save, and print the eccentricity for each planet.
    plt.subplots(1,1,figsize=(10,3))
    plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.15)
    for i in range(planet_count_max):
        # 4.0.5.0: When the index exceeds the actual planet number, skip plotting.
        if i>random_pnum-1: # -1 as we start at index 0.
            # 4.0.5.0.0: Instead, write a NaN in the position of the text file.
            txt_file.write(str(np.nan)+",")
            continue
        # 4.0.5.1: Plot each individual eccentricity against kyr.
        plt.plot(times*0.001, e[i,:], label = '$e_'+str(i)+'$')
        # 4.0.5.2: Save the final value of eccentricity averaged after the cutoff.
        txt_file.write(str(np.nanmean(e[i,-cutoff:]))+",")
        # 4.0.5.3: Print the final value of eccentricity averaged after the cutoff.
        print("\t* Final Eccentricity of Planet "+str(i)+": "+str(np.nanmean(e[i, -cutoff:])))
    plt.ylabel('Eccentricity')
    plt.xlabel('Time [kyr]')
    plt.legend()
    plt.savefig('plots/pdfs/Migration Runs '+date+"/"+version+'e.pdf')
    plt.savefig('plots/jpgs/Migration Runs '+date+"/"+version+'e.jpg')

    # 4.0.6: Plot, save, and print the delta of each adjacent planet pair.
    # Note: We define delta as a measure of the proximity of a period ratio to the closest first, second or third order resonance.
    # Note: In other words, delta = p_ratio - p/(p-q) for the closest p, q pair. Delta ~ 0 for a system near resonance.
    plt.subplots(1,1,figsize=(10,3))
    plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.15)
    for i in range(planet_count_max-1):
        # 4.0.6.0: When the index exceeds the actual planet number, skip plotting.
        if i>random_pnum-2: # -1 as we start at index 0 and another -1 for the fact that there is one less delta than there are planets.
            # 4.0.6.0.0: Instead, write a NaN in the position of the text file for the delta and closest resonance.
            txt_file.write(str(np.nan)+",")
            txt_file.write(str(np.nan)+",")
            continue
        # 4.0.6.1: Determine the period ratio of a planet at the end of the simulation
        period_ratio = np.nanmean(per[i+1,-cutoff:]/per[i,-cutoff:])
        # 4.0.6.2: Determine the difference of that period ratio to every resonance in our library of resonance to find the cloest resonance at the end of the simulation.
        # Note: we could modify this to measure proximity to the closest resonance at every time point, letting the closest resonance parameter change over time.
        all_deltas = period_ratio-decimal_resonances
        # 4.0.6.3: Determine the cloest resonance at the end of the simulation.
        smallest_delta_idx = np.argmin(np.abs(all_deltas))
        closest_resonance = decimal_resonances[smallest_delta_idx]
        # 4.0.6.4: Determine the delta corresponding to the closest resonance at the end of the simulation.
        smallest_delta = (per[i+1,:]/per[i,:]) - closest_resonance
        # 4.0.6.5: Plot delta for each planet pair against kyr.
        plt.plot(times*0.001, smallest_delta, label = r"$\Delta_{"+str(i)+str(i+1)+"}$") # Note: could code in display of final closest resonance.
        # 4.0.6.6: Save the final value of delta averaged after the cutoff.
        txt_file.write(str(np.nanmean(smallest_delta[-cutoff:]))+',')
        # 4.0.6.7: Save the closest resonance that delta is determined from.
        txt_file.write(str(closest_resonance)+',')
        # 4.0.6.8: Print the final value of delta averaged after the cutoff.
        print("\t* Final Delta of Planet Pair "+str(i)+str(i+1)+": "+str(np.nanmean(smallest_delta[-cutoff:])))
    plt.ylabel(r'$\Delta$')
    plt.legend()
    plt.xlabel('Time [kyr]')
    plt.savefig('plots/pdfs/Migration Runs '+date+"/"+version+'delta.pdf')
    plt.savefig('plots/jpgs/Migration Runs '+date+"/"+version+'delta.jpg')

    # 4.0.7: Plot, save, and print the two body resonant angles of each adjacent planet pair.
    plt.subplots(1,1,figsize=(10,3))
    plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.15)
    for i in range(planet_count_max-1):
        # 4.0.7.0: When the index exceeds the actual planet number, skip plotting.
        if i>random_pnum-2: # -1 as we start at index 0 and another -1 for the fact that there is one less 2 body resonant angle than there are planets.
            # 4.0.7.0.0: Instead, write a NaN for each resonant angle and resonant angle amplitude.
            txt_file.write(str(np.nan)+",")
            txt_file.write(str(np.nan)+",")
            continue
        # 4.0.7.1: Determine the p and q integers and look up the f and g coefficients for the closest resonance at the end of the simulation.
        period_ratio = np.nanmean(per[i+1,-cutoff:]/per[i,-cutoff:])
        all_deltas = period_ratio-decimal_resonances
        smallest_delta_idx = np.argmin(np.abs(all_deltas))
        closest_fractional_resonance = fractional_resonances[smallest_delta_idx]
        p, q = get_p_q(closest_fractional_resonance)
        f, g = get_f_g(p, p-q)
        # 4.0.7.2: Compute the longitude of pericenter for each planet.
        varpi_i = omega+Omega
        # 4.0.7.3: Compute the mixed longitude of pericenter for each planet pair.
        # Note: This should be equivilent to z = np.angle(f*e[i,:]*np.exp(1j*varpi_i[i,:]) + g*e[i+1,:]*np.exp(1j*varpi_i[i+1,:]))
        varpi_i_iplus1 = np.arctan2((f*e[i,:]*np.sin(varpi_i[i,:])+g*e[i+1,:]*np.sin(varpi_i[i+1,:])), (f*e[i,:]*np.cos(varpi_i[i,:])+g*e[i+1,:]*np.cos(varpi_i[i+1,:])))
        # 4.0.7.4: Compute the resonant angle for each planet pair.
        phi = q*lam[i,:] - p*lam[i+1,:] + (p-q)*varpi_i_iplus1
        # 4.0.7.5: Convert the resonant angle into degrees.
        phi_deg = (phi*u.rad).to(u.deg).value
        # 4.0.7.6: Use report_phi to ensure that the resonant angle is within a 0 to 360 (or -180 to 180) range and compute the amplitude of phi.
        # Note: phi_hat, like phi, is now only defined for the specific case being considered here. Similarly, phi_amp is just a scalar value.
        phi_hat, phi_amp = report_phi(phi_deg, range_0_360 = True)
        # 4.0.7.7: Plot the resonant angle for each adjacent planet pair over time. Establish a separate cutoff for phi.
        phi_cutoff = int(len(phi_hat)/10)
        if phi_cutoff<1:
            phi_cutoff = 1
        plt.scatter(times*1e-3, phi_hat,marker = '.', s= 0.5)
        plt.plot(times[0:1]*1e-3, phi_hat[0:1], label = r'$\phi_{' +str(i)+str(i+1)+'}$') # Note: I believe this line simply makes the markers clearer for labeling.
        # 4.0.7.8: Save the resonant angle for each adjacent planet pair averaged after the cutoff.
        # Note: In the initial version of this script for TOI 1136, I used twice the cutoff for displaying (but not for saving) these values.
        txt_file.write(str(np.nanmean(phi_hat[-phi_cutoff:]))+",")
        txt_file.write(str(phi_amp)+",")
        # 4.0.7.9: Print the resonant angle for each adjacent planet pair averaged after the cutoff.
        print("\t* Final Resonant Angle of Planet Pair "+str(i)+str(i+1)+": "+str(np.nanmean(phi_hat[-phi_cutoff:])))
        # # 4.0.7.10: Print the resonant angle amplitude for each adjacent planet pair averaged after the cutoff.
        print("\t* Final Resonant Angle Amplitude of Planet Pair "+str(i)+str(i+1)+": "+str(np.nanmean(phi_amp)))
    plt.xlabel('Time [kyr]')
    plt.ylabel(r'2-Body Critical Angle $\phi$ [deg]')
    plt.ylim(0,360)
    plt.legend(loc = 'lower right')
    plt.savefig('plots/pdfs/Migration Runs '+date+"/"+version+'2body_angle.pdf')
    plt.savefig('plots/jpgs/Migration Runs '+date+"/"+version+'2body_angle.jpg')

    # 4.0.8: Plot and save the three body resonant angles of each adjacent planet pair.
    plt.subplots(1,1,figsize=(10,3))
    plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.15)
    for i in range(planet_count_max-2):
        # 4.0.8.0: When the index exceeds the actual planet number, skip plotting.
        if i>random_pnum-3: # -1 as we start at index 0 and another -2 for the fact that there is two less three body angles than there are planets.
            # 4.0.7.0.0: Instead, write a NaN for each resonant angle and resonant angle amplitude.
            txt_file.write(str(np.nan)+",")
            txt_file.write(str(np.nan)+",")
            continue
        # NOTE: Pickup here. Challenging as now we need p_12, q_12, p_23, q_23.
        # 4.0.8.1: Determine the p and q integers for the closest resonance at the end of the simulation for the i, i+1 planets.
        period_ratio = np.nanmean(per[i+1,-cutoff:]/per[i,-cutoff:])
        all_deltas = period_ratio-decimal_resonances
        smallest_delta_idx = np.argmin(np.abs(all_deltas))
        closest_fractional_resonance = fractional_resonances[smallest_delta_idx]
        p_i_iplus1, q_i_iplus1 = get_p_q(closest_fractional_resonance)
        # 4.0.8.2: Determine the p and q integers for the closest resonance at the end of the simulation for the i+1, i+2 planets.
        period_ratio = np.nanmean(per[i+2,-cutoff:]/per[i+1,-cutoff:])
        all_deltas = period_ratio-decimal_resonances
        smallest_delta_idx = np.argmin(np.abs(all_deltas))
        closest_fractional_resonance = fractional_resonances[smallest_delta_idx]
        p_iplus1_iplus2, q_iplus1_iplus2 = get_p_q(closest_fractional_resonance)
        # 4.0.8.3: Compute the three body resonant angle for the i, i+1, i+2 planets. If the order is the same, just cancel the final term. If the orders are different, multiply each two body angle by the order of the other to cancel the final term.
        if (p_i_iplus1 - q_i_iplus1) == (p_iplus1_iplus2 - q_iplus1_iplus2):
            phi_i_iplus1_iplus2 = q_i_iplus1 * lam[i,:] - (p_i_iplus1+q_iplus1_iplus2) * lam[i+1,:] + p_iplus1_iplus2 * lam[i+2,:]
        elif (p_i_iplus1 - q_i_iplus1) != (p_iplus1_iplus2 - q_iplus1_iplus2):
            phi_i_iplus1_iplus2 = (p_iplus1_iplus2-q_iplus1_iplus2) * q_i_iplus1 * lam[i,:] - ((p_iplus1_iplus2 - q_iplus1_iplus2) * p_i_iplus1 + (p_i_iplus1 - q_i_iplus1) * q_iplus1_iplus2) * lam[i+1,:] + (p_i_iplus1 - q_i_iplus1) * p_iplus1_iplus2 * lam[i+2,:]          
        # 4.0.8.4: Convert the resonant angle into degrees.
        phi_deg = (phi_i_iplus1_iplus2*u.rad).to(u.deg).value
        # 4.0.8.6: Use report_phi to ensure that the resonant angle is within a 0 to 360 (or -180 to 180) range and compute the amplitude of phi.
        # Note: phi_hat, like phi, is now only defined for the specific case being considered here. Similarly, phi_amp is just a scalar value.
        phi_hat, phi_amp = report_phi(phi_deg, range_0_360 = True)
        # 4.0.8.7: Plot the resonant angle for each planet triad over time.
        plt.scatter(times*1e-3, phi_hat, marker = '.', s= 0.5)
        plt.plot(times[0:1]*1e-3, phi_hat[0:1], label = r'$\phi_{' +str(i)+str(i+1)+str(i+2)+'}$') # Note: I believe this line simply makes the markers clearer for labeling.
        # 4.0.8.8: Save the resonant angle for each adjacent planet pair averaged after the cutoff.
        # Note: In the initial version of this script for TOI 1136, I used twice the cutoff for displaying (but not for saving) these values.
        txt_file.write(str(np.nanmean(phi_hat[-phi_cutoff:]))+",")
        txt_file.write(str(phi_amp)+",")
        # 4.0.8.9: Print the resonant angle for each adjacent planet triad averaged after the cutoff.
        print("\t* Final Resonant Angle of Planet Triplet "+str(i)+str(i+1)+str(i+2)+": "+str(np.nanmean(phi_hat[-phi_cutoff:])))
        # 4.0.8.10: Print the resonant angle amplitude for each adjacent planet triad averaged after the cutoff.
        print("\t* Final Resonant Angle Amplitude of Planet Triplet "+str(i)+str(i+1)+str(i+2)+": "+str(np.nanmean(phi_amp)))
    plt.xlabel('Time [kyr]')
    plt.ylabel(r'3-Body Critical Angle $\phi$ [deg]')
    plt.ylim(0,360)
    plt.legend(loc = 'lower right')
    plt.savefig('plots/pdfs/Migration Runs '+date+"/"+version+'3body_angle.pdf')
    plt.savefig('plots/jpgs/Migration Runs '+date+"/"+version+'3body_angle.jpg')

    plt.close('all')

# Stage 5: Write the migration function.
def run_migration_edge(index, txt_file, simulation_parameters, parameter_method, date, planet_count_max = 7, integration_time = "3taua", n_step=10000):
    random_pnum, random_pl_masses, random_st_mass, random_pl_orbsmaxes, random_pl_orbeccens, random_pl_orbincs, random_pl_long_ascs, random_pl_arg_pericenters, random_pl_mean_anoms, inner_edge, inner_edge_width, K, sd_0, alpha, beta, minimum_itime, maximum_itime, timestep = simulation_parameters
    txt_file.write(str(index)+",")
    # 5.1: Initialize semi-major axes, masses, mean anomalies, and argument of pericenters (using lists to print on the same line)).
    print("\nInitial Parameters:")
    # 5.1.0: Save and print the number of planets.
    txt_file.write(str(random_pnum)+",")
    print("\t* Number of Planets:", str(random_pnum))
    # 5.1.1: Print and save initial semi-major axes. When the index exceeds the actual planet number, skip printing.
    for i in range(planet_count_max):
        if i>random_pnum-1: # -1 as we start at index 0
            # 5.1.1.0: Write a NaN in the position of the text file.
            txt_file.write(str(np.nan)+",")
            continue
        # 5.1.1.1: Write the initial semi-major axes.
        txt_file.write(str(random_pl_orbsmaxes[i])+',')
    # 5.1.1.2: Print the initial semi-major axes.
    print("\t* Semi-Major Axes: ", str(list(random_pl_orbsmaxes)))

    # 5.1.2: Print the initial period ratios by determining them from Kepler's third law.
    period_ratios = []
    for i in range(len(random_pl_orbsmaxes)-1):
        # 5.1.2.0: Find each period ratio via semi-major axis ratios (Kepler's third law).
        period_ratios.append((random_pl_orbsmaxes[i+1]/random_pl_orbsmaxes[i])**(3/2.))
    # 5.1.2.1: Print all of the period ratios.
    print("\t* Period Ratios: ", str(list(period_ratios)))

    # 5.1.3: Save and print the initial masses, eccentricities, inclinations, and longitude of the ascending nodes. When the index exceeds the actual planet number, skip printing.
    for i in range(planet_count_max):
        if i>random_pnum-1: # -1 as we start at index 0
            # 5.1.3.0: Write a NaN in the position of the text file.
            txt_file.write(str(np.nan)+",")
            continue
        txt_file.write(str(random_pl_masses[i])+',')
    # 5.1.3.1: Print and save planet masses, star massses, eccentricities, inclinations, and longitude of the ascending nodes.
    print("\t* Planet Masses: ", str(list(random_pl_masses))) # already in solar mass units due to samples() function.
    print("\t* Star Mass: ", str(random_st_mass))
    txt_file.write(str(random_st_mass)+",")
    print("\t* Eccentricities: ", str(list(random_pl_orbeccens)))
    print("\t* Inclinations:", str(list(random_pl_orbincs)))
    print("\t* Longitudes of the Ascdending Node:", str(list(random_pl_long_ascs)))
    if random_pl_orbeccens.all() == random_pl_orbincs.all() == random_pl_long_ascs.all() == np.zeros(random_pnum).all():
        txt_file.write(str(0)+",")
    else:
        # 5.1.3.2: Double check that the eccentricities, inclinations, and longitude of the ascending nodes are all zero before saving as a 0.
        print("\t* The eccentricities, inclinations, and longitude of the ascending nodes are not all 0. Adjust code before saving as currently will save with a NaN value for all angles.")
        txt_file.write(str(np.nan)+",")

    # 5.1.4: Save and print the mean argument of pericenters and mean anomalies.
    for i in range(planet_count_max):
        if i>random_pnum-1:  # -1 as we start at index 0
            # 5.1.4.0: Write a NaN in the position of the text file.
            txt_file.write(str(np.nan)+",")
            continue
        # 5.1.4.1: Otherwise, write the appropriate argument of pericenter.
        txt_file.write(str(random_pl_arg_pericenters[i])+',')
    # 5.4.1.2: Print the mean argument of pericenters.
    print("\t* Mean Argument of Pericenters: ", str(list(random_pl_arg_pericenters)))
    for i in range(planet_count_max):
        if i>random_pnum-1: # -1 as we start at index 0
            # 5.1.4.3: Write a NaN in the position of the text file.
            txt_file.write(str(np.nan)+",")
            continue
        # 5.1.4.4: Otherwise, write the appropriate mean anomaly.
        txt_file.write(str(random_pl_mean_anoms[i])+',')
    print("\t* Mean Anomalies:", str(list(random_pl_mean_anoms)))

    # 5.2: Establish the orbital simulation and set up migration conditions.
    print("\nMigration Conditions: ")
    # 5.2.1: Initialize rebound simulation.
    sim = rebound.Simulation()
   
    # 5.2.2: Add appropriate units.
    sim.units = ('yr', 'AU', 'Msun')
   
    # 5.2.3: Add star.
    sim.add(m = random_st_mass, r = 0.0)
   
    # 5.2.4: Add planets (mass, semi-major axis, eccentricity, inclination, longitude of the ascending node, argument of the pericenter, mean anomaly).
    for i in range(random_pnum):
        sim.add(m = random_pl_masses[i], a=random_pl_orbsmaxes[i], e=random_pl_orbeccens[i], inc = random_pl_orbincs[i],
                Omega = random_pl_long_ascs[i], omega = random_pl_arg_pericenters[i], M = random_pl_mean_anoms[i])

    # 5.2.5: Move to center of mass frame and label planets.
    sim.move_to_com()
    ps = sim.particles

    # 5.2.6: Use reboundx to add the disk migration force.
    rebx = reboundx.Extras(sim)
    mig = rebx.load_force("type_I_migration")
    rebx.add_force(mig)


    # 5.2.7: Establish, save, and print parameters of the disk migration force.
    # 5.2.7.0: Set disk scale height as defined by inverting equation 3.9 of Pichierri et al. (2018).
    h_0 = np.sqrt(0.780/((2.7+1.1*alpha)*K))
    mig.params["tIm_scale_height_1"] = h_0
    # 5.2.7.1: Set, save, and print disk surface density.
    mig.params["tIm_surface_density_1"] = sd_0
    txt_file.write(str((sd_0*u.Msun/u.AU**2).to(u.g/u.cm**2).value)+",")
    print('\t* Density:', (sd_0*(u.Msun/u.AU**2)).to(u.g/u.cm**2).value, "g/cm^2")
    # 5.2.7.2: Set, save, and print disk surface density exponent.
    mig.params["tIm_surface_density_exponent"] = alpha
    txt_file.write(str(alpha)+",")
    print("\t* Surface Density Exponent:", alpha)
    # 5.2.7.3: Set, save, and print disk flaring index.
    mig.params["tIm_flaring_index"] = beta
    txt_file.write(str(beta)+",")
    print("\t* Flaring Index:", beta)
    # 5.2.7.4: Determine the scale height at the final planet (if the flaring index is 0, the scale height is constant). Save and print the disk scale height at 1 AU and at the outermost planet.
    h_outermost = h_0*ps[-1].a**(-mig.params["tIm_flaring_index"])
    print('\t* Scale Height at 1 AU:', (h_0*(u.AU)).to(u.cm).value, "cm")
    txt_file.write(str(h_0)+",")
    print('\t* Scale Height at Outermost Planet:', (h_outermost*(u.AU)).to(u.cm).value, "cm")
    txt_file.write(str(h_outermost)+",")
    # 5.2.7.5: Save, and print the "K factor" as defined in equation 3.9 of Pichierri et al. (2018).
    txt_file.write(str(K)+",")
    print('\t* K Factor:',K,'at 1 AU')
    # Set and print the K factor at the outermost planet.
    K_outermost = 0.78/(2.7+1.1*alpha)*h_outermost**(-2) # == K*h**2*h_i**(-2)
    txt_file.write(str(K_outermost)+",")
    print('\t* K Factor:',K_outermost,'at Outermost Planet')
   
    # 5.2.7: Calculate, save, and print relevant migration timescaes
    # 5.2.7.0: Calculate tau_wave as defined in equation 3.3 of Pichierri et al. (2018).
    tau_wave = (random_st_mass / ps[-1].m) * (random_st_mass /(sd_0*ps[-1].a**(-mig.params["tIm_surface_density_exponent"])*ps[-1].a**2)) * (h_outermost**4 / np.sqrt(sim.G*random_st_mass/ps[-1].a**3))
    # 5.2.7.1: Calculate, save, and print tau_e as defined in equation 3.2 of Pichierri et al. (2018).
    # Note: this is equivilent to tau_e = Mstar **2/ps[-1].m/(sd0*ps[-1].a**(-mig.params["tIm_surface_density_exponent"]))/ps[-1].a**2*h_i**4/np.sqrt(sim.G*Mstar/ps[-1].a**3)/0.78) as written by Prof. Dai.
    tau_e = tau_wave/0.780
    txt_file.write(str(tau_e)+",")
    print("\t* Tau_e:", tau_e, "yr")
    # 5.2.7.2: Calculate, save, and print tau_a of the outermost planet. This should be the largest tau_a as Koutermost is proportional to h_outermost.
    # Note: this is equivilent to tau_mig = (2*tau_wave)/(2.7+1.1*alpha)*h**(-2) => tau_a = tau_mig/2.
    tau_a = K_outermost*tau_e
    txt_file.write(str(tau_a)+",")  
    print("\t* Tau_a:", tau_a, "yr")

    # 5.2.8: Set inner disk edge parameters.
    # 5.2.8.0: Set and save the inner disk edge position.
    mig.params["ide_position"] = inner_edge
    txt_file.write(str(inner_edge)+",")
    # 5.2.8.1: Set and save the inner disk edge width.
    mig.params["ide_width"] = inner_edge_width
    txt_file.write(str(inner_edge_width)+",")
    # 5.2.8.2: Print planet stopping criterion at inner disk edge.
    print('\t* Planet will stop within {0:.3f} AU of the inner disk edge at {1} AU'.format(mig.params["ide_width"], mig.params["ide_position"]))

    # 5.3: Declare the run version by specifying the run index, inner edge value, K factor for the outermost planet value, and tau_a value.
    global version
    version = str(index)+'_inner_edge_' +'{0:1.1e}'.format(K_outermost)+'_taua_' +'{0:1.1e}'.format(tau_a)+'_'

    # 5.4: Establish temporal parameters and integrate
    # 5.4.1: Set the integrator to whfast.
    # Note: whfast is a fast integration scheme, but it does not provide accurate predictions when planets have close approaches. We will throw out simulations where this occurs later.
    sim.integrator = 'whfast'
   
    # 5.4.2: Set the timestep of the simulator to be 5% of corresponding period to a planet sitting at the inner disk edge.
    # 5.4.2.0: Assert that the timestep is 0.05 so that code throws an error if we adjust it accidentally.
    assert timestep == 0.05
    # 5.4.2.1: Set integration timestep using Kepler's third law.
    sim.dt = mig.params["ide_position"]**(3/2)*timestep

    # 5.4.3: Determine, print, and save the integration time and number of steps to be saved for plotting.
    if integration_time == "3tau_a":
       # 5.4.3.0: Set the integration time to 3*tau_a by default.
        T = 3*tau_a
        if T<minimum_itime:
            # 5.4.3.1: Increase the integration time to the minimum value specified in simulation_parameters. This (hopefully) enables more resonance capture.
            T = minimum_itime
            # 5.4.3.2: Print this modified integration time.
            print("\n3taua = "+str(3*tau_a)+"<"+str(minimum_itime)+"yr. Integrating for ", T, "yr with", n_step, "steps:")
        elif T>maximum_itime:
            # 5.4.3.3: Decrease the integration time to the maximum value specified in simulation_parameters. This reduces computational time but may yield missing resonances.
            T = maximum_itime
            # 5.4.3.4: Print this modified integration time.
            print("\n3taua = "+str(3*tau_a)+">"+str(maximum_itime)+"yr. Integrating for ", T, "yr with", n_step, "steps:")
        else:
            # 5.4.3.5: Print the standard 3_taua integration time if it is within the range of allowed integration times.
            print("\nIntegrating for ", T, "yr with", n_step, "steps:")
    else:
        # 5.4.3.1: If another value than "3tau_a" is specified, integrate for that time period with the specified number of plotting steps.
        T = integration_time
        print("\nIntegrating for ", T, "yr with", n_step, "steps:")

    # 5.4.4: Declare the times to save plotting results and initialize corresponding arrays.
    times = np.linspace(0, T, n_step)
    # 5.4.4.1: Initialize semi-major axes.
    a = np.zeros((random_pnum, n_step))
    # 5.4.4.2: Initialize periods.
    per = np.zeros((random_pnum, n_step))
    # 5.4.4.3: Initialize mean longitudes of pericenter.
    lam = np.zeros((random_pnum, n_step))
    # 5.4.4.4: Initialize eccentricities.
    e = np.zeros((random_pnum, n_step))
    # 5.4.4.5: Initialize mean arguments of pericenter.
    omega = np.zeros((random_pnum, n_step))
    # 5.4.4.6: Initialize longitudes of the ascending node.
    Omega = np.zeros((random_pnum, n_step))

    close_encounter = False
    # 5.4.6: Create a dictionary of the larger hill radius corresponding to each possible planet pair in the system.
    # 5.4.7: Integrate for every time in times and update values in arrays
    for i, t in enumerate(times):
        for i1, i2 in combinations(range(1, random_pnum+1),2):
            condition = sim.particles[i2].a-sim.particles[i1].a
            #dp = sim.particles[i1] - sim.particles[i2]
            #condition = np.sqrt(dp.x*dp.x+dp.y*dp.y+dp.z*dp.z)
            mutual_hill_radius = ((sim.particles[i1].a+sim.particles[i2].a)/2) * ((sim.particles[i1].m+sim.particles[i2].m)/(3*sim.particles[0].m))**(1/3)
            if condition<=5*mutual_hill_radius or np.isnan(mutual_hill_radius)==True:
                # 5.4.8.3: Stop integrating and save orbital parameters at last values before the close encounter.
                print("\t* Planets "+str(i1)+" and "+str(i2)+" had a close encounter. Ending simulation.")
                txt_file.write(str(t)+",")
                txt_file.write(str(True)+",")
                times = times[:i-1]
                a, per, lam, e, omega, Omega = a[:,:i-1], per[:,:i-1], lam[:,:i-1], e[:,:i-1], omega[:,:i-1], Omega[:,:i-1]
                # 5.4.8.4: update the close encounter flag to true as there was one.
                close_encounter = True
                # 5.4.8.5: If there immediately is a close encounter, fill all remaining cells in the spreadsheet with NaNs (except the parameter method and newline). Return to stop execution entirely (no plots are generated).
                if i == 0:
                    print("\t* As the simulation ended after"+str(i)+"steps, all final values are the same as the initial ones.")
                    txt_file.write(str(np.nan)+","+str(np.nan)+","+str(np.nan)+","+str(np.nan)+","+str(np.nan)+","+str(np.nan)+","+str(np.nan)+","+str(np.nan)+","+str(np.nan)+","+str(np.nan)+","+str(np.nan)+","+str(np.nan)+","+str(np.nan)+","+str(np.nan)+","+str(np.nan)+","+str(np.nan)+","+str(np.nan)+","+str(np.nan)+","+str(np.nan)+","+str(np.nan)+","+str(np.nan)+","+str(np.nan)+","+str(np.nan)+","+str(np.nan)+","+str(np.nan)+","+str(np.nan)+","+str(np.nan)+","+str(np.nan)+","+str(np.nan)+","+str(np.nan)+","+str(np.nan)+","+str(np.nan)+","+str(np.nan)+","+str(np.nan)+","+str(np.nan)+","+str(np.nan)+","+str(np.nan)+","+str(np.nan)+","+str(np.nan)+","+str(np.nan)+","+str(np.nan)+","+str(np.nan)+","+str(np.nan)+","+str(np.nan)+","+str(np.nan)+","+str(np.nan)+","+str(np.nan)+","+str(np.nan)+","+str(np.nan)+","+str(np.nan)+","+str(np.nan)+","+str(np.nan)+","+str(np.nan)+","+str(np.nan)+","+str(np.nan)+","+str(np.nan)+","+str(np.nan)+","+str(np.nan)+","+str(np.nan)+","+str(np.nan)+","+str(np.nan)+","+str(np.nan)+","+str(np.nan)+","+str(np.nan)+","+str(np.nan)+","+str(np.nan)+","+str(np.nan)+",")
                    txt_file.write(parameter_method+",")
                    txt_file.write(str(1)+",\n")
                    return close_encounter, random_pnum
                break
        # 5.4.9: Use the close encounter flag to stop integrating if a close encounter was reached.
        if close_encounter == True:
            break
        sim.integrate(t)
        for i_p in range(random_pnum):
                # 5.4.6.4: Update semi-major axes.
                a[i_p,i] = sim.particles[1+i_p].a#ps[i_p].a
                # 5.4.6.5: Update periods.
                per[i_p,i] = sim.particles[1+i_p].P#ps[i_p].P
                # 5.4.6.6: Update mean longitudes of pericenter.
                lam[i_p,i] = sim.particles[1+i_p].M+sim.particles[1+i_p].omega+sim.particles[1+i_p].Omega
                # 5.4.6.7: Update eccentricities.
                e[i_p,i] = sim.particles[1+i_p].e
                # 5.4.6.8: Update mean arguments of pericenter.
                omega[i_p,i] = sim.particles[1+i_p].omega
                # 5.4.6.9: Update longitudes of the ascending node.
                Omega[i_p,i] = sim.particles[1+i_p].Omega
        # 5.4.7.10: After every print_steps number of steps, print the time integrated for and semi-major axes, periods, and eccentricities.
        print_steps =  n_step/5 # after 20% of progress is made
        if i % print_steps == 0:
            print("\t* Integrated for "+str(i)+" steps ("+str(t)+ " years).")
            print("\t \t* Semi-Major Axes: ", str(list(a[:,i])))
            print("\t \t* Periods: ", str(list(per[:,i])))
            print("\t \t* Eccentricities: ", str(list(e[:,i])))

    # 5.4.10: Mark the completion of the integration.
    print("Integration completed after "+str(i+1)+" steps ("+str(t)+" years).")
    if t == T:
        txt_file.write(str(T)+",")
        txt_file.write(str(False)+",")

    # 5.5: Save simulation, call plot_save_print, and append the method used to generate initial conditions and a final column and newline marker.
    sim.save_to_file('save/Migration Runs '+ date+"/"+version+str(index)+'.bin')
    plot_save_print(times, a, omega, Omega, lam, per, e, random_pnum, version, date, txt_file, planet_count_max)
    txt_file.write(parameter_method+",")
    txt_file.write(str(1)+",\n")

    return close_encounter, random_pnum



# Stage 6: Run the code by defining a runit function with a variable number of simulations.
def runit(date, num_sim = 100):
        # 6.1: Remove previous file versions with the same name if present.
        if os.path.exists("save/Migration Runs "+ date+"/") == True:
            os.rmdir("save/Migration Runs "+ date+"/")
        if os.path.exists("plots/pdfs/Migration Runs "+ date+"/") == True:
            os.rmdir("plots/pdfs/Migration Runs "+ date+"/")
        if os.path.exists("plots/jpgs/Migration Runs "+ date+"/") == True:
            os.rmdir("plots/jpgs/Migration Runs "+ date+"/")
        if os.path.exists('results/Migration Runs '+date+".csv") == True:
            os.remove('results/Migration Runs '+date+".csv")
       
        # 6.2: Open the CSV file to save results, write the columns, and close it.
        txt_file = open("results/Migration Runs "+date+".csv", "a")
        txt_file.write("index,pnum,pl_orbsmax_0,pl_orbsmax_1,pl_orbsmax_2,pl_orbsmax_3,pl_orbsmax_4,pl_orbsmax_5,pl_orbsmax_6,pl_mass_0,pl_mass_1,pl_mass_2,pl_mass_3,pl_mass_4,pl_mass_5,pl_mass_6,st_mass,kep_angles,arg_peri_0,arg_peri_1,arg_peri_2,arg_peri_3,arg_peri_4,arg_peri_5,arg_peri_6,mean_anom_0,mean_anom_1,mean_anom_2,mean_anom_3,mean_anom_4,mean_anom_5,mean_anom_6,Sigma,alpha,beta,h,h_outermost,K,K_outermost,taue,taua,disk_inner_edge,disk_inner_edge_width,itime,close_encounter_flag,a_0,a_1,a_2,a_3,a_4,a_5,a_6,a_ratio_01,a_ratio_12,a_ratio_23,a_ratio_34,a_ratio_45,a_ratio_56,p_0,p_1,p_2,p_3,p_4,p_5,p_6,p_ratio_01,p_ratio_12,p_ratio_23,p_ratio_34,p_ratio_45,p_ratio_56,e_0,e_1,e_2,e_3,e_4,e_5,e_6,delta_wu_01,prox_res_01,delta_wu_12,prox_res_12,delta_wu_23,prox_res_23,delta_wu_34,prox_res_34,delta_wu_45,prox_res_45,delta_wu_56,prox_res_56,phi_cen_2_01,phi_amp_2_01,phi_cen_2_12,phi_amp_2_12,phi_cen_2_23,phi_amp_2_23,phi_cen_2_34,phi_amp_2_34,phi_cen_2_45,phi_amp_2_45,phi_cen_2_56,phi_amp_2_56,phi_cen_3_012,phi_amp_3_012,phi_cen_3_123,phi_amp_3_123,phi_cen_3_234,phi_amp_3_234,phi_cen_3_345,phi_amp_3_345,phi_cen_3_456,phi_amp_3_456,parameter_method,tmp\n")
        txt_file.close()

        # 6.3: Create folders for storing saves, pdfs, and jpgs.
        os.makedirs("save/Migration Runs "+ date+"/")
        os.makedirs("plots/jpgs/Migration Runs "+ date+"/")
        os.makedirs("plots/pdfs/Migration Runs "+ date+"/")

        # 6.3: Print results in the terminal and write to the CSV by calling run_migration_edge.
        for i in range(num_sim):
            # 6.3.1: Open the text file.
            txt_file = open('results/Migration Runs '+date+".csv", 'a')
            # 6.3.2: Print dividers between runs.
            if i>=1:
                print("\n\n")
            # 6.3.3: Print ruun index.
            print("Run "+str(i)+":")
            # 6.3.4: Initiate run_migration_edge.
            close_encounter, random_pnum = run_migration_edge(i, txt_file, general_simulation_parameters_v10(), "general_v10", date, integration_time="3tau_a", n_step = 10000)
            # 6.3.5: Integrate another planet of the same planet number if a close encounter occured.
            while_counter = 1
            while close_encounter == True:
                print("Run "+str(i)+"."+str(while_counter)+":")
                close_encounter, random_pnum = run_migration_edge(i, txt_file, general_simulation_parameters_v10(input_pnum = random_pnum), "general_v10", date, integration_time="3tau_a", n_step = 10000)
                while_counter+=1
                if while_counter>10:
                    break
            del close_encounter, while_counter
            # 6.3.6: Close the CSV.
            txt_file.close()

runit("test", 100)