# Part of Generalized Migration Runs v10 at the following repository: https://github.com/Sparro3hawk/Higher-Order-MMRs.git
# Written by Finnegan Keller 05/28/25
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

# 0.2: import codes needed for setting up initial conditions. These are hardcoded as local directories (no setup.py used).
import mr_forecast as mr # code to predict masses from planetary radii measurements
from Exoplanet_Archive_Examiner import makeframe, drop_systems_below_pnum # functions to pull from the Exoplanet Archive

# Let's clean up this code for future use. Currently, I'm going to make 1 dataset. We could call this every time for variation or just once at reread it in.
# Stage 1: Data Imports and Radius/Mass Thresholding
# 1.0: Create radius dataset. We only want to run this if the radius dataset does not already exist.
def setup_radius_dataset(planet_count_min = 3, planet_count_max = 7):
    PSC = pd.read_csv("Exoplanet Datasets/PSCompPars_2024.08.01_14.31.11.csv", header = 88)

    # 1.0.1: Gather all planets with stellar masses and radius measurements<4Rearth. We remove all systems with fewer than the minimum number of planets of interest.
    min_radius = 0.5
    max_radius = 4 #Mpost2R([30], unit='Earth')[0]
    planet_count_min = 3
    planet_count_max = 7
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

    # 1.0.4.3: Fit a lognormal distribution to the radius ratios for later sampling from. I think prior issues were wrong ordering of shape, loc, scale and loc!=0.
    pl_rade_shape, pl_rade_loc, pl_rade_scale = lognorm.fit(radius_ratios, floc=0)
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
            random_r_ratio_lognormal = lognorm.rvs(pl_rade_shape, pl_rade_loc, pl_rade_scale, size = number_of_planets_to_generate)
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

# 1.1: Write this dataset to a CSV.
# 1.1.0: Create the dataset,
potential_systems = setup_radius_dataset(planet_count_min = 3)
# 1.1.1: Write this dataset to a CSV.
potential_systems.to_csv("Exoplanet Datasets/Radius_Dataset.csv")

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
def general_simulation_parameters(input_pnum = "N/A"):
    # 2.5.0: Colate all simulation parameters.
    random_pnum, random_pl_masses, random_st_mass, random_pl_orbsmaxes = Keplerian_parameters(input_pnum = input_pnum)
    random_pl_orbeccens, random_pl_orbincs, random_pl_long_ascs, random_pl_arg_pericenters, random_pl_mean_anoms = Keplerian_angle_parameters(random_pnum)
    inner_edge, inner_edge_width, K, sd_0, alpha, beta = migration_parameters()
    minimum_itime, maximum_itime, timestep = integration_timeframe()
   
    # 2.5.1: Return all simulation parameters.
    return random_pnum, random_pl_masses, random_st_mass, random_pl_orbsmaxes, random_pl_orbeccens, random_pl_orbincs, random_pl_long_ascs, random_pl_arg_pericenters, random_pl_mean_anoms, inner_edge, inner_edge_width, K, sd_0, alpha, beta, minimum_itime, maximum_itime, timestep

# Run a random initialization. Other parameters can be found from these.
# All of these parameters are in a consistent unit system for integration (yr, Msun, AU) but are printed out in more interpretable units here.
random_pnum, random_pl_masses, random_st_mass, random_pl_orbsmaxes, random_pl_orbeccens, random_pl_orbincs, random_pl_long_ascs, random_pl_arg_pericenters, random_pl_mean_anoms, inner_edge, inner_edge_width, K, sd_0, alpha, beta, minimum_itime, maximum_itime, timestep = general_simulation_parameters()
print("Planet Multiplicity: "+str(int(random_pnum)))
print("Planet Masses: "+str((random_pl_masses*u.Msun).to(u.Mearth)))
print("Stellar Mass: "+str(random_st_mass*u.Msun))
print("Planet Semi-Major Axes: "+str(random_pl_orbsmaxes*u.AU))
print("Planet Eccentricities (Initialized as Circular): "+str(random_pl_orbeccens))
print("Planet Inclinations: (Initialized as Coplanar): "+str(random_pl_orbincs))
print("Planet Longitude of the Ascending Nodes (0 as Coplanar System): "+str(random_pl_long_ascs*u.deg))
print("Planet Argument of Pericenters: "+str(random_pl_arg_pericenters*u.deg))
print("Planet Mean Anomalies: "+str(random_pl_mean_anoms*u.deg))
# I have code to randomize inner edge based on stellar rotation periods if curious to add variation
print("Disk Inner Edge: "+str(inner_edge*u.AU))
print("Disk Inner Edge Width: "+str(inner_edge_width*u.AU))
print("K Factor: "+str(K))
print("Surface Density at 1 AU: "+str((sd_0*u.Msun/u.AU**2).to(u.g/u.cm**2)))
print("Surface Density Exponent: "+str(alpha))
print("Disk Flaring Index: "+str(beta))
print("Minimum Integration Time: "+str(minimum_itime*u.yr))
print("Maximum Integration Time: "+str(maximum_itime*u.yr))
print("Timestep: "+str(timestep)+"periods of test mass at inder disk edge ("+str(inner_edge*u.AU)+")")