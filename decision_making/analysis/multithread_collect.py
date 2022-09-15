#!/usr/bin/python

import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from multiprocessing import Pool, Lock

import json 

from planners import *
from envs import *

"""_summary_ This file is intended to run a user specified
        decision making algorithm from those available in config
        
    
"""

## CONFIG -------------------------------------------------
algo = sys.argv[1]
alg_config_file = sys.argv[2]
env_config_file = sys.argv[3]
if len(sys.argv) >= 5:
    num_cores = sys.argv[4]
else:
    num_cores = 1
if len(sys.argv) >= 6:
    fp = sys.argv[5]
else:
    fp = None

f = open(current + "../config/algorithms/" + alg_config_file +  ".json")
alg_config = json.load(f)
f = open(current + "../config/envs/" + env_config_file +  ".json")
env_config = json.load(f)   

# SETUP CONFIG -------------------

# ENVS
if env_config == "grid world":
    pass
elif env_config == "sailing":
    pass
elif env_config == "grid trap":
    pass


# ALGS
if env_config == "grid world":
    pass
elif env_config == "sailing":
    pass
elif env_config == "grid trap":
    pass

# PRINT CONFIG -------------------
    
print("Testing ", algo, " in ", env_name)
print("Rendering? ", do_render)
if fp != None:
    print("Saving output to ", fp)



## Evaluation -------------------------------------------------
def runWrapper(obj): 
    # Run each trial
    for t in range(num_time_steps):
        # Display map for current time step, if only one trial
        if config.enable_plots and num_threads == 1:
            displayMap(obj, plt, map_fig, map_ax)
            if save_plots == 1:
                map_fig.savefig("figures/fig%d.png" % t)
            print("\nt = {0}".format(t))

        if save_prev_exp:
            lock.acquire()
            prev_exp_data.record(obj, t)
            lock.release()

        terminal_condition = obj.simulationStep(t)
        if slow_mode and num_threads == 1:
            time.sleep(0.5)
        
        if ((t == num_time_steps - 1) or (enable_terminal_condition and terminal_condition)):
            # Save final step of prev exp if at final time step
            if save_prev_exp:
                lock.acquire()
                prev_exp_data.record(obj, t)
                lock.release()

            # Display final map if at final time step
            if config.enable_plots and num_threads == 1:
                displayMap(obj, plt, map_fig, map_ax)
                if save_plots == 1:
                    t = t+1
                    map_fig.savefig("figures/fig%d.png" % t)

        # End simulation early if terminal condition reached
        if enable_terminal_condition and terminal_condition:
            break

    return obj
    
def poolHandler():
    # Initialize worlds
    print("Initializing worlds...")
    sim_worlds = [World(i, food_layer, home_layer, obstacle_layer, robot_layer, robot_personality_list, perception_range, battery_size, heading_size, policy_filepath_list, v_filepath_list, q_filepath_list, arbitration_type_list, use_prev_exp, prev_exp_filepath, num_time_steps, heading_change_times, food_respawn, real_world_exp=False, manual_control=use_manual_control) for i in range(num_monte_carlo_trials)]
    
    # Run pool of Monte Carlo trials
    print("Beginning Monte Carlo trials...")
    if num_threads > 1:
        # Initialize multiprocessing pool and map objects' run methods to begin simulation
        p = Pool(num_threads)
        sim_worlds = p.map(runWrapper, sim_worlds)
    else:
        # Run each trial sequentially, for debugging purposes, with no multiprocessing
        for i in range(num_monte_carlo_trials):
            runWrapper(sim_worlds[i])
    print("Simulations complete.")

    # Save results
    saveResultsFile(results_filename, sim_worlds)
    if save_prev_exp:
        prev_exp_data.save(prev_exp_filepath)

if __name__=='__main__':
    poolHandler()

## Post Processing---------------------------------------------