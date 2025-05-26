
import os
import sys
import time
import argparse
import datetime
import torch 
import numpy as np
import matplotlib.pyplot as plt
import simpy
import pickle
from itertools import chain

# Import agent modules
import src.utils as u
from src.loss import *
from src.model import DAIF


# Improving debugging as there're different losses used for different modules
torch.autograd.set_detect_anomaly(True)


""" Settings """

seed = 0
horizon = 300
samples = 10
npf = False

if npf:
    from src.prod_environment_test_npf import System
    print('New preference function is used!')
else:
    from src.prod_environment_test_opf import System
    print('Old preference function is used!') 

checkpoint_path = './'
test_results_path = './'
calc_mean = False

latent_dim = 16
pi_dim = 7 
lambda_s = 1.5

lr_policy = 5e-04
lr_test_policy = lr_policy/5 # as it's fine-tuning

beta = 5 # of beta-vae

test_replications = 10
print_rem_interval = 5 * 24 * 60 * 60  # 5 days

o_dim = 10 + 1 + 6*5 + 3 
po_dim = o_dim  # all the times are removed in the observation and prediction
init_time = 1 * 24 * 60 * 60  # 1 days
response_time = 1 * 24 * 60 * 60  # 1 days is for response to the policy after random init 
test_time = 30 * 24 * 60 * 60  # 30 + 1 days (test_time_span is 30 days)
sim_interval = 10


""" Logging """

# recording prints in a txt
sys.stdout = u.Logger(f'{test_results_path}/test_output_random.txt')


""" Testing """

# Initialize a dictionary to store replication results
replication_stats = {
    "test_throughput_loss": [],
    "test_energy_saving": [],
    "test_improving_consum_part": [],
    "test_throughput": [],
    "test_consumption": [],
    "test_consum_part": [],
}
    
for replication in range(test_replications):

    """ Testing on an independent environment (planning with gradient every h step)"""

    # # model.eval()
    test_model = DAIF(o_dim=o_dim, latent_dims=latent_dim, action_dim=pi_dim, dropout_prob=0.05, samples=samples, calc_mean=calc_mean,lambda_s=lambda_s) # instantiating a new model

    # checkpoint = torch.load(checkpoint_path)

    # # Restore model and optimizer states
    # test_model.load_state_dict(checkpoint['model_state_dict'])


    # test_model.eval()
    # u.reset_recordings(test_model, batch_size=1)

    env_test=simpy.Environment()

    system_test = System(env=env_test, number_of_systems=1, dmodel=test_model, warmup=False)

    # systems warmup for 10 days, which also removes the profile
    system_test.warmup()

    # triggering decisions from the AIF module/algorithm rather than "ALL ON" of the system (module)
    system_test.dmodel_from_now()

    # systems' initialization with random agent; this will create a good preference start (i.e., ~0.67) to improve
    test_model.decision = 'random'
    env_test.run(until=env_test.now + init_time)

    test_model.decision = 'random'

    # # Train the policy (planner) based on the generative model
    # test_model.train_policy_freeze_generative()
    # optimizer_test_policy = torch.optim.Adam(test_model.policy.parameters(), lr=lr_test_policy)

    # # initial planning
    # o0 = system_test.systems[0].observation()
    # o0 = torch.from_numpy(o0).float().view(1,-1)
    # efe, efe_terms = test_model.efe(o0, calc_mean=calc_mean, samples=samples)
    # train_policy(test_model, optimizer_test_policy, efe)

    current_time = env_test.now

    env_test.run(until=current_time + response_time)

    print(f'Replication {replication} - Warm up and initial response time is now done -> Starting testing performance span. Time of the test env:',  env_test.now/(24*60*60))

    system_test.systems[0].start_test_stats() 

    current_time = env_test.now

    env_test.run(until=current_time + test_time)

    # creating a query form the last time window of the testing system
    # one test env batch -> its id is 0 then
    # rewards = system_test.systems[0].reward() # Returns: (r_prod, r_energy, composite_reward)
    performances = system_test.systems[0].performance_test() # Returns: (pd_throughput, pd_consumption, pd_energy_consumption_per_part)
    raw_performances = system_test.systems[0].raw_performance_test() # Returns: (throughput, avg_consumption, avg_energy_consumption_per_part)

    print(f"\nReplication {replication} - Storing Performance Metrics:")

    # Storing values from performance()
    replication_stats["test_throughput_loss"].append(performances[0])  # pd_throughput
    print(f"Replication {replication} - Test Throughput Loss (%): {performances[0]:.4f}")

    replication_stats["test_energy_saving"].append(performances[1])  # pd_consumption
    print(f"Replication {replication} - Test Energy Saving (%): {performances[1]:.4f}")

    replication_stats["test_improving_consum_part"].append(performances[2])  # pd_energy_consumption_per_part
    print(f"Replication {replication} - Test Improving Consum Part (%): {performances[2]:.4f}")

    # Storing values from raw_performance()
    replication_stats["test_throughput"].append(raw_performances[0])  # throughput
    print(f"Replication {replication} - Test Throughput (Part/Min): {raw_performances[0]:.4f}")

    replication_stats["test_consumption"].append(raw_performances[1])  # avg_consumption
    print(f"Replication {replication} - Test Consumption (KW): {raw_performances[1]:.4f}")

    replication_stats["test_consum_part"].append(raw_performances[2])  # avg_energy_consumption_per_part
    print(f"Replication {replication} - Test Consum Part (J): {raw_performances[2]:.4f} \n")


    # Compute statistics
metrics = {metric: u.compute_stats(data) for metric, data in replication_stats.items()}

# Print results
print(f"\nTest Performances of the Policy Checkpoint @ {checkpoint_path}:\n")
for metric, (mean, std, margin_error) in metrics.items():
    print(f"{metric}: {mean:.4f} ± {margin_error:.4f} (std: {std:.4f})")




