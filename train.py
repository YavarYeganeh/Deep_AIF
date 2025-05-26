
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

parser = argparse.ArgumentParser(description='Training script.')
parser.add_argument('--seed', type=int, default=0, help="Random seed")
parser.add_argument('-b', '--batch', type=int, default=1, help='Select batch size.')
parser.add_argument('--horizon', type=int, default=300, help='Horizon for the transition') 
parser.add_argument('--samples', type=int, default=10, help='How many samples should be used to calculate EFE') 
parser.add_argument('--calc_mean', action='store_true', help='Whether mean should be considered during calculation of EFE')
parser.add_argument('--num_threads', type=int, default=30, help="Number of threads to use (only for CPU)")
parser.add_argument('--reward_multiplier', type=int, default=20, help="Reward multiplier for the production reward")
parser.add_argument('--replay_policy_update', action='store_true', help= "To also use replay scenarios during the policy planning when training (motivating learning a generic policy!)")
parser.add_argument('--npf', action='store_true', help= "To employ the new preference function that features sigmoid scaling of the energy-saving element!")
args = parser.parse_args()

if args.npf:
    from src.prod_environment_npf_optimized import System
    print('New preference function with optimized prod code is used!')
else:
    from src.prod_environment_opf import System
    print('Old preference function is used!') 

latent_dim = 16
pi_dim = 7 
lambda_s = 1.5
lr_model = 1e-05
lr_policy = 5e-05
lr_test_policy = lr_policy/5 # as it's fine-tuning
beta = 5 # of beta-vae
epochs = 100 #400
ROUNDS = 100 #1000
replay_capacity = 32
replay_size = 16
test_frequency = 1
test_replications = 3
record_frequency = 1
saving_model_threshold = 0.5
init_all_on = True # to help the training as it the beginning of the training the policy is random and rewards would not change ...
init_all_on_epochs = 10

o_dim = 10 + 1 + 6*5 + 3 
po_dim = o_dim  # all the times are removed in the observation and prediction
init_time = 1 * 24 * 60 * 60  # 1 days
test_time = 1 * 24 * 60 * 60  # 1 days
sim_interval = 10

torch.set_num_threads(args.num_threads)
os.environ["OMP_NUM_THREADS"] = str(args.num_threads)
os.environ["MKL_NUM_THREADS"] = str(args.num_threads)


""" Results Folder """

signature = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
signature += '_' + str(args.seed) + '_' + str(args.horizon) + '_' + str(args.samples) + '_' + str(latent_dim) + '_' + str(lambda_s) + '_' + str(args.calc_mean) + '_' + str(args.batch) + '_' + str(lr_model) + '_' + str(lr_policy) + '_' + str(ROUNDS) + '_' + str(args.replay_policy_update) + '_' + str(args.npf) + '_' + str(args.num_threads)
signature = signature.replace('.', '-')
folder = './results'
folder_results = folder + '/' + signature
folder_chp = folder_results + '/checkpoints'

try: os.mkdir(folder)
except: print('Folder already exists!')
try: os.mkdir(folder_results)
except: print('Folder results already exists!')
try: os.mkdir(folder_chp)
except: print('Folder chp already exists!')


""" Logging """

# recording prints in a txt
sys.stdout = u.Logger(f'{folder_results}/training_log.txt')


""" Reproducibility """

# set random seeds for reproducibility
u.set_seed(args.seed)


""" Model Instantiation """

model = DAIF(o_dim=o_dim, latent_dims=latent_dim, action_dim=pi_dim, dropout_prob=0.1, samples=args.samples, calc_mean=args.calc_mean, lambda_s=lambda_s) 
replay_buffer = u.ReplayBuffer(capacity=replay_capacity)


""" Recording Stats """

# training performance will be recorded here; EFE will also be retrieved from the the model parameters
stats_start = {
    'l_transition': {},
    'l_encoder_decoder': {},
    'mae_reward': {},
    'loss_reward': {},
    'bce_b': {},
    'bce_m': {},
    'efe': {},
    'efe_term1': {},
    'efe_term2': {},
    'efe_term3': {},
    'training_reward': {},
    'test_reward': {},
    'test_reward_energy': {},
    'test_reward_prod': {},
    'test_reward_average': {},
    'test_throughput_loss': {},
    'test_energy_saving': {},
    'test_improving_consum_part': {},
    'test_throughput': {},
    'test_consumption': {},
    'test_consum_part': {},
    'best_test': {}
}
stats = stats_start


""" Training """

# defining optimizers
optimizer_policy = torch.optim.Adam(model.policy.parameters(), lr=lr_policy)
optimizer_transition = torch.optim.Adam(model.transition.parameters(), lr=lr_model)
optimizer_encoder_decoder = torch.optim.Adam(chain(model.encoder.parameters(), model.decoder.parameters()), lr=lr_model) # Combine parameters of the encoder and decoder into a single iterable

# for training loop
best_test_reward = 0
best_test_epoch = 0
start_epoch = 1

start_time = time.time()
# timer = u.Timer()
for epoch in range(start_epoch, epochs + 1):

    u.reset_recordings(model, batch_size=args.batch)

    # Stats epoch init
    stats['l_transition'][epoch] = []
    stats['l_encoder_decoder'][epoch] = []
    stats['mae_reward'][epoch] = []
    stats['loss_reward'][epoch] = [] 
    stats['bce_b'][epoch] = []
    stats['bce_m'][epoch] = []
    stats['efe'][epoch] = []
    stats['efe_term1'][epoch] = []
    stats['efe_term2'][epoch] = []
    stats['efe_term3'][epoch] = []
    stats['training_reward'][epoch] = []
    stats['test_reward'][epoch] = []
    stats['test_reward_energy'][epoch] = []
    stats['test_reward_prod'][epoch] = []
    stats['test_reward_average'][epoch] = []
    stats['test_throughput_loss'][epoch] = []
    stats['test_energy_saving'][epoch] = []
    stats['test_improving_consum_part'][epoch] = []
    stats['test_throughput'][epoch] = []
    stats['test_consumption'][epoch] = []
    stats['test_consum_part'][epoch] = []

    env=simpy.Environment()
    system = System(env=env, number_of_systems=args.batch, dmodel=model, warmup=False, reward_multiplier=args.reward_multiplier)

    # systems warmup, which also removes the profile
    system.warmup()  
    # print("warm up done!")
    # timer.elapsed() 

    # triggering decisions from the AIF module/algorithm rather than "ALL ON" of the system (module)
    system.dmodel_from_now()

    if init_all_on and epoch < init_all_on_epochs:
        # systems' initialization with all-on policy (in contrast to random) helps having changes in the reward/performance for model be trained with untrained policy
        model.decision = 'all-on'
        print (f"Epoch {epoch}: Starting All-ON init for the training environment!")
        env.run(until=env.now + init_time)

    else:
        # systems' initialization with random agent; this will create a good preference start (i.e., ~0.67) to improve
        model.decision = 'random'
        print (f"Epoch {epoch}: Starting random init for the training environment!")
        env.run(until=env.now + init_time)

    print(f'Init is now done!')

    model.decision = 'aif'
    print(f'Decisions are now based on the {model.decision} policy!')
    # timer.elapsed()
    
    # pre-filling of the replay buffer with the model (which is random at this beginning)
    while replay_buffer.size() < replay_size:
        o0, ref, oh, r = u.batch_observe(env, model, sim_interval, batch_size=args.batch, steps=args.horizon)     
        replay_buffer.record_interaction(o0, ref, oh, r)

    # untrained testing and stat recording 
    if epoch == 1:

        epoch = 0  # after recording performance of untrained agent (index 0) it makes the the epoch 1
        stats['test_reward'][epoch] = []
        stats['test_reward_energy'][epoch] = []
        stats['test_reward_prod'][epoch] = []
        stats['test_reward_average'][epoch] = []
        stats['test_throughput_loss'][epoch] = []
        stats['test_energy_saving'][epoch] = []
        stats['test_improving_consum_part'][epoch] = []
        stats['test_throughput'][epoch] = []
        stats['test_consumption'][epoch] = []
        stats['test_consum_part'][epoch] = []

        for replication in range(test_replications):

            model.eval()
            test_model = DAIF(o_dim=o_dim, latent_dims=latent_dim, action_dim=pi_dim, dropout_prob=0.1, samples=args.samples, calc_mean=args.calc_mean, lambda_s=lambda_s) # instantiating a new model
            test_model.load_state_dict(model.state_dict()) # creating an independent model copy for testing based on the latest trained model as the policy 
            u.reset_recordings(test_model, batch_size=1)

            env_test=simpy.Environment()

            system_test = System(env=env_test, number_of_systems=1, dmodel=test_model, warmup=False, reward_multiplier=args.reward_multiplier)

            # systems warmup, which also removes the profile
            system_test.warmup()

            # triggering decisions from the AIF module/algorithm rather than "ALL ON" of the system (module)
            system_test.dmodel_from_now()

            # systems' initialization with random agent; this will create a good preference start (i.e., ~0.67) to improve
            test_model.decision = 'random'
            env_test.run(until=env_test.now + init_time)

            test_model.decision = 'aif'

            print(f'Initial testing  - replication {replication} (untrained policy without planning)')

            env_test.run(until = env_test.now + test_time)
            
            # creating a query form the last time window of the testing system
            # one test env batch -> its id is 0 then
            rewards = system_test.systems[0].reward() # Returns: (r_prod, r_energy, composite_reward)
            performances = system_test.systems[0].performance() # Returns: (pd_throughput, pd_consumption, pd_energy_consumption_per_part)
            raw_performances = system_test.systems[0].raw_performance() # Returns: (throughput, avg_consumption, avg_energy_consumption_per_part)

            print(f'Testing untrained agent - replication {replication}: Rewards -> {rewards} - Performances -> {performances} - Raw Performances -> {raw_performances}')
            print('Please note the aforementioned testing was done without policy gradient/planning steps during it!')

            stats['test_reward'][epoch].append(rewards[-1])  # composite_reward
            stats['test_reward_energy'][epoch].append(rewards[-2])  # r_energy
            stats['test_reward_prod'][epoch].append(rewards[-3])  # r_prod
            stats['test_reward_average'][epoch].append(
                torch.cat(test_model.o_i[0], dim=0)
                    .view(-1, o_dim)[:, -1]
                    .mean()
                    .item()
            )

            # storing values from performance()  
            stats['test_throughput_loss'][epoch].append(performances[0])  # pd_throughput
            stats['test_energy_saving'][epoch].append(performances[1])  # pd_consumption
            stats['test_improving_consum_part'][epoch].append(performances[2])  # pd_energy_consumption_per_part

            # storing values from raw_performance()
            stats['test_throughput'][epoch].append(raw_performances[0])  # throughput
            stats['test_consumption'][epoch].append(raw_performances[1])  # avg_consumption
            stats['test_consum_part'][epoch].append(raw_performances[2])  # avg_energy_consumption_per_part

        if np.mean(stats['test_reward'][epoch]) > best_test_reward:
    
            best_test_reward = np.mean(stats['test_reward'][epoch])
            best_test_epoch = epoch
            stats['best_test'][epoch] = epoch
            
        stats['best_test'][epoch] = best_test_epoch
       
        # saving the initial stats dictionary for further analysis during training
        stats_path = folder_results + '/stats_epoch_' + str(epoch) + '.pkl'

        # Save the dictionary to a binary file using pickle
        with open(stats_path, 'wb') as file:
            pickle.dump(stats, file)

        epoch = 1  # now this make the epoch 1 for the first training

    print(f"Starting training rounds!")

    for round in range(ROUNDS):

        # print(f"Starting round {round} in Epoch {epoch}")
        # timer.elapsed()

        """" 
        This part of the code is responsible for having a new observation for all the systems in the batch
        we check whether i_th element of observation is obtained for all
        sim_interval is better to be small to prevent having more than one observation in either case before update
        """

        model.train()

        o0, ref, oh, r = u.batch_observe(env, model, sim_interval, batch_size=args.batch, steps=args.horizon) # obtaining new observation for all systems in the batch

        batch = replay_buffer.sample(replay_size - 1)
        o0b, refb, ohb, rb = map(list, zip(*batch))
        
        # Ensure the last interaction is added
        o0b.append(o0)
        refb.append(ref)
        ohb.append(oh)
        rb.append(r)

        # Record the new interaction in the replay buffer
        replay_buffer.record_interaction(o0, ref, oh, r)

        # Convert lists to torch tensors and reshape for training
        update_size = args.batch * replay_size
        # o0b, ohb, and rb are now lists of numpy arrays, while refb is a list of tensors
        o0b = torch.tensor(np.vstack(o0b), dtype=torch.float32).reshape(update_size, -1)
        refb = torch.cat(refb, dim=0).reshape(update_size, -1)
        ohb = torch.tensor(np.vstack(ohb), dtype=torch.float32).reshape(update_size, -1)
        rb = torch.tensor(np.vstack(rb), dtype=torch.float32).reshape(update_size, -1)

        # print("observed done!")
        # timer.elapsed()

        # Train the generative model 
        model.freeze_policy_train_generative()
        refb_detached = refb.detach()  # We need the tensor as a static input and don't want the gradients of subsequent computations to affect the part of the model that produced it during this training step
        l_transition, l_encoder_decoder, mae_reward, loss_reward, bce_b, bce_m = train_generative(model, optimizer_transition, optimizer_encoder_decoder, beta, o0b, refb_detached, ohb)

        # print(l_transition, l_encoder_decoder, mae_reward, loss_reward, bce_b, bce_m)
        # print("Trained the model!")
        # timer.elapsed()

        # Train the policy based on the generative model (also model, including policy, is still in the training mode!)
        model.train_policy_freeze_generative() 

        if args.calc_mean == True:
            raise("Still -> args.calc_mean == True <- is to be implemented for the EFE")
        
        if args.replay_policy_update:
            efe, efe_terms = model.efe(ohb, calc_mean=args.calc_mean, samples=args.samples) 
        else:
            efe, efe_terms = model.efe(oh, calc_mean=args.calc_mean, samples=args.samples)
        
        # print(efe, efe_terms)

        train_policy(model, optimizer_policy, efe)

        # print("Trained the policy!")
        # timer.elapsed()

        #train-level stats
        stats['l_transition'][epoch].append(l_transition)
        stats['l_encoder_decoder'][epoch].append(l_encoder_decoder)
        stats['mae_reward'][epoch].append(mae_reward)
        stats['loss_reward'][epoch].append(loss_reward)
        stats['bce_b'][epoch].append(bce_b)
        stats['bce_m'][epoch].append(bce_m)
        stats['efe'][epoch].append(efe.mean().item())
        stats['efe_term1'][epoch].append(efe_terms[0].mean().item())
        stats['efe_term2'][epoch].append(efe_terms[1].mean().item())
        stats['efe_term3'][epoch].append(efe_terms[2].mean().item())
        stats['training_reward'][epoch].append(r.mean())  # r is coming from a batch of systems being used


    """ Testing """

    if epoch % test_frequency == 0: 
            
        for replication in range(test_replications):

            """ Testing on an independent environment (planning with gradient every h step)"""

            model.eval()
            test_model = DAIF(o_dim=o_dim, latent_dims=latent_dim, action_dim=pi_dim, dropout_prob=0.1, samples=args.samples, calc_mean=args.calc_mean,lambda_s=lambda_s) # instantiating a new model
            test_model.load_state_dict(model.state_dict()) # creating an independent model copy for testing based on the latest trained model as the policy 
            test_model.eval()
            u.reset_recordings(test_model, batch_size=1)

            env_test=simpy.Environment()

            system_test = System(env=env_test, number_of_systems=1, dmodel=test_model, warmup=False, reward_multiplier=args.reward_multiplier)

            # systems warmup, which also removes the profile
            system_test.warmup()

            # triggering decisions from the AIF module/algorithm rather than "ALL ON" of the system (module)
            system_test.dmodel_from_now()

            # systems' initialization with random agent; this will create a good preference start (i.e., ~0.67) to improve
            test_model.decision = 'random'
            env_test.run(until=env_test.now + init_time)

            test_model.decision = 'aif'

            # Train the policy (planner) based on the generative model
            test_model.train_policy_freeze_generative()
            optimizer_test_policy = torch.optim.Adam(test_model.policy.parameters(), lr=lr_test_policy)

            # initial planning
            o0 = system_test.systems[0].observation()
            o0 = torch.from_numpy(o0).float().view(1,-1)
            efe, efe_terms = test_model.efe(o0, calc_mean=args.calc_mean, samples=args.samples)
            train_policy(test_model, optimizer_test_policy, efe)

            current_time = env_test.now

            while env_test.now < (current_time + test_time):

                o0, ref, oh, r = u.batch_observe(env_test, test_model, sim_interval, batch_size=1, steps=args.horizon) # simulating h steps

                # planning
                efe, efe_terms = test_model.efe(oh, calc_mean=args.calc_mean, samples=args.samples)
                train_policy(test_model, optimizer_test_policy, efe)

            # creating a query form the last time window of the testing system
            # one test env batch -> its id is 0 then
            rewards = system_test.systems[0].reward() # Returns: (r_prod, r_energy, composite_reward)
            performances = system_test.systems[0].performance() # Returns: (pd_throughput, pd_consumption, pd_energy_consumption_per_part)
            raw_performances = system_test.systems[0].raw_performance() # Returns: (throughput, avg_consumption, avg_energy_consumption_per_part)

            print(f'Testing in epoch {epoch} - replication {replication}: Rewards -> {rewards} - Performances -> {performances} - Raw Performances -> {raw_performances}')
            print('Please note the aforementioned testing was done with policy gradient/planning steps during it!')

            stats['test_reward'][epoch].append(rewards[-1])  # composite_reward
            stats['test_reward_energy'][epoch].append(rewards[-2])  # r_energy
            stats['test_reward_prod'][epoch].append(rewards[-3])  # r_prod
            stats['test_reward_average'][epoch].append(
                torch.cat(test_model.o_i[0], dim=0)
                    .view(-1, o_dim)[:, -1]
                    .mean()
                    .item()
            )

            # storing values from performance()  
            stats['test_throughput_loss'][epoch].append(performances[0])  # pd_throughput
            stats['test_energy_saving'][epoch].append(performances[1])  # pd_consumption
            stats['test_improving_consum_part'][epoch].append(performances[2])  # pd_energy_consumption_per_part

            # storing values from raw_performance()
            stats['test_throughput'][epoch].append(raw_performances[0])  # throughput
            stats['test_consumption'][epoch].append(raw_performances[1])  # avg_consumption
            stats['test_consum_part'][epoch].append(raw_performances[2])  # avg_energy_consumption_per_part


        # printing the epoch performance
        metrics_data = {
            "Test Reward": stats['test_reward'][epoch],
            "Test Reward Avg": stats['test_reward_average'][epoch],
            "Test Throughput Loss (%)": stats['test_throughput_loss'][epoch],
            "Test Energy Saving (%)": stats['test_energy_saving'][epoch],
            "Test Improving Consum Part (%)": stats['test_improving_consum_part'][epoch],
            "Test Throughput (Part/Min)": stats['test_throughput'][epoch], 
            "Test Consumption (KW)": stats['test_consumption'][epoch],  
            "Test Consum Part (J)": stats['test_consum_part'][epoch], 
        }

        # Compute statistics
        metrics = {metric: u.compute_stats(data) for metric, data in metrics_data.items()}

        # Print results
        print(f"\nPerformances in epoch {epoch}:\n")
        for metric, (mean, std, margin_error) in metrics.items():
            print(f"{metric}: {mean:.4f} ± {margin_error:.4f} (std: {std:.4f})")


        # recoding the best reward
        if np.mean(stats['test_reward'][epoch]) > best_test_reward:

            best_test_reward = np.mean(stats['test_reward'][epoch])
            best_test_epoch = epoch
            stats['best_test'][epoch] = epoch
            
        stats['best_test'][epoch] = best_test_epoch

        u.reset_recordings(model, batch_size=args.batch)


    """ Recording Stats """

    if epoch % record_frequency == 0:

        # save the stats dictionary for further analysis during training
        # stats_path = folder_results + '/stats_epoch_' + str(epoch) + '.pkl'
        stats_path = folder_results + '/stats_latest.pkl' # to prevent too much disk use

        # save the dictionary to a binary file using pickle
        with open(stats_path, 'wb') as file:
            pickle.dump(stats, file)


    """ Saving the model checkpoint """

    if np.mean(stats['test_reward'][epoch]) > saving_model_threshold:

        checkpoint = {
            'epoch': epoch,  
            'performance': np.mean(stats['test_reward'][epoch]),
            'model_state_dict': model.state_dict()
        }

        torch.save(checkpoint, f'{folder_chp}/checkpoint_epoch_{epoch}.pth')

        print(f'Saved checkpoint at {folder_chp}/checkpoint_epoch_{epoch}.pth with average reward:', np.mean(stats['test_reward'][epoch]))


# save the stats dictionary for further analysis 
stats_path = folder_results + '/stats_final.pkl'

# save the dictionary to a binary file using pickle
with open(stats_path, 'wb') as file:
    pickle.dump(stats, file)
