

from src.workstation import *

import math
import numpy as np
 
"Input Data"

# RANDOM_SEED = 42 # Random seed  # random seed is fixed in the train module

warm_up_days = 1 #10 #0.33 #10 # days # The warm-up period is imposed for each experiment, which is equal to 10 days of production period
warm_up_time = warm_up_days * 24 * 60 * 60  # Simulation time in seconds

NUM_WORKSTATION = 1  # Number of workstations in the Production system
NUM_MACHINES = [6]  # Number of machines in the workstation
BCAP = [10]  # Buffer Capacity in the workstation
ARRIVAL_MEAN = 20  # Avg. arrival time [sec] to the 1st workstation

# Machines data: (HP: Each workstation is characterized by identical machines)
PT_MEAN = [83.7]  # Avg. processing time [sec] of machines in the workstation
ST_MEAN = [30]  # Avg. startup time [sec] of machines in the workstation
MTTF = [3600]  # Avg. time to failure [sec] of machines in the workstation 
MTTR = [30]  # Time it takes to repair a machine [sec] in the workstation 
W_BUSY = [15]  # Consumption [kW] for busy state of machines in the workstation
W_IDLE = [9.30]  # Consumption [kW] for idle state of machines in the workstation
W_STARTUP = [10]  # Consumption [kW] for startup state of machines in the workstation
W_STANDBY = [0]  # Consumption [kW] for standby state of machines in each workstation

ALPHA = 0.97 # shouldn't be 1 or 0 as those need some adjustments in the code see 
reward_multiplier = 14 # based on the set point for 60 KW energy consumption while keeping the throughput

reward_time_span = 8 * 60 * 60  # in seconds

"""Production System is the Environment, it contains workstations and machines"""
class Prod_System:  # This is the environment with whom the agent interacts

    def __init__(self, env, dmodel, cap, num_ws, num_m, pw_busy, pw_idle, pw_startup, pw_standby, mttf, mttr, st, pt, alpha, policy):
    
        self.env = env  # This is the simulation environment
        self.parts_made = 0  # Nr of Parts Made
        self.discarded = 0  # Nr of Parts Discarded
        self.accepted = 0  # Nr of Parts Accepted in the system
        self.num_ws = num_ws  # Nr of WS
        self.num_m = num_m  # Nr of machines per ws
        self.bufcap = cap  # Buffer capacity
        self.mttf = mttf  # Data on Machines MTTF
        self.mttr = mttr  # Data on Machines MTTF
        self.st = st  # Data on Machines startup time
        self.pt = pt  # Data on Machines processing time
        self.consumption_busy = 0  # To measure consumption of the sys when busy
        self.consumption_idle = 0  # To measure consumption of the sys when idle
        self.consumption_startup = 0  # To measure consumption of the sys when startup
        self.consumption_standby = 0  # To measure consumption of the sys when standby
        self.consumption_tot = 0  # To measure consumption of the sys
        self.pw_busy = pw_busy  # Power request during busy state [kW]
        self.pw_idle = pw_idle  # Power request during idle state [kW]
        self.pw_startup = pw_startup  # Power request during startup state [kW]
        self.pw_standby = pw_standby  # Power request during standby state [kW]
        self.workstations = []  # Vector containing the objects "workstation of the system"
        self.policy = policy  # Policy for action
        self.alpha = alpha
        self.dmodel = dmodel # decision model

        self.o_dim = ((cap[0] + 1) + num_m[0]*(5) + 3)  # also observing composite_reward, r_prod, r_energy
        self.act_dim = (num_m[0] + 1)  # Dimension of action vector
        self.max_TH = []  # Helpful for compute reward function
        self.max_consum = []  # max consumption

        self.parts_made_buffer = {}  # for the new reward
        self.consumption_tot_buffer = {}  # for the new reward        

        self.warmup = False # for removing the warmup profile from the reward function; default False as it will become True if warmup() called from the parent class!
        self.warmup_time = 0 # for removing the warmup profile from the reward function; default 0 as it will be set after warmup() called from the parent class! 

        for i in range(self.num_ws):  # There is a "for" cycle because it considers when there are more WS, here it
            self.workstations.append(Last_Workstation(self.env, self, i, self.bufcap[i], self.num_m[i], self.pw_busy[i], self.pw_idle[i], self.pw_startup[i], self.pw_standby[i], self.mttf[i], self.mttr[i], self.st[i], self.pt[i]))

            max_prod = (60 / (self.pt[i] / (self.num_m[i]))) * (self.mttf[i]/(self.mttf[i] + self.mttr[i]))  # I need max_th for reward function
            max_arr = 60 / ARRIVAL_MEAN
            max_th = min(max_prod, max_arr)
            self.max_TH.append(max_th)

            max_consum = self.num_m[i] * max(W_BUSY[i], W_IDLE[i], W_STARTUP[i], W_STANDBY[i])
            self.max_consum.append(max_consum)

        self.process = env.process(self.arrival_process_system())  
        
        self.env.stop_after_event = False # it's false at the beginning

        self.batch_id = 'NA' # Not Assigned  


    def record_parts_made(self):

        timestamp = self.env.now
        self.parts_made_buffer[timestamp] = self.parts_made

        # checking to truncate the buffer in the interest of the memory and speed
        if min(self.parts_made_buffer.keys()) < timestamp - 1.1 * reward_time_span:
            self.truncate_buffer(self.parts_made_buffer)
    

    def record_consumption_tot(self):
        
        timestamp = self.env.now
        self.consumption_tot_buffer[timestamp] = self.consumption_tot

        # checking to truncate the buffer in the interest of the memory and speed
        if min(self.consumption_tot_buffer.keys()) < timestamp - 1.1 * reward_time_span:
            self.truncate_buffer(self.consumption_tot_buffer)


    def reward(self):

        """ old reward/preference function that considers consumption rate and throughput of a specified past period compared to the max theoretical ones. This specific function is linear response to throughput and average consumption"""    

        # throughput
        if len(self.parts_made_buffer)>1: # avoiding empty buffer
            
            # start_timestamp: closest timestamps for parts made, respecting reward time span
            start_parts_made, start_timestamp_parts_made = self.start_timestamp(self.parts_made_buffer)
            
            timespan_parts_made = self.env.now - start_timestamp_parts_made
            span_parts_made = self.parts_made - start_parts_made
            throughput = span_parts_made / (timespan_parts_made / 60)  # throughput is parts/min
            r_prod = throughput / sum(self.max_TH)  # sum() since this function calculates for the whole prod system 

            if r_prod > 1: r_prod = 1  # to avoid errors for aif

        else:
            r_prod = 0

        # consumption 
        if len(self.consumption_tot_buffer)>1: # avoiding empty buffer

            # start_timestamp: closest timestamps for total consumption, respecting reward time span
            start_consumption_tot, start_timestamp_consumption_tot = self.start_timestamp(self.consumption_tot_buffer)
            
            timespan_consumption_tot = self.env.now - start_timestamp_consumption_tot
            span_consumption_tot = self.consumption_tot - start_consumption_tot
            avg_consumption = span_consumption_tot / timespan_consumption_tot
            r_energy = 1 - (avg_consumption / sum(self.max_consum))  # sum() since this function calculates for the whole prod system
        
        else: 
            r_energy = 0
            
        composite_reward = ALPHA * r_prod + (1 - ALPHA) * r_energy

        return r_prod, r_energy, composite_reward


    def reward_npf(self):  

        # throughput
        if len(self.parts_made_buffer)>1: # avoiding empty buffer
            
            # start_timestamp: closest timestamps for parts made, respecting reward time span
            start_parts_made, start_timestamp_parts_made = self.start_timestamp(self.parts_made_buffer)
            
            timespan_parts_made = self.env.now - start_timestamp_parts_made
            span_parts_made = self.parts_made - start_parts_made
            throughput = span_parts_made / (timespan_parts_made / 60)  # throughput is parts/min
            r_prod = throughput / sum(self.max_TH)  # sum() since this function calculates for the whole prod system 

            if r_prod > 1: r_prod = 1  # to avoid errors for aif

        else:
            r_prod = 0

        # consumption 
        if len(self.consumption_tot_buffer)>1: # avoiding empty buffer

            # start_timestamp: closest timestamps for total consumption, respecting reward time span
            start_consumption_tot, start_timestamp_consumption_tot = self.start_timestamp(self.consumption_tot_buffer)
            
            timespan_consumption_tot = self.env.now - start_timestamp_consumption_tot
            span_consumption_tot = self.consumption_tot - start_consumption_tot
            avg_consumption = span_consumption_tot / timespan_consumption_tot
            r_energy = 1 - (avg_consumption / sum(self.max_consum))  # sum() since this function calculates for the whole prod system
        
        else: 
            r_energy = 0
            
        composite_reward = r_prod * (1 / (1 + np.exp(-reward_multiplier * r_energy)))

        return r_prod, r_energy, composite_reward.item()


    def raw_performance(self):
    
        # throughput
        if len(self.parts_made_buffer)>1: # avoiding empty buffer
            
            # start_timestamp: closest timestamps for parts made, respecting reward time span
            start_parts_made, start_timestamp_parts_made = self.start_timestamp(self.parts_made_buffer)
            timespan_parts_made = self.env.now - start_timestamp_parts_made
            span_parts_made = self.parts_made - start_parts_made
            throughput = span_parts_made / (timespan_parts_made / 60)  # throughput is parts/min
            # print(f"Throughput: {throughput} parts/min over {timespan_parts_made} seconds")
        else:
            throughput = 0

        # consumption 
        if len(self.consumption_tot_buffer)>1: # avoiding empty buffer

            # start_timestamp: closest timestamps for total consumption, respecting reward time span
            start_consumption_tot, start_timestamp_consumption_tot = self.start_timestamp(self.consumption_tot_buffer)
            timespan_consumption_tot = self.env.now - start_timestamp_consumption_tot
            span_consumption_tot = self.consumption_tot - start_consumption_tot
            avg_consumption = span_consumption_tot / timespan_consumption_tot
            # print(f"Avg Consumption: {avg_consumption} KW over {timespan_consumption_tot} seconds")
        else: 
            avg_consumption = 0
        
        # approximate average energy consumption per part (as the exact timespans and start/stops are not perfectly matched!)
        if len(self.parts_made_buffer) > 1 and len(self.consumption_tot_buffer) > 1 and span_parts_made != 0:  # Avoiding empty buffer  # span_parts_made != 0 as it can create numerical instability
            avg_energy_consumption_per_part = span_consumption_tot / span_parts_made
            # avg_energy_consumption_per_part = avg_consumption / span_parts_made  # this may remove some bias
            # print(f"Approximate Avg Energy Consumption per Part: {avg_energy_consumption_per_part}")
        else:
            avg_energy_consumption_per_part = 0

        return throughput, avg_consumption, avg_energy_consumption_per_part
    
    def performance(self, reference_throughput=3.0031190052525085, reference_consumption=79.3809907129650014,  reference_energy_consumption_per_part=1586.6565164873900358):
        # Calculate metrics
        throughput, avg_consumption, avg_energy_consumption_per_part = self.raw_performance()

        # Calculate percentage difference
        pd_throughput = ((reference_throughput - throughput) / reference_throughput) * 100
        pd_consumption = ((reference_consumption - avg_consumption) / reference_consumption) * 100
        pd_energy_consumption_per_part = ((reference_energy_consumption_per_part - avg_energy_consumption_per_part) / reference_energy_consumption_per_part) * 100

        # Print metrics
        # print("Throughput Loss:", pd_throughput, "%")
        # print("Energy Saving:", pd_consumption, "%")
        # print("Improvement (of Avg Energy Consumption per Part):", pd_energy_consumption_per_part, "%")

        return pd_throughput, pd_consumption, pd_energy_consumption_per_part 


    # Optimized start_timestamp function using binary search (efficient and only working as the list of keys of the buffer, i.e. times, are sorted!)
    def start_timestamp(self, data_buffer):

        target_timestamp = self.env.now - reward_time_span
        keys = list(data_buffer.keys())
        lo = 0
        hi = len(keys) - 1

        while lo <= hi:
            mid = (lo + hi) // 2
            if keys[mid] == target_timestamp:
                return data_buffer[target_timestamp], target_timestamp
            elif keys[mid] < target_timestamp:
                lo = mid + 1
            else:
                hi = mid - 1

        if lo == 0:
            closest_timestamp = keys[lo]
        elif lo == len(keys):
            closest_timestamp = keys[-1]
        else:
            if abs(keys[lo] - target_timestamp) < abs(keys[lo - 1] - target_timestamp):
                closest_timestamp = keys[lo]
            else:
                closest_timestamp = keys[lo - 1]

        return data_buffer[closest_timestamp], closest_timestamp


    def start_timestamp_left(self, data_buffer):
        """ This function prioritize closest timestamp on the left side of the target if existing"""
        target_timestamp = self.env.now - reward_time_span
        if target_timestamp < 0:
            target_timestamp = 0
        closest_timestamp = max((timestamp for timestamp in data_buffer.keys() if timestamp <= target_timestamp), default=None)
        if closest_timestamp is not None:
            return data_buffer[closest_timestamp], closest_timestamp
        else:
            closest_timestamp = min(data_buffer.keys(), key=lambda x: abs(x - target_timestamp))
            return data_buffer[closest_timestamp], closest_timestamp


    def truncate_buffer(self, data_buffer):
        threshold = self.env.now - (1.05 * reward_time_span)  # keeps within 20% of reward_time_span
        # threshold = self.env.now - (1.20 * reward_time_span)  # keeps within 20% of reward_time_span
        keys_to_delete = [timestamp for timestamp in data_buffer.keys() if timestamp < threshold]
        for key in keys_to_delete:
            del data_buffer[key]


    def observation(self):
        
        o = np.zeros(self.o_dim, dtype=np.float32)
        
        o[-3:] = self.reward()

        o[0:self.bufcap[0] + 1] = np.eye(self.bufcap[0] + 1)[self.workstations[0].buffer]

        for i in range(self.num_m[0]):

            o[self.bufcap[0] + 1 + i*5: self.bufcap[0] + 1 + i*5 + 5] = np.eye(5)[self.workstations[0].machines[i].state -1]

        return o
    
    def arrival_process_system(self):  # Arrival to the first WS

        while True:

            arrival = random.expovariate(1 / ARRIVAL_MEAN)  # Everything is exponential (this can change)

            while arrival > 0:

                if self.workstations[0].buffer < self.workstations[0].bufcap:  # If buffer has space -> accept the part

                    yield self.env.timeout(arrival)
                    self.process = self.workstations[0].arrival_ws()  # I call the arrival to the workstation
                    arrival = 0  # To go out of the while cycle

                else:  # If buffer has no space -> let's discard the part

                    self.discarded += 1
                    yield self.env.timeout(arrival)
                    self.process = self.workstations[0].discarded_ws()  # breaking the frozen state scenario # breaking_frozen_state()
                    arrival = 0  # To go out of the while cycle
                    
                    # frozen state scenario: 
                    # 1. if all are switched off 
                    # 2. buffer is full
                    # solution: calling decision

        
# random.seed(RANDOM_SEED)  # random seed is fixed in the train module

class System:

    def __init__(self, env, number_of_systems=1, dmodel=None, warmup=True, warmup_time=warm_up_time, reward_multiplier=None):

        self.env = env
        
        self.systems = [Prod_System(self.env, dmodel=dmodel, cap=BCAP, num_ws=NUM_WORKSTATION, num_m=NUM_MACHINES, pw_busy=W_BUSY, pw_idle=W_IDLE, pw_startup=W_STARTUP, pw_standby=W_STANDBY, mttf=MTTF, mttr=MTTR, st=ST_MEAN, pt=PT_MEAN, alpha=ALPHA, policy="Always_ON") for i in range(number_of_systems)]

        self.number_of_systems = number_of_systems
        
        for i in range(number_of_systems):
            self.systems[i].batch_id = i

        self.warmup_time = warmup_time
        if warmup: self.warmup() # Initialize all the systems with policy="Always_ON" for warmup following by removing the profile

    def dmodel_from_now(self):
        for i in range(self.number_of_systems):
            self.systems[i].policy = "Decision_Model"

    def all_on_from_now(self):
        for i in range(self.number_of_systems):
            self.systems[i].policy = "Always_ON"
        print("all-on_from_now")

    def warmup(self):
        """ 
        Initialize all the systems with policy="Always_ON" for warmup following by removing the profile; therefore the next observation is after warmup without its trace!
        """

        # adjusting the systems to remove warmup profile in their reward functions 
        for i in range(self.number_of_systems): 
            
            self.systems[i].warmup = True 
            self.systems[i].warmup_time = self.warmup_time
        
        # Initialize all the systems with policy="Always_ON" for warmup
        self.env.run(until=self.warmup_time)
        
        # Now removing the warmup profile from the workstation to be shown in the reward (prod_new, cons_new, cons_old, time) for each of the systems
        for i in range(self.number_of_systems): 
            
            # removing profile of the production system
            self.systems[i].parts_made = 0  # Nr of Parts Made
            self.systems[i].discarded = 0  # Nr of Parts Discarded
            self.systems[i].accepted = 0  # Nr of Parts Accepted in the system
            self.systems[i].consumption_busy = 0  # To measure consumption of the sys when busy
            self.systems[i].consumption_idle = 0  # To measure consumption of the sys when idle
            self.systems[i].consumption_startup = 0  # To measure consumption of the sys when startup
            self.systems[i].consumption_standby = 0  # To measure consumption of the sys when standby
            self.systems[i].consumption_tot = 0  # To measure consumption of the sys
            self.systems[i].parts_made_buffer = {}  # reinstating the buffer for the new reward
            self.systems[i].consumption_tot_buffer = {}  # reinstating for the buffer the new reward 
            
            # removing profile of each workstation
            for j in range(self.systems[i].num_ws): 

                self.systems[i].workstations[j].parts_made = 0  # Counters
                self.systems[i].workstations[j].discarded = 0
                self.systems[i].workstations[j].accepted = 0
                self.systems[i].workstations[j].consumption_busy = 0
                self.systems[i].workstations[j].consumption_idle = 0
                self.systems[i].workstations[j].consumption_startup = 0
                self.systems[i].workstations[j].consumption_standby = 0
                self.systems[i].workstations[j].consumption_tot = 0
                self.systems[i].workstations[j].prod_old = 0
                self.systems[i].workstations[j].prod_new = 0
                self.systems[i].workstations[j].cons_old = 0
                self.systems[i].workstations[j].cons_new = 0

                # removing profile of each machine
                for k in range(self.systems[i].num_m[0]):

                    self.systems[i].workstations[j].machines[k].parts_made_partial = 0  # I can count parts made for each Machine
                    self.systems[i].workstations[j].machines[k].failures = 0  # I can count nr of failures
                    self.systems[i].workstations[j].machines[k].repaired = 0
                    self.systems[i].workstations[j].machines[k].switched_on_times = 0
                    self.systems[i].workstations[j].machines[k].switched_off_times = 0
                    self.systems[i].workstations[j].machines[k].start_sby = 0  # useful for energy consumption calculation
                    self.systems[i].workstations[j].machines[k].finish_sby = 0  # useful for energy consumption calculation
                    self.systems[i].workstations[j].machines[k].start_idle = 0  # useful for energy consumption calculation
                    self.systems[i].workstations[j].machines[k].finish_idle = 0  # useful for energy consumption calculation
                    self.systems[i].workstations[j].machines[k].consumption_busy = 0  # useful for energy consumption calculation
                    self.systems[i].workstations[j].machines[k].consumption_idle = 0  # useful for energy consumption calculation
                    self.systems[i].workstations[j].machines[k].consumption_startup = 0  # useful for energy consumption calculation
                    self.systems[i].workstations[j].machines[k].consumption_standby = 0  # useful for energy consumption calculation
                    self.systems[i].workstations[j].machines[k].consumption_tot = 0   # useful for energy consumption calculation



# """" Printing Results for Testing """

# import simpy

# env = simpy.Environment()

# Systems = System(env=env,number_of_systems=1, dmodel=None, warmup=False)

# environment = Systems.systems[0]

# DAYS = 30.3 #11.3 #10.3 # 20.3 #10.001

# SIM_TIME = DAYS * 24 * 60 * 60 

# start_time = time.time()

# Systems.warmup()

# env.run(SIM_TIME/2)

# Systems.dmodel_from_now()

# env.run((2*SIM_TIME)/3)

# env.run(SIM_TIME)

# end_time = time.time()

# print('Running the simulation took', end_time - start_time, 'seconds!')
# print('Production system results after %s days' % DAYS)
# print("The production system has:")
# if environment.warmup:
#     print("A throughput of %0.2f [parts/min] and an average consumption of %0.2f [kJ/part]" % ((60 * environment.parts_made) / (SIM_TIME-environment.warmup_time), environment.consumption_tot/environment.parts_made))
# else:
#     print("A throughput of %0.2f [parts/min] and an average consumption of %0.2f [kJ/part]" % ((60 * environment.parts_made) / SIM_TIME, environment.consumption_tot/environment.parts_made))
# print("A total production of %d [parts] and a total consumption of %0.2f [MJ]" % (environment.parts_made, environment.consumption_tot*0.001))
# print('System Reward is:', environment.reward())
# print('System Observation is:', environment.observation())