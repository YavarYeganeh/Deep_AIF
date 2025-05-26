
from src.machine import *
import src.utils as u

class Last_Workstation(object):  # Last workstation -> Easier because never blocked (no check on next buffer)

    def __init__(self, env, prod_system, id, cap, num_mach, pw_busy, pw_idle, pw_startup, pw_standby, mttf, mttr, st, pt):

        self.env = env  # This a recall of the simulation environment
        self.id = id  # Workstation ID
        self.name = id + 1  # Workstation Name (ID + 1, Python starts enumerating from 0)
        self.prod_system = prod_system
        self.parts_made = 0  # Counters
        self.discarded = 0
        self.accepted = 0
        self.num_mach = num_mach  # Parameters
        self.mttf = mttf
        self.mttr = mttr
        self.st = st
        self.pt = pt
        self.buffer = 0  # Counts buffer level
        self.bufcap = cap
        self.consumption_busy = 0
        self.consumption_idle = 0
        self.consumption_startup = 0
        self.consumption_standby = 0
        self.consumption_tot = 0
        self.pw_busy = pw_busy  # Power request during busy state [kW]
        self.pw_idle = pw_idle  # Power request during idle state [kW]
        self.pw_startup = pw_startup  # Power request during startup state [kW]
        self.pw_standby = pw_standby  # Power request during standby state [kW]
        self.machines = []  # Vector containing the objects "machines in the workstation"
        self.tobe_active = 0  # How many machines should be active in this ws?
        self.choice = -1  # Which machine should start processing the next part?
        
        # For the reward function
        self.prod_old = 0
        self.prod_new = 0
        self.cons_old = 0
        self.cons_new = 0
        self.counter = 0

        for i in range(self.num_mach):  # I populate the vector of the machines (i create them)

            self.machines.append(Last_Machine(self.env, i, self.prod_system, self, self.pw_busy, self.pw_idle, self.pw_startup, self.pw_standby, self.mttf, self.mttr, self.st, self.pt))


    def arrival_ws(self):  # What happens when a part arrives

        self.buffer += 1


        # decision step
        
        self.prod_new = self.parts_made
        self.cons_new = self.consumption_tot    
        
        if self.prod_system.policy == "Always_ON":
    
            self.tobe_active = self.num_mach 

        elif self.prod_system.policy == "Decision_Model":
            
            self.tobe_active = u.prod_planner(self.prod_system.observation(), self.prod_system.batch_id, self.prod_system.dmodel)
        
        else:
            
            raise ValueError(f"The selected policy ({self.prod_system.policy}) is not defined!")
        
        self.prod_old = self.prod_new
        self.cons_old = self.cons_new


        # Switch on/off Procedure
        for i in range(self.tobe_active, self.num_mach):  # I switch off these machines

            self.process = self.machines[i].switch_off()

        for i in range(self.tobe_active):  # I use/switch on these machines

            self.process = self.machines[i].select()  # I select the machine that should process the next part

            if self.choice > -1:
                self.accepted += 1
                self.process = self.env.process(self.machines[i].working())  # We call the processing function


    def departure_last(self):  # I have to check again the buffer level when I have a departure! Only useful with 1 worsktation
        # because otherwise I would have that departure = arrival to next ws (i.e. you perform a check on buffer level)

        # decision step
        
        self.prod_new = self.parts_made
        self.cons_new = self.consumption_tot    
        
        if self.prod_system.policy == "Always_ON":
    
            self.tobe_active = self.num_mach 

        elif self.prod_system.policy == "Decision_Model":
            
            self.tobe_active = u.prod_planner(self.prod_system.observation(), self.prod_system.batch_id, self.prod_system.dmodel)
        
        else:
            
            raise ValueError(f"The selected policy ({self.prod_system.policy}) is not defined!")
        
        self.prod_old = self.prod_new
        self.cons_old = self.cons_new 

        for i in range(self.tobe_active, self.num_mach):

            self.process = self.machines[i].switch_off()


    def discarded_ws(self):  # What happens when a part arrives   # self.tobe_active 

        # decision step
        
        self.prod_new = self.parts_made
        self.cons_new = self.consumption_tot    
        
        if self.prod_system.policy == "Always_ON":
    
            self.tobe_active = self.num_mach 

        elif self.prod_system.policy == "Decision_Model":

            self.tobe_active = u.prod_planner(self.prod_system.observation(), self.prod_system.batch_id, self.prod_system.dmodel)
        
        else:
            
            raise ValueError(f"The selected policy ({self.prod_system.policy}) is not defined!")
        
        self.prod_old = self.prod_new
        self.cons_old = self.cons_new

        # Switch on/off Procedure
        for i in range(self.tobe_active, self.num_mach):  # I switch off these machines

            self.process = self.machines[i].switch_off()

        for i in range(self.tobe_active):  # I use/switch on these machines

            self.process = self.machines[i].select()  # I select the machine that should process the next part

            if self.choice > -1:
                self.accepted += 1
                self.process = self.env.process(self.machines[i].working())  # We call the processing function

