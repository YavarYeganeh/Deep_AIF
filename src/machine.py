import random

import src.utils as u

class Last_Machine(object):

    def __init__(self, env, id, prod_system, workstation, pw_busy, pw_idle, pw_startup, pw_standby, mttf, mttr, st, pt):

        self.env = env
        self.name = id + 1
        self.id = id
        self.workstation = workstation
        self.prod_system = prod_system
        self.parts_made_partial = 0  # I can count parts made for each Machine
        self.failures = 0  # I can count nr of failures
        self.repaired = 0
        self.switched_on_times = 0
        self.switched_off_times = 0
        self.mttf = mttf
        self.mttr = mttr
        self.st = st
        self.pt = pt
        self.remaining = random.expovariate(1 / self.mttf)  # Remaining time before failure
        self.state = 1  # 1 = standby, 2 = startup, 3 = idle, 4 = busy, 5 = failed
        self.start_sby = 0  # useful for energy consumption calculation
        self.finish_sby = 0  # useful for energy consumption calculation
        self.start_idle = 0  # useful for energy consumption calculation
        self.finish_idle = 0  # useful for energy consumption calculation
        self.to_switch_off = False  # True when I have to switch off the machine
        self.consumption_busy = 0  # useful for energy consumption calculation
        self.consumption_idle = 0  # useful for energy consumption calculation
        self.consumption_startup = 0  # useful for energy consumption calculation
        self.consumption_standby = 0  # useful for energy consumption calculation
        self.consumption_tot = 0   # useful for energy consumption calculation
        self.pw_busy = pw_busy  # Power request during busy state [kW]
        self.pw_idle = pw_idle  # Power request during idle state [kW]
        self.pw_startup = pw_startup  # Power request during startup state [kW]
        self.pw_standby = pw_standby  # Power request during standby state [kW]

    def select(self):  # I check with this if, when a part arrives to the workstation, this machine can work the part
        # i.e. if this machine can be selected among the ones in parallel

        while True:

            if self.state == 3 or self.state == 1:  # I select this machine only when in standby or idle
                self.workstation.choice = self.id
                break
            else:
                self.workstation.choice = -1
                break

    def switch_off(self):

        if self.state == 1:
            self.to_switch_off = False

        if self.state == 2:
            self.to_switch_off = True

        if self.state == 4:
            self.to_switch_off = True

        if self.state == 3:
            self.state = 1
            self.finish_idle = self.env.now
            self.consumption_idle += (self.finish_idle - self.start_idle) * self.pw_idle
            self.consumption_tot += (self.finish_idle - self.start_idle) * self.pw_idle
            self.workstation.consumption_idle += (self.finish_idle - self.start_idle) * self.pw_idle
            self.workstation.consumption_tot += (self.finish_idle - self.start_idle) * self.pw_idle
            self.prod_system.consumption_idle += (self.finish_idle - self.start_idle) * self.pw_idle
            self.prod_system.consumption_tot += (self.finish_idle - self.start_idle) * self.pw_idle
            self.prod_system.record_consumption_tot()
            self.start_sby = self.env.now
            self.switched_off_times += 1

    def working(self):

        while True:

            if self.state == 1:  # I first switch it on and then I work

                startup = random.expovariate(1 / self.st)
                self.finish_sby = self.env.now
                self.consumption_standby += (self.finish_sby - self.start_sby) * self.pw_standby
                self.consumption_tot += (self.finish_sby - self.start_sby) * self.pw_standby
                self.workstation.consumption_standby += (self.finish_sby - self.start_sby) * self.pw_standby
                self.workstation.consumption_tot += (self.finish_sby - self.start_sby) * self.pw_standby
                self.prod_system.consumption_standby += (self.finish_sby - self.start_sby) * self.pw_standby
                self.prod_system.consumption_tot += (self.finish_sby - self.start_sby) * self.pw_standby
                self.prod_system.record_consumption_tot()
                self.switched_on_times += 1
                self.state = 2  # Now its in startup
                yield self.env.timeout(startup)
                self.consumption_startup += startup * self.pw_startup
                self.consumption_tot += startup * self.pw_startup
                self.workstation.consumption_startup += startup * self.pw_startup
                self.workstation.consumption_tot += startup * self.pw_startup
                self.prod_system.consumption_startup += startup * self.pw_startup
                self.prod_system.consumption_tot += startup * self.pw_startup
                self.prod_system.record_consumption_tot()
                self.state = 3  # Now its idle
                self.start_idle = self.env.now

                if self.to_switch_off is True:

                    self.process = self.switch_off()
                    self.to_switch_off = False
                    break

                else:

                    continue

            if self.state == 2:
                break

            if self.state == 4:
                break

            if self.workstation.buffer < 1:  # why? the EEC is a single upstream buffer problem -> nothing to process
                break

            if self.state == 3:  # If machine is idle

                processing_time = random.expovariate(1 / self.pt)  # We produce a part
                while processing_time > 0:

                    if self.remaining <= 0:  # If the time is left until next failure is over, we have a failure
                        self.failures += 1
                        self.state = 5
                        self.finish_idle = self.env.now
                        self.consumption_idle += (self.finish_idle - self.start_idle) * self.pw_idle
                        self.consumption_tot += (self.finish_idle - self.start_idle) * self.pw_idle
                        self.workstation.consumption_idle += (self.finish_idle - self.start_idle) * self.pw_idle
                        self.workstation.consumption_tot += (self.finish_idle - self.start_idle) * self.pw_idle
                        self.prod_system.consumption_idle += (self.finish_idle - self.start_idle) * self.pw_idle
                        self.prod_system.consumption_tot += (self.finish_idle - self.start_idle) * self.pw_idle
                        self.prod_system.record_consumption_tot()
                        yield self.env.timeout(self.mttr)
                        self.repaired += 1
                        self.remaining = random.expovariate(1 / self.mttf)  # We understand how much time is left until next failure
                        self.state = 4  # We can produce a part finally
                        self.workstation.buffer -= 1

                        # originally there was a redundant decision step here!

                        yield self.env.timeout(processing_time)
                        self.consumption_busy += processing_time * self.pw_busy
                        self.consumption_tot += processing_time * self.pw_busy
                        self.workstation.consumption_busy += processing_time * self.pw_busy
                        self.workstation.consumption_tot += processing_time * self.pw_busy
                        self.prod_system.consumption_busy += processing_time * self.pw_busy
                        self.prod_system.consumption_tot += processing_time * self.pw_busy
                        self.prod_system.record_consumption_tot()
                        self.state = 3
                        self.start_idle = self.env.now
                        self.parts_made_partial += 1
                        self.workstation.parts_made += 1
                        self.prod_system.parts_made += 1
                        self.prod_system.record_parts_made()

                        if self.to_switch_off is True:

                            self.process = self.switch_off()
                            self.to_switch_off = False
                            self.workstation.departure_last()
                            processing_time = 0  # To go out of the while cycle

                        else:

                            self.workstation.departure_last()
                            processing_time = 0  # To go out of the while cycle

                    else:  # If the time is left until next failure is not over, we can produce

                        self.state = 4
                        self.workstation.buffer -= 1
                        self.finish_idle = self.env.now
                        self.consumption_idle += (self.finish_idle - self.start_idle) * self.pw_idle
                        self.consumption_tot += (self.finish_idle - self.start_idle) * self.pw_idle
                        self.workstation.consumption_idle += (self.finish_idle - self.start_idle) * self.pw_idle
                        self.workstation.consumption_tot += (self.finish_idle - self.start_idle) * self.pw_idle
                        self.prod_system.consumption_idle += (self.finish_idle - self.start_idle) * self.pw_idle
                        self.prod_system.consumption_tot += (self.finish_idle - self.start_idle) * self.pw_idle
                        self.prod_system.record_consumption_tot()
                        yield self.env.timeout(processing_time)
                        self.remaining = self.remaining - processing_time  # Time left until next failure decreases (Machine fails only while producing)
                        self.state = 3
                        self.start_idle = self.env.now
                        self.consumption_busy += processing_time * self.pw_busy
                        self.consumption_tot += processing_time * self.pw_busy
                        self.workstation.consumption_busy += processing_time * self.pw_busy
                        self.workstation.consumption_tot += processing_time * self.pw_busy
                        self.prod_system.consumption_busy += processing_time * self.pw_busy
                        self.prod_system.consumption_tot += processing_time * self.pw_busy
                        self.prod_system.record_consumption_tot()
                        self.parts_made_partial += 1
                        self.workstation.parts_made += 1
                        self.prod_system.parts_made += 1
                        self.prod_system.record_parts_made()

                        if self.to_switch_off is True:

                            self.process = self.switch_off()
                            self.to_switch_off = False
                            self.workstation.departure_last()
                            processing_time = 0  # To go out of the while cycle

                        else:
                            self.workstation.departure_last()
                            processing_time = 0  # To go out of the while cycle

            else:
                break
