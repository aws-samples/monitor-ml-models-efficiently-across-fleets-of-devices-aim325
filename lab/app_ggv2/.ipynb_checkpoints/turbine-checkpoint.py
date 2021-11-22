# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
import random
import time

class WindTurbine(object):
    """ Represents virtually and graphically a wind turbine
        It uses the raw data collected from a Wind Turbine in a circular buffer
        to simulate the real turbine sensors.
    """
    def __init__(self, turbine_id=0, raw_data=None):
        if raw_data is None or len(raw_data) == 0:
            raise Exception("You need to pass an array with at least one row for raw data")

        self.turbine_id = turbine_id # id of the turbine
        self.raw_data = raw_data # buffer with the raw sensors data
        self.raw_data_idx = random.randint(0, len(raw_data)-1)
        
        self.running = False # running status
        self.halted = False # if True you can't use this turbine anymore. create a new one.

    def is_running(self):
        return self.running
    
    def detected_anomalies(self, values, anomalies):
        """ Updates the status of the 'inject noise' buttons (pressed or not)"""
        self.vibration_status.value = not anomalies[0:3].any()
        self.voltage_status.value = not anomalies[3:5].any()
        self.rotation_status.value = not anomalies[5]

    def is_noise_enabled(self, typ):
        """ Returns the status of the 'inject noise' buttons (pressed or not)"""
        assert(typ == 'Vol' or typ == 'Rot' or typ == 'Vib')
        idx = 0
        if typ == 'Vol': idx = 0
        elif typ == 'Rot': idx = 1
        elif typ == 'Vib': idx = 2
        return False
            
    def halt(self):
        """ Halts the turnine and disable it. After calling this method you can't use it anymore."""
        self.running = False
        self.button.description = 'Halted'
        self.img.value = self.stopped_img
        self.anomaly_status.layout.visibility='hidden'
        self.halted = True                    

    def read_next_sample(self):        
        """ next step in this simulation """
        if self.raw_data_idx >= len(self.raw_data): self.raw_data_idx = 0
        sample = self.raw_data[self.raw_data_idx]
        self.raw_data_idx += 1
        return sample
