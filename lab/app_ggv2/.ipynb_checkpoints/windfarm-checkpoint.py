# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
import threading
import math
import random
import pywt
import numpy as np
import pandas as pd
import logging
import time
import os
from turbine import WindTurbine
from edgeagentclient import EdgeAgentClient

class WindTurbineFarm(object):
    """ 
    This is the application class. It is respoisible for:
        - Creating virtual edge devices (as threads)
        - Launch one Edge Agent in each virtual device
        - Load the Anomaly detection for the Wind Turbine in the Edge Agent
        - Launch the Virtual Wind Turbines
        - Launch a Edge Agent Client that integrates the Wind Turbine with the Edge Device
        - Display the UI
    """
    def __init__(self, n_turbines):
        self.artifacts_path = os.environ.get("ARTIFACTS_PATH")
        self.raw_data = pd.read_csv(f'{self.artifacts_path}/dataset_wind.csv.gz', compression="gzip", sep=',', low_memory=False).values 
        self.n_turbines = n_turbines
        self.turbines = [WindTurbine(i, self.raw_data) for i in range(self.n_turbines)]
        
        self.data_buffer = [[] for i in range(self.n_turbines)]
        
        ## launch edge agent clients
        self.edge_agent = EdgeAgentClient('/tmp/aws.greengrass.SageMakerEdgeManager.sock')
        self.model_meta = [{'model_name':None} for i in range(self.n_turbines)]

        # we need to load the statistics computed in the data prep notebook
        # these statistics will be used to compute normalize the input
        self.raw_std = np.load(f'{self.artifacts_path}/raw_std.npy')
        self.mean = np.load(f'{self.artifacts_path}/mean.npy')
        self.std = np.load(f'{self.artifacts_path}/std.npy')
        # then we load the thresholds computed in the training notebook
        # for more info, take a look on the Notebook #2
        self.thresholds = np.load(f'{self.artifacts_path}/thresholds.npy')
        
        # configurations to format the time based data for the anomaly detection model
        # If you change these parameters you need to retrain your model with the new parameters        
        self.INTERVAL = 5 # seconds
        self.TIME_STEPS = 20 * self.INTERVAL # 50ms -> seg: 50ms * 20
        self.STEP = 10
        
        # these are the features used in this application
        self.feature_ids = [8, 9, 10, 7, 22, 5, 6] # qX,qy,qz,qw  ,wind_seed_rps, rps, voltage        
        self.n_features = 6 # roll, pitch, yaw, wind_speed, rotor_speed, voltage
        self.running = False # running status
        
        # minimal buffer length for denoising. We need to accumulate some sample before denoising
        self.min_num_samples = 500
        
        self.max_buffer_size = 500
        for idx in range(n_turbines):
            for j in range(self.max_buffer_size):
                self.__read_next_turbine_sample__(idx)

    def __create_dataset__(self, X, time_steps=1, step=1):
        """
        This encodes a list of readings into the correct shape
        expected by the model. It uses the concept of a sliding window
        """
        Xs = []
        for i in range(0, len(X) - time_steps, step):
            v = X[i:(i + time_steps)]
            Xs.append(v)
        return np.array(Xs)

    def __euler_from_quaternion__(self, x, y, z, w):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)

        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)

        return roll_x, pitch_y, yaw_z # in radians
        
    def __wavelet_denoise__(self, data, wavelet, noise_sigma):
        '''
        Filter accelerometer data using wavelet denoising
        Modification of F. Blanco-Silva's code at: https://goo.gl/gOQwy5
        '''

        wavelet = pywt.Wavelet(wavelet)
        levels  = min(5, (np.floor(np.log2(data.shape[0]))).astype(int))

        # Francisco's code used wavedec2 for image data
        wavelet_coeffs = pywt.wavedec(data, wavelet, level=levels)
        threshold = noise_sigma*np.sqrt(2*np.log2(data.size))

        new_wavelet_coeffs = map(lambda x: pywt.threshold(x, threshold, mode='soft'), wavelet_coeffs)

        return pywt.waverec(list(new_wavelet_coeffs), wavelet)

    def __del__(self):
        """Destructor"""
        self.halt()

    def is_noise_enabled(self, turbine_id):
        return [self.turbines[turbine_id].is_noise_enabled('Vib'),
                self.turbines[turbine_id].is_noise_enabled('Rot'),
                self.turbines[turbine_id].is_noise_enabled('Vol')]
    
    def __data_prep__(self, turbine_id, buffer):
        """
        This method is called for each reading.
        Here we do some data prep and accumulate the data in the buffer
        for denoising
        """
        new_buffer = []
        for data in buffer:
            roll,pitch,yaw = self.__euler_from_quaternion__(
                data[self.feature_ids[0]],data[self.feature_ids[1]],
                data[self.feature_ids[2]],data[self.feature_ids[3]]
            )
            row = [roll,pitch,yaw, data[self.feature_ids[4]],data[self.feature_ids[5]], data[self.feature_ids[6]]]
            new_buffer.append(row)
        return np.array(new_buffer)
    
    def __prep_turbine_sample__(self, turbine_id, data):
        vib_noise,rot_noise,vol_noise = self.is_noise_enabled(turbine_id)
        #np.array([8,9,10,7,  22, 5, 6]) # qX,qy,qz,qw  ,wind_seed_rps, rps, voltage        
        if vib_noise: data[self.feature_ids[0:4]] = np.random.rand(4) * 100 # out of the radians range
        if rot_noise: data[self.feature_ids[5]] = np.random.rand(1) * 100 # out of the normalized wind range
        if vol_noise: data[self.feature_ids[6]] = int(np.random.rand(1)[0] * 10000) # out of the normalized voltage range

        self.data_buffer[turbine_id].append(data)
        if len(self.data_buffer[turbine_id]) > self.max_buffer_size:
            del self.data_buffer[turbine_id][0]
            
    def get_raw_data(self, turbine_id):        
        assert(turbine_id >= 0 and turbine_id < len(self.data_buffer))
        self.__read_next_turbine_sample__(turbine_id)
        return self.data_buffer[turbine_id]
    
    def __read_next_turbine_sample__(self, turbine_id):
        self.__prep_turbine_sample__(turbine_id, self.turbines[turbine_id].read_next_sample() )
            
    def __detect_anomalies__(self):     
        """
        Keeps processing the data collected from the turbines
        and do anomaly detection. It reports to each turbine the 
        anomalies detected (through a callback)
        """
        while self.running:
            # for each turbine, check the buffer
            start_time = time.time()
            for idx in range(self.n_turbines): 
                print(idx)
                buffer = self.get_raw_data(idx)
                if len(buffer) >= self.min_num_samples:
                    # create a copy & prep the data
                    data = self.__data_prep__(idx, np.array(buffer) )

                    if not self.edge_agent.is_model_loaded(self.model_meta[idx]['model_name']):
                        print('model is not loaded')
                        continue

                    # denoise
                    data = np.array([self.__wavelet_denoise__(data[:,i], 'db6', self.raw_std[i]) for i in range(self.n_features)])
                    data = data.transpose((1,0))

                    # normalize                     
                    data -= self.mean
                    data /= self.std
                    data = data[-(self.TIME_STEPS+self.STEP):]

                    # create the dataset and reshape it
                    x = self.__create_dataset__(data, self.TIME_STEPS, self.STEP)                    
                    x = np.transpose(x, (0, 2, 1)).reshape(x.shape[0], self.n_features, 10, 10)

                    # run the model                    
                    p = self.edge_agent.predict(self.model_meta[idx]['model_name'], x)
                    if p is not None:
                        a = x.reshape(x.shape[0], self.n_features, 100).transpose((0,2,1))
                        b = p.reshape(p.shape[0], self.n_features, 100).transpose((0,2,1))
                        # check the anomalies
                        pred_mae_loss = np.mean(np.abs(b - a), axis=1).transpose((1,0))

                        values = np.mean(pred_mae_loss, axis=1)
                        anomalies = (values > self.thresholds)
                        
            elapsed_time = time.time() - start_time
            time.sleep(0.5-elapsed_time)

    def load_model(self, model_name, model_version):
        logging.info("Loading model %s version %s" % ( model_name, model_version))
        model_path = os.environ.get("MODEL_PATH")
        
        ret = self.edge_agent.load_model(model_name, model_path)
        
        if ret is not None:
            for device_id in range(self.n_turbines):
                self.model_meta[device_id]['model_name'] = model_name
                self.model_meta[device_id]['model_path'] = model_path
                self.model_meta[device_id]['model_version'] = model_version
            

    def start(self):
        """
        Run the main application by creating the Edge Agents, loading the model and
        kicking-off the anomaly detector program
        """
        self.load_model("WindTurbineAnomalyDetection", "1.0")
        if not self.running:
            self.running = True
            
            logging.info("Starting the anomaly detector loop...")
            # finally start the anomaly detection loop
            self.processing = threading.Thread(target=self.__detect_anomalies__)
            self.processing.start()
            
    
    def halt(self):
        """
        Destroys the application and halts the agents & turbines
        """
        if self.running:
            self.running = False
            self.processing.join()
            