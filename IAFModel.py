import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CONNECTIONS = {0: {1: 1},
               1: {0: 1}}
MODEL_PARAMS = {}

#TODO: poissonic stimulation

class IAFSimulation(object):

    def __init__(self, number_of_neurons, connections, electrodes,
                 V_rest=-70, V_reset=-70, V_th=-54, V_is=-80, V_es=0, tm=20, P_max=1, rm_gs=0.05, Rm=10, ts=10):
        self.V_es = V_es
        self.V_is = V_is
        self.V_th = V_th
        self.V_rest = V_rest
        self.V_reset = -70
        self.Rm = Rm
        self.P_max = P_max
        self.rm_gs = rm_gs
        self.ts = ts
        self.tm = tm
        self.number_of_neurons = len(connections)
        self.connectivity_matrix = pd.DataFrame.from_dict(CONNECTIONS).replace(np.NaN, 0).values
        self.neurons = None
        self.P_synapses = None
        self.dt = None

    def

    def simultate(self, dt = 1, total_time = 1000 * 20):
        self.neurons = np.number_of_neurons((total_time, self.number_of_neurons))
        self.neurons[0, :] = self.V_rest
        self.P_synapses = np.zeroes((self.number_of_neurons, self.number_of_neurons))
        self.dt = dt
        self.last_AP = np.zeros(self.number_of_neurons)
        for t in range(total_time - 1):
            self.propagate(t)

    def get_tau_v(self, t):
        return self.tm / (1 + self.rm_gs * np.matmul(np.self.P_synapses[t:, :], self.connectivity_matrix))

    def get_V_inf(self, t):
        tot_syn = np.matmul(np.self.P_synapses[t:, :], self.connectivity_matrix)
        return (self.V_rest + self.rm_gs * tot_syn * self.V_es + self.Rm * self.I[t, :]) / (1 + self.rm_gs * tot_syn)

    def propagate(self, t):
        """
        Solve the equation: eff_tau_v* dV/dt = eff_V_inf - V
        """
        # propagate voltage
        v_inf = self.get_V_inf(t)
        self.neurons[t + 1, :] = np.where(self.neurons[t, :] > self.V_th,
                                          self.V_reset,
                                          (self.neurons[t, :] - v_inf) * np.exp(-self.dt / self.get_tau_v(t)))
        # check for action potential
        self.last_AP = np.where(self.neurons[t, :] > self.V_th,
                                t*self.dt,
                                self.last_AP)
        # propagate Ps
        self.P_synapses[t + 1, :, :] = np.where(self.last_AP > 0,
                                                self.P_synapses[t, :, :] * ((t*self.dt-self.last_AP)/self.ts) * self.P_max,
                                                0)

    def plot_neurons(self):
        pass