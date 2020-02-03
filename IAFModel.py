import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
import networkx as nx
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


AP_VOLTAGE = 0
# From Neuron->To Neuron
CONNECTIONS = {0: {0:0, 1:0, 3:1},
               1: {0:1, 1:0, 4:1},
               2: {0:1, 1:1},
               3: {1:1, 2:1},
               4: {3:1, 2:1}}
n=5
CONNECTIONS_FULLY = dict([(j, dict([(i,1) for i in set(range(n)).difference([j])])) for j in range(n)])
#TODO: poissonic stimulation

class IAFSimulation(object):

    def __init__(self, connections, input_current=2,
                 V_rest=-0.07, V_reset=-0.07, V_th=-0.054, V_is=-0.08, V_es=0, tm=0.02, ts=0.01, P_max=1, rm_gs=0.5, Rm=0.010):
        self.V_es = V_es
        self.V_is = V_is
        self.V_th = V_th
        self.V_rest = V_rest
        self.V_reset = V_reset
        self.Rm = Rm
        self.input_current=input_current
        self.P_max = P_max
        self.rm_gs = rm_gs
        self.ts = ts
        self.tm = tm
        self.number_of_neurons = len(connections)
        self.connectivity_matrix = np.fromfunction(np.vectorize(lambda i, j:connections.get(i,{}).get(j,0)), (self.number_of_neurons,self.number_of_neurons)).T
        self.neurons = None
        self.P_synapses = None
        self.dt = None
        self.Ie = None

    def simultate(self, dt = 0.0001, total_time = 1000 * 20):
        self.neurons = np.zeros((total_time, self.number_of_neurons))
        self.Ie = np.random.normal(self.input_current, self.input_current /3, (total_time, self.number_of_neurons))
        self.neurons[0, :] = self.V_rest
        self.P_synapses = np.zeros((total_time, self.number_of_neurons, self.number_of_neurons))
        self.dt = dt
        self.last_AP = np.zeros(self.number_of_neurons)
        for t in range(total_time - 1):
            self.propagate(t)
        self.neurons = np.where(self.neurons > self.V_th,
                                AP_VOLTAGE,
                                self.neurons)

    def get_tau_v(self, t):
        return self.tm / (1 + self.rm_gs * np.sum(self.P_synapses[t, :, :] * self.connectivity_matrix, axis=1))

    def get_V_inf(self, t):
        tot_syn = np.sum(self.P_synapses[t, :, :] * self.connectivity_matrix, axis=1)
        return (self.V_rest + self.rm_gs * tot_syn * self.V_es + self.Rm * self.Ie[t, :]) / (1 + self.rm_gs * tot_syn)

    def propagate(self, t):
        """
        Solve the equation: eff_tau_v* dV/dt = eff_V_inf - V
        """
        # propagate voltage
        v_inf = self.get_V_inf(t)
        self.neurons[t + 1, :] = np.where(self.neurons[t, :] > self.V_th,
                                          self.V_reset,
                                          v_inf + (self.neurons[t, :] - v_inf) * np.exp(-self.dt / self.get_tau_v(t)))
        # check for action potential
        self.last_AP = np.where(self.neurons[t, :] > self.V_th,
                                t,
                                self.last_AP)
        # propagate Ps
        self.P_synapses[t + 1, :, :] = np.where(self.last_AP > 0,
                                                (((t*self.dt - self.last_AP*self.dt) / self.ts) * np.exp(1 - (t*self.dt-self.last_AP*self.dt)/self.ts)) * self.P_max,
                                                0)

    def plot_results(self, N=100):
        gs = gridspec.GridSpec(self.number_of_neurons, 4)
        f = plt.figure(figsize=(20,20))

        for i in range(self.number_of_neurons):
            ax0 = f.add_subplot(gs[i,0])
            ax0.plot(range(self.P_synapses[:N].shape[0]), self.P_synapses[:N, i, i])
            ax0.set_title("synapse {}".format(i))
            ax1 = f.add_subplot(gs[i, 1])
            ax1.plot(range(self.neurons[:N, i].shape[0]), self.neurons[:N, i])
            ax1.set_title("neuron {}".format(i))
            ax2 = f.add_subplot(gs[i, 2])
            ax2.plot(range(self.Ie[:N, i].shape[0]), self.Ie[:N, i])
            ax2.set_title("Ie {}".format(i))
        ax3 = f.add_subplot(gs[:, 3])
        g = nx.from_numpy_matrix(self.connectivity_matrix.T, create_using=nx.DiGraph)
        nx.draw_networkx(g, with_labels=True, ax=ax3)
        plt.tight_layout()


def analyze_APs(neurons, dt=0.001, interval=0.02):
    df = pd.DataFrame(data=neurons, columns=map("neuron_{}".format, range(neurons.shape[1])))
    df = df.groupby(pd.cut(df.index, np.arange(0, df.index.shape[0], int(interval/dt)))).max().reset_index(drop=True)
    return (df >= AP_VOLTAGE).astype(int)

#TODO: graph of the Average Neuron Input Strength



def run_single():
    iaf = IAFSimulation(CONNECTIONS)
    dt=0.001
    iaf.simultate(dt=dt, total_time=int(1/dt * 20))
    iaf.plot_results(N=int(1/(dt*10)))
    df = analyze_APs(iaf.neurons, dt, 0.02)
    plt.pause(0.001)

def run_simulation(current, synapse_weight):
    dt = 0.001
    # run simulation with the given params
    iaf = IAFSimulation(CONNECTIONS_FULLY)
    iaf.simultate(dt=dt, total_time=int(1 / dt * 20))
    df = analyze_APs(iaf.neurons, dt, dt)
    # calculate the firing rate
    return df.mean().mean() * (1/dt)


def compute_grid():
    x = np.linspace(0.5, 4, 12)
    y = np.linspace(0.05, 2, 12)
    xx, yy = np.meshgrid(x, y)
    vect_sim = np.vectorize(run_simulation)
    z = vect_sim(xx, yy)
    fig = plt.figure(figsize = [15,10])
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(xx, yy, z, cmap=cm.coolwarm,linewidth=0)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.set_xlabel(r'$|I_{e}|$')
    ax.set_ylabel(r'$r_{m}g_{s}$')
    ax.set_zlabel(r'$FireRate(Hz)$')
    plt.pause(0.001)

compute_grid()
input("")

