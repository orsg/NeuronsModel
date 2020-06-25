import matplotlib.pyplot as plt

from __future__ import print_function, division
import jax.numpy as np
from jax import grad, jit, vmap
from jax import random
import numpy as onp

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
import networkx as nx
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

import MaximumEntropy

AP_VOLTAGE = 0
# From Neuron->To Neuron
CONNECTIONS = {0: {0: 0, 1: 0, 3: 1},
               1: {0: 1, 1: 0, 4: 1},
               2: {0: 1, 1: 10},
               3: {1: 1, 2: 1},
               4: {3: 1, 2: 1}}


def CONNECTIONS_FULLY(n):
    return dict([(j, dict([(i, 1) for i in set(range(n)).difference([j])])) for j in range(n)])


# TODO: poissonic stimulation

class IAFSimulation(object):

    def __init__(self, connections, input_current=2,
                 V_rest=-0.07, V_reset=-0.07, V_th=-0.054, V_is=-0.08, V_es=0, tm=0.02, ts=0.01, P_max=1, rm_gs=0.5,
                 Rm=0.010):
        self.V_es = V_es
        self.V_is = V_is
        self.V_th = V_th
        self.V_rest = V_rest
        self.V_reset = V_reset
        self.Rm = Rm
        self.input_current = input_current
        self.P_max = P_max
        self.rm_gs = rm_gs
        self.ts = ts
        self.tm = tm
        self.number_of_neurons = len(connections)
        self.connectivity_matrix = np.fromfunction(np.vectorize(lambda i, j: connections.get(i, {}).get(j, 0)),
                                                   (self.number_of_neurons, self.number_of_neurons)).T
        self.neurons = None
        self.P_synapses = None
        self.dt = None
        self.Ie = None

    def simultate(self, dt=0.0001, total_time=1000 * 20):
        self.neurons = np.zeros((total_time, self.number_of_neurons))
        self.Ie = np.random.normal(self.input_current, self.input_current / 3, (total_time, self.number_of_neurons))
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
        dadt = -a + np.dot(wI, x) + np.dot(wR, h) + np.dot(wF, z)
        a = a + dtdivtau * dadt
        h = np.tanh(a)
        z = np.dot(wO, h)
        return a, h, z

    def plot_results(self, N=100):
        gs = gridspec.GridSpec(self.number_of_neurons, 4)
        f = plt.figure(figsize=(20, 20))

        for i in range(self.number_of_neurons):
            ax0 = f.add_subplot(gs[i, 0])
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
    df = df.groupby(pd.cut(df.index, np.arange(0, df.index.shape[0], int(interval / dt)))).max().reset_index(drop=True)
    return (df >= AP_VOLTAGE).astype(int)


# TODO: graph of the Average Neuron Input Strength


def run_single(plot=True, connections=CONNECTIONS):
    iaf = IAFSimulation(connections, input_current=2)
    dt = 0.001
    iaf.simultate(dt=dt, total_time=int(1 / dt * 20))
    if plot:
        iaf.plot_results(N=int(1 / (dt * 10)))
    df = analyze_APs(iaf.neurons, dt, dt)
    plt.pause(0.001)
    return df

df = run_single(connections=CONNECTIONS_FULLY(4))
input("")



def esn(x, a, h, z, wI, wR, wF, wO, dtdivtau):
    """Run the continuous-time Fire Rate network one step.

      da/dt = -a + wI x + wR h + wF z

      Arguments:
        x: ndarray of input to ESN
        a: ndarray of activations (pre nonlinearity) from prev time step
        h: ndarray of hidden states from prev time step
        z: ndarray of output from prev time step
        wI: ndarray, input matrix, shape (n, u)
        wR: ndarray, recurrent matrix, shape (n, n)
        wF: ndarray, feedback matrix, shape (n, m)
        wO: ndarray, output matrix, shape (m, n)
        dtdivtau: dt / tau

      Returns:
        The update to the ESN at this time step.
    """
    dadt = -a + np.dot(wI, x) + np.dot(wR, h) + np.dot(wF, z)
    a = a + dtdivtau * dadt
    h = np.tanh(a)
    z = np.dot(wO, h)
    return a, h, z


def esn_run_and_train_jax(params, fparams, x_t, f_t=None, do_train=False):
    """Run the Echostate network forward a number of steps the length of x_t.

      This implementation uses JAX to build the outer time loop from basic
      Python for loop.

      Arguments:
        params: dict of ESN params
        fparams: dict of RLS params
        x_t: ndarray of input time series, shape (t, u)
        f_t: ndarray of target time series, shape (t, m)
        do_train: Should the network be trained on this run?

      Returns:
        4-tuple of params, fparams, h_t, z_t, after running ESN and potentially
          updating the readout vector.
    """
    # per-example predictions
    a = params['a0']
    h = np.tanh(a)
    wO = params['wO']
    z = np.dot(wO, h)
    if do_train:
        P = fparams['P']
    else:
        P = None
    h_t = []
    z_t = []

    wI = params['wI']
    wR = params['wR']
    wF = params['wF']
    dtdivtau = params['dt_over_tau']
    for tidx, x in enumerate(x_t):
        a, h, z = esn(x, a, h, z, wI, wR, wF, wO, dtdivtau)
        if do_train:
            wO, P = rls(h, z, f_t[tidx], wO, P)
        h_t.append(h)
        z_t.append(z)

    if do_train:
        fparams['P'] = P
    params['wO'] = wO
    h_t = np.array(h_t)
    z_t = np.array(z_t)
    return params, fparams, h_t, z_t