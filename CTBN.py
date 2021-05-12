import scipy.stats as st
import pandas as pd
from Graph import *


class CTBN:
    def __init__(self, parent_list, n_states, cim_list):
        """

        :param parent_list:
        :param n_states:
        :param cim_list:
        """
        self.graph = Graph(n_states, parent_list)
        self.n_states = n_states
        self.cim_list = cim_list
        self.N = len(self.cim_list)

    def get_CIM(self, node, node_values):
        """
        Get the conditioned intensity matrix for node 'node'.

        :param node:
        :param node_values:
        :return: 2-D array.
        """
        cim_code = self.graph.encode_parent_value(node, node_values)
        return self.cim_list[node][cim_code]

    def get_CIM_q(self, node, node_values):
        cur_value = node_values[node]
        return np.abs(self.get_CIM(node, node_values)[cur_value][cur_value])

    def get_CIMs(self, node_values):
        return np.array([self.get_CIM(node, node_values) for node in range(self.N)])

    def get_CIMs_q(self, node_values):
        """
        Get the absolute transition intensity q for each node.

        :param node_values:
        :return:
        """
        return np.array([self.get_CIM_q(node, node_values) for node in range(self.N)])

    def sample(self, tau, init_value=None, verbose=False):
        """
        Sample the trajectories for CTBN nodes in time [0, tau].

        :param tau: double
            End time.
        :param init_value: 1-D array
            Init values of the nodes.
        :param verbose: bool
        :return:
        """
        # init_value not given, default zero
        if init_value is None:
            init_value = [0] * self.N
        # cur_value: 1-D array, current value of nodes
        cur_value = np.array(init_value)
        # traj: trajectory data being sampled.
        traj = None
        time = 0  # initial time
        intensities = self.get_CIMs_q(cur_value)
        while time < tau:
            # For each sampling, first get the transition intensities of each node,
            # add them for a sum intensity, and sample the transition time with exponential distribution.
            # intensities = self.get_CIMs_q(cur_value)
            sum_intensity = sum(intensities)
            trans_time = st.expon.rvs(size=1, scale=1/sum_intensity)[0]
            # Then, choose from a multinomial distribution which node transitions.
            trans_node = np.random.choice(self.N, 1, p=intensities/sum_intensity)[0]
            # Finally choose from another multinomial distribution the state which the node transitions to.
            cur_cim = self.get_CIM(trans_node, cur_value)
            cur_value_node = cur_value[trans_node]
            cur_trans_intensities = np.array(cur_cim[cur_value_node], dtype=float)
            cur_trans_intensities /= (-cur_trans_intensities[cur_value_node])
            cur_trans_intensities[cur_value_node] = 0
            next_value_node = np.random.choice(self.n_states, 1, p=cur_trans_intensities)[0]
            if verbose:
                print("intensities:", intensities)
                print("sum_intensity:", sum_intensity)
                print("trans_time:", trans_time)
                print("trans_node:%d, state:%d->%d" % (trans_node, cur_value_node, next_value_node))
            # Update time
            time += trans_time
            # Update cur_value
            pre_value = cur_value.copy()
            cur_value[trans_node] = next_value_node
            # Update intensities
            intensities[trans_node] = self.get_CIM_q(trans_node, cur_value)
            # Add into data
            cur_data = np.hstack((pre_value, np.array([trans_time, trans_node, cur_value_node, next_value_node])))
            if traj is None:
                traj = cur_data
            else:
                traj = np.vstack((traj, cur_data))
        result = pd.DataFrame(traj)
        return result
