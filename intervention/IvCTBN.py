from CTBN import *
from IvSampler import *
from IvSampleContainer import *


class IvCTBN(CTBN):
    def __init__(self, graph, cim_list, mu):
        super(IvCTBN, self).__init__(graph.parent_list, graph.n_states, cim_list)
        self.mu = mu
        self.iv_sample = None

    def sample(self, tau, init_value=None, verbose=False, cnt_line=True):
        """
        Sample with intervention.

        :param tau: float, Time limit.
        :param init_value: initial system value.
        :param verbose:
        :param cnt_line:
        :return: pd.DataFrame. columns: 0~N-1: system value;
            last 5 columns: intervention flag, stay time, transition node, begin state, next state
        """
        # initialize intervention sample
        iv_sampler = IvSampler(self.n_states, self.N)
        iv_sample = iv_sampler.sample(tau, self.mu)
        self.iv_sample = iv_sample
        iv_row_n = 0
        iv_len = len(iv_sample)
        # do traditional CTBN sampling
        # init_value not given, default zero
        if init_value is None:
            init_value = [0] * self.N
        # cur_value: 1-D array, current value of nodes
        cur_value = np.array(init_value)
        # traj: trajectory data being sampled.
        traj = IvSampleContainer(self.N + 5)
        time = 0  # initial time
        intensities = self.get_CIMs_q(cur_value)
        # output count
        line_cnt = 0
        while time < tau:
            # For each sampling, first get the transition intensities of each node,
            # add them for a sum intensity, and sample the transition time with exponential distribution.
            # intensities = self.get_CIMs_q(cur_value)
            sum_intensity = sum(intensities)
            trans_time = st.expon.rvs(size=1, scale=1 / sum_intensity)[0]
            # Then, choose from a multinomial distribution which node transitions.
            trans_node = np.random.choice(self.N, 1, p=intensities / sum_intensity)[0]
            # Finally choose from another multinomial distribution the state which the node transitions to.
            cur_cim = self.get_CIM(trans_node, cur_value)
            cur_value_node = cur_value[trans_node]
            cur_trans_intensities = np.array(cur_cim[cur_value_node], dtype=float)
            cur_trans_intensities /= (-cur_trans_intensities[cur_value_node])
            cur_trans_intensities[cur_value_node] = 0
            next_value_node = np.random.choice(self.n_states, 1, p=cur_trans_intensities)[0]
            intervention = False
            # Also note that if iv_row_n == iv_len, all intervention have been done.
            if iv_row_n < iv_len:
                iv_row = iv_sample.loc[iv_row_n]
                iv_time = iv_row["Tp"]
                # next proposed transition time after intervention time, ignore next transition
                while time + trans_time >= iv_time:
                    # enter an intervention, change iv_row_n next loop
                    iv_row_n += 1
                    # Note that the intervention must change the state of the system, or we'll ignore it.
                    if cur_value[int(iv_row["node"])] == iv_row["next_state"]:
                        if verbose:
                            print("Intervention ignored.")
                        if iv_row_n == iv_len:
                            break  # no more intervention
                        iv_row = iv_sample.loc[iv_row_n]
                        iv_time = iv_row["Tp"]
                    # activate the intervention
                    else:
                        # intervention time is the time transition happens
                        trans_time = iv_time - time
                        trans_node = int(iv_row["node"])
                        cur_value_node = cur_value[trans_node]
                        next_value_node = iv_row["next_state"]
                        intervention = True
                        if verbose:
                            print("Intervention activated.")
                        break  # found proper intervention
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
            cur_data = np.hstack((pre_value,
                                  np.array([intervention, trans_time, trans_node, cur_value_node, next_value_node])))
            traj.append(cur_data)
            if cnt_line:
                line_cnt += 1
                if line_cnt % 10000 == 0:
                    print("Generating data: " + str(line_cnt))
        result = pd.DataFrame(traj.get_values())
        return result
