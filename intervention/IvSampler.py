import pandas as pd
from IvSampleContainer import *


class IvSampler:
    def __init__(self, n_states, N):
        self.N = N
        self.n_states = n_states

    def sample(self, tau, mu):
        """
        Sample the interventions.
        :param tau: float, Time limit
        :param mu: float, Expected intervention time interval
        :return: pd.DataFrame, each row: [intervened time, intervened node, next state]
        """
        time = 0.0
        sample = IvSampleContainer(3)
        while time < tau:
            trans_time = np.random.rand() * 2 * mu
            if time + trans_time >= tau:
                break
            trans_node = np.random.randint(self.N)
            next_state = np.random.randint(self.n_states)
            time += trans_time
            trans_data = np.array([time, trans_node, next_state])
            sample.append(trans_data)
        result = pd.DataFrame(sample.get_values(), columns=["Tp", "node", "next_state"])
        return result
