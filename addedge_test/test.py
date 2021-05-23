import pandas as pd
import numpy as np
import scipy.stats as st
np.random.seed(1000)

print(st.binom.rvs(1, size=100, p=0.9))
print(st.norm.rvs(size=100, loc=5, scale=1))
