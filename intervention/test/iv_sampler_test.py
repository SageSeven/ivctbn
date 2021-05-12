from IvSampler import *

iv_sampler = IvSampler(2, 6)
sample = iv_sampler.sample(1000, 10)
print(sample)

for index, row in sample.iterrows():
    pass

print(sample.loc[0])
print(len(sample))
print(np.array(sample.loc[0]))
