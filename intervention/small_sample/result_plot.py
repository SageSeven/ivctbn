import pandas as pd
import matplotlib.pyplot as plt

iv_rates = [0.1, 0.2, 0.3, 0.4, 0.5]
data_lens = [1000, 2000, 3000, 4000, 5000, 10000]


def get_file_name(iv_rate_, data_len_):
    return "result_"+str(data_len_)+"_"+str(iv_rate_)+".csv"


plt.figure(figsize=[20, 16])
plt.subplots_adjust(hspace=0.3)

for i, iv_rate in enumerate(iv_rates):
    plt.subplot(2, 3, i+1)
    x = data_lens
    y0 = x.copy()
    y1 = x.copy()
    for j, data_len in enumerate(data_lens):
        data = pd.read_csv(get_file_name(iv_rate, data_len), index_col=0)
        y0[j] = data["iv"].mean()
        y1[j] = data["trad"].mean()
    plt.plot(x, y0, "o-", label="IvCTBN")
    plt.plot(x, y1, "o--", label="CTBN")
    plt.legend()
    plt.ylim([0, 1.03])
    #if i >= 3:
     #   plt.xlabel("data length")
    if i % 3 == 0:
        plt.ylabel("F-score")
    plt.title("("+list("abcde")[i]+")", y=-0.3)

plt.show()
