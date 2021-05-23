import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

iv_rates = [0.1, 0.2, 0.3, 0.4, 0.5]
data_lens = [1000, 2000, 3000, 4000, 5000, 10000, 15000, 20000]


def get_file_name(iv_rate_, data_len_):
    return "result_"+str(data_len_)+"_"+str(iv_rate_)+".csv"


plt.figure(figsize=[20, 16])
gs = gridspec.GridSpec(2, 6)
plt.subplots_adjust(hspace=0.3, wspace=0.5)

for i, iv_rate in enumerate(iv_rates):
    if i < 3:
        plt.subplot(gs[0, (2*i):(2*i+2)])
        data_n = 6
    else:
        j = i - 3
        plt.subplot(gs[1, (3*j):(3*j+3)])
        data_n = 7
    x = data_lens[:data_n]
    y0 = x.copy()
    y1 = x.copy()
    for j, data_len in enumerate(data_lens[:data_n]):
        data = pd.read_csv(get_file_name(iv_rate, data_len), index_col=0)
        y0[j] = data["iv"].mean()
        y1[j] = data["trad"].mean()
    plt.plot(x, y0, "o-", label="IvCTBN")
    plt.plot(x, y1, "o--", label="CTBN")
    plt.legend()
    plt.ylim([0, 1.03])
    # if i >= 3:
    #   plt.xlabel("data length")
    if i % 3 == 0:
        plt.ylabel("F-score")
    plt.title("("+list("abcde")[i]+")", y=-0.3)

plt.show()
