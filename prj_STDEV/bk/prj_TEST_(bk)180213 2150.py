import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import matplotlib.pyplot as plt

resp = range(200)
# resp = np.random.randint(0,100,200)

basis_stdev = np.std(resp[0:5])

# lst = []   # lst = list() 와 동일
colNames = ['number', 'ref_stdev', 'lower_stdev', 'upper_stdev', 'diff', "rms"]
df = pd.DataFrame(columns=colNames)

for i in resp:
    # print(i, end=" ")
    if (int(i) == 0) or (int(i+1) >= len(resp)):
        continue
    ref_stdev = np.std(resp[i:i+5])
    lower_stdev = np.std(resp[:i])
    upper_stdev = np.std(resp[i+1:])
    diff = lower_stdev - upper_stdev
    rms = ((lower_stdev - upper_stdev)**2)**0.5

    lst = [i, ref_stdev, lower_stdev, upper_stdev, diff, rms]
    series = pd.Series(lst, index=colNames)    # (잘됨)
    # print(series)
    df = df.append(series, ignore_index=True)

    # ref.update({i:ref_std})    # dictionary에 추가할 때

print(df.head(20))

# lst_ref = sorted(ref_std.items())     # 키를 기준으로 정렬
# x1,y1 = zip(*lists_ref)             # unpack a list of pairs into two tuples

plt.figure(num=1, figsize=(6,4))    # make the first figure
plt.subplot(2,1,1)                  # the first subplot in the first figure
plt.axhline(basis_stdev, linestyle="--", color="m", )
plt.plot(df["number"], df["ref_stdev"], "k.")
plt.plot(df["number"], df["lower_stdev"], "c.", alpha=0.4)
plt.plot(df["number"], df["upper_stdev"], "g.", alpha=0.4)
# plt.axis([0,100,-50,50])    # [Xmin, Xmax, Ymin, Ymax]

plt.subplot(2,1,2)                  # the second subplot in the first figure
plt.plot(df["number"], df["diff"], "r+", alpha=0.4)
plt.plot(df["number"], df["rms"], "b+", alpha=0.4)

# plt.figure(num=2, figsize=(6,4))        # make the second figure
# plt.subplot(1,1,1)                      # the first subplot in the second figure
# plt.plot(df["number"], df["rms"], "b+")
# plt.axis([0,100,-50,50])

# plt.figure(num=1)    # figure1 current; subplot(212) still current
# plt.subplot(211)     # make subplt(211) in figure1 current
plt.show()
