import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import matplotlib.pyplot as plt

resp = range(200)
# resp = np.random.randint(0,100,200)

basis_stdev = np.std(resp[0:5])

# lst = []   # lst = list() 와 동일
colNames = ['number', 'ref_stdev', 'lower_stdev', 'upper_stdev', 'diff']
df = pd.DataFrame(columns=colNames)

for i in resp:
    # print(i, end=" ")
    if (int(i) == 0) or (int(i+1) >= len(resp)):
        continue
    ref_stdev = np.std(resp[i:i+5])
    lower_stdev = np.std(resp[:i])
    upper_stdev = np.std(resp[i+1:])
    diff = lower_stdev - upper_stdev

    lst = [i, ref_stdev, lower_stdev, upper_stdev, diff]
    series = pd.Series(lst, index=colNames)    # (잘됨)
    # print(series)
    df = df.append(series, ignore_index=True)

    # ref.update({i:ref_std})    # dictionary에 추가할 때

print(df.head(20))

# df.min[axis=1"diff"]

# lst_ref = sorted(ref_std.items())     # 키를 기준으로 정렬
# x1,y1 = zip(*lists_ref)             # unpack a list of pairs into two tuples

plt.figure()
plt.axhline(basis_stdev, linestyle="--", color="m", )
plt.plot(df["number"], df["ref_stdev"], "b.")
plt.plot(df["number"], df["lower_stdev"], "c.")
plt.plot(df["number"], df["upper_stdev"], "g.")
plt.plot(df["number"], df["diff"], "r+")
plt.show()
