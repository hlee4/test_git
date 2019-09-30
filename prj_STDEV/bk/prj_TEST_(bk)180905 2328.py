# -*- coding: utf-8 -*-
'''
    (설명부분)
'''
# (A) Import Modules
import collections, datetime, glob
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.style as style

# (Matplotlib Style Settings)
style.use('fivethirtyeight')

# (Matplotlib Font Settings)
font_location = 'c:/Windows/Fonts/malgun.ttf'
# font_location = '/usr/share/fonts/truetype/msfonts/malgun.ttf'
font_name = mpl.font_manager.FontProperties(fname=font_location).get_name()
mpl.rcParams['font.family'] = font_name
mpl.rcParams['font.size'] = 12

# (Pandas Display options)
pd.set_option('display.max_columns', 20)
# pd.set_option('max_colwidth', 18)
# pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.width', 100)
pd.set_option('display.unicode.east_asian_width', True)



# (B) Define Functions
# (01) FUNC :: main function
def main():
    # (1-1) Load Data from Files
    # files = glob.glob('./dn_files/**/*.csv', recursive=True)
    # file = files[-1]
    # print(file)

    # df = pd.read_csv(file, sep=',', header=None)
    # df = pd.read_excel(file, sheet_name=0, headers=True, skiprows=2, usecols=[1,2,3])
    # print(df.head(4))

    # (1-2) Data Munging :: ㅇㅇㅇ
    raw_data = {
        # 'date': pd.date_range(start='2018-01-01', periods=60),
        'val1': np.random.randint(low=0, high=100, size=60)
        # 'val2': range(60)
    }
    df = pd.DataFrame(data=raw_data)

    df['val3'] = df['val1'].cumsum(axis=0)
    df['val4'] = df['val3'].rolling(window=2).sum()
    df['val5'] = pd.Series([df['val1'][0:i].tolist() for i in range(1,df['val1'].size)])
    df['val6'] = [np.std(x) for x in df['val5']]
    df['val7'] = [np.mean(x) for x in df['val5']]
    print(df.head(5))
    # return()

    # for i in range(len(df['val1'])):
    #   df_tmp = df['val1'][i:]
    #   df['val7'] = pd.Series([df_tmp[0:j].tolist() for j in range(1,df_tmp.size)])
    #   df['val8'] = [np.std(x) for x in df['val7']]

    #   print(df['val7'].head(5))

    # return()

    base = np.std(df['val1'][0:3]) / np.mean(df['val1'][0:3])

    for i in range(1,df['val1'].size):
        for j in range(1,df['val1'].size):
            # print(df['val1'][i:i+j])
            cmp = np.std(df['val1'][i:i+j])/np.mean(df['val1'][i:i+j])
            if cmp <= base:
                print(cmp)
                # continue

    # stdev = np.std(resp[i:i+5])
    # lower_stdev = np.std(resp[:i])
    # upper_stdev = np.std(resp[i+1:])
    # diff = lower_stdev - upper_stdev
    # rms = ((lower_stdev - upper_stdev)**2)**0.5

    # lst_ref = sorted(ref_std.items())     # 키를 기준으로 정렬
    # x1,y1 = zip(*lists_ref)             # unpack a list of pairs into two tuples

    # df.to_csv('./df_neat.csv', encoding='cp949')

    # (1-3) Plot

    # (1-4) 참고자료
    # lst = [range(0,10)];        # print(lst)
    # lst = list(range(0,10));    # print(lst)
    # lst1 = [*range(0,11)];      # print(lst1)
    return()


# (11) FUNC :: Plot DateFrame
def plot_01_STDEV(df):
    # fig, (ax1) = plt.subplots(nrows=1, ncols=1, figsize=(8,6))
    # df['col1'].plot(x=df.index, y='col2', ax=ax1)
    # df.groupby(by='col1').plot(x='col2', y='col3', ax=ax1, legend=True)

    # ax1.set_xlim(['2018-01-01','2018-03-31'])

    # (x-axis ticker locator settings)
    # ax1.xaxis.set_major_locator(mticker.NullLocator())
    # ax1.xaxis.set_major_locator(mticker.AutoLocator())
    # ax1.xaxis.set_major_locator(mticker.MaxNLocator(4))
    # ax1.xaxis.set_major_locator(mdates.YearLocator())
    # ax1.xaxis.set_minor_locator(mdates.MonthLocator())
    # ax1.xaxis.set_minor_locator(mdates.DayLocator())
    # ax1.xaxis.set_minor_locator(mdates.HourLocator(interval=6))

    # (x-axis ticker formatter settings)
    # ax1.xaxis.set_major_formatter(mticker.NullFormatter())
    # ax1.xaxis.set_major_formatter(mticker.FixedFormatter('%y\n%m-%d'))
    # ax1.xaxis.set_major_formatter(mdates.AutoDateFormatter(mdates.AutoDateLocator()))
    # ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))  # (잘안됨)
    # ax1.xaxis.set_tick_params(rotation=30, labelsize=10)
    # ax1.format_xdata = mdates.DateFormatter('%Y-%m-%d')
    # fig.autofmt_xdate()

    # plt.xticks(rotation = 30)
    # plt.grid()
    # plt.xlabel('XAXIS')
    # plt.ylabel('YAXIS')
    # plt.title('TITLE')
    # plt.legend(loc='best')
    _ = plt.tight_layout()
    _ = plt.show()
    # _ = fig.savefig('./output.png')


# (21) FUNC :: Date Count
def DateCount_ByWeekdayName(st, en):
    weekdays = collections.Counter()
    st = datetime.datetime.strptime(st, '%Y-%m-%d')
    en = datetime.datetime.strptime(en, '%Y-%m-%d')
    for i in range((en - st).days+1):
        weekdays[(st+datetime.timedelta(i)).strftime("%a")] += 1
    print(weekdays)
    return(weekdays)


# (Z) Run Code
if __name__ == '__main__':
    # print("helo~")
    main()
