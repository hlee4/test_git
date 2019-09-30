# -*- coding: utf-8 -*-
'''
    (설명부분)
'''
# (A) Import Modules
import collections, datetime, glob, timeit
import numpy as np
import pandas as pd
import matplotlib as mpl
# import matplotlib.dates as mdates
import matplotlib.pyplot as plt
# import matplotlib.ticker as mticker
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

# (numpy error)
np.seterr(all='ignore')


# (B) Define Functions
# (1) FUNC :: main function
def main():
    # (1-1) Load Data from Files
    files = glob.glob('./dn_files/**/*.csv', recursive=True)
    file = files[-1]
    print(file)

    df = pd.read_csv(file, sep=',', skiprows=[1], nrows=50)
    # df = pd.read_excel(file, sheet_name=0, headers=True, skiprows=2,
    #                    usecols=[1,2,3])
    print(df.head(4))
    # return()

    # (1-2) Data Munging :: ㅇㅇㅇ
    # raw_data = {
    #     'date': pd.date_range(start='2018-01-01', periods=60),
    #     'val00': np.random.randint(low=0, high=100, size=50)
    #     'val00': range(50,0,-1)
    # }
    # df = pd.DataFrame(data=raw_data)

    # df['val3'] = df['val00'].cumsum(axis=0)
    # df['val4'] = df['val00'].rolling(window=2).sum()
    Std_Mean_Ratio(df)

    # base = np.std(df['val00'][0:3]) / np.mean(df['val1'][0:3])

    # for i in range(1,df['val1'].size):
    #     for j in range(1,df['val1'].size):
    #         # print(df['val1'][i:i+j])
    #         cmp = np.std(df['val1'][i:i+j])/np.mean(df['val1'][i:i+j])
    #         if cmp <= base:
    #             print(i, i+j, cmp)
    #             # continue

    # stdev = np.std(resp[i:i+5])
    # lower_stdev = np.std(resp[:i])
    # upper_stdev = np.std(resp[i+1:])
    # diff = lower_stdev - upper_stdev
    # rms = ((lower_stdev - upper_stdev)**2)**0.5

    # lst_ref = sorted(ref_std.items())   # 키를 기준으로 정렬
    # x1,y1 = zip(*lists_ref)             # unpack a list of pairs into two tuples

    # df.to_csv('./df_neat.csv', encoding='cp949')

    # (1-3) Plot

    # (1-4) 참고자료
    # lst = [range(0,10)];        # print(lst)
    # lst = list(range(0,10));    # print(lst)
    # lst1 = [*range(0,11)];      # print(lst1)

    # DateCount_ByWeekdayName(st, en)
    return()


# (2) FUNC :: Calculate Std/Mean ratio
def Std_Mean_Ratio(df):
    df['val00'] = df['Adj Close']

    # (2-1) ㅇㅇㅇ :: (아래는 전부 잘됨)
    # df['val01'] = pd.Series(df['val00'][0:i].tolist()
    #                        for i in range(1, df['val00'].size+1))
    # df['val01'] = [df['val00'][0:i].tolist() for i in range(1, df['val00'].size+1)]
    # df['val02'] = [np.std(i) for i in df['val01']]
    # df['val03'] = [np.mean(i) for i in df['val01']]
    # df['val04'] = df['val02'] / df['val03']
    # print(df[['val01','val02','val03','val04']].tail(5))
    # return()

    # (2-2) ㅇㅇㅇ :: (아래는 전부 잘됨)
    # df['val11'] = [[df['val00'][0+j:i].tolist()
    #                for i in range(1+j, df['val00'].size+1)]
    #                for j in range(df['val00'].size)]                      # (잘됨)
    df['val11'] = pd.Series([df['val00'][0+j:i].tolist()
                            for i in range(1+j, df['val00'].size+1)]
                            for j in range(df['val00'].size))               # (잘됨)
    df['val12'] = pd.Series([np.std(i) for i in j] for j in df['val11'])   # (잘됨)
    df['val13'] = pd.Series([np.mean(i) for i in j] for j in df['val11'])  # (잘됨)
    df['val14'] = pd.Series([x/y for x, y in zip(df['val12'][i], df['val13'][i])]
                            for i in range(df['val12'].size))            # (잘됨)
    # print(df[['val11','val12','val13','val14']].tail(5))
    # return()

    fig, (ax1) = plt.subplots(nrows=1, ncols=1, figsize=(10,8))
    df['val14'].apply(pd.Series).plot(kind='bar', ax=ax1)  # (잘됨)

    # fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(10,8))
    # df[['val12','val13']].unstack().apply(pd.Series).plot(kind='line', ax=ax1)  # (잘됨)
    # df['val12'].apply(pd.Series).plot(kind='line', ax=ax1)  # (잘됨)
    # df['val13'].apply(pd.Series).plot(kind='line', ax=ax2)  # (잘됨)
    # df['val14'].apply(pd.Series).plot(kind='bar', ax=ax3)  # (잘됨)
    plt.show()
    return(df['val14'])

    # (2-3) ㅇㅇㅇ :: (아래는 전부 잘됨) ; 그런데 시간이 오래걸리네
    # lst1 = [[df['val00'][0+j:i].tolist()
    #         for i in range(1+j, df['val00'].size+1)]
    #         for j in range(df['val00'].size)]        # (잘됨)
    # lst2 = [[np.std(x) for x in y] for y in lst1]   # (잘됨)
    # lst3 = [[np.mean(x) for x in y] for y in lst1]  # (잘됨)
    # lst4 = [[x/y for x,y in zip(lst2[i], lst3[i])]
    #           for i in range(len(lst2))]            # (잘됨)
    # print(lst1, '\n', lst2, '\n', lst3, '\n', lst4)
    # return()

    # fig, (ax1) = plt.subplots(nrows=1, ncols=1, figsize=(10,8))
    # ax1.plot(lst4[20], lst[15], lst4[10], lst4[5])
    # plt.show()
    # return()


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

# (22) FUNC :: Runtime Check
def RuntimeCheck(func):
    def wrapper(*args, **kwargs):
        import timeit
        start = timeit.default_timer()
        func
        end = timeit.default_timer()
        print(end - start)
        return func()
    return wrapper

# (Z) Run Code
if __name__ == '__main__':
    # print("helo~")
    main()
