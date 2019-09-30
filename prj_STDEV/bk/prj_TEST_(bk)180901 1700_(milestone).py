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
    # df = pd.read_excel(file, sheet_name=0, headers=True, skiprows=2, usecols=[1,2,3], index_col=0)
    # print(df.head(4))

    raw_data = {
        # 'date': pd.date_range(start='2018-01-01', periods=60),
        'val1': np.random.randint(low=0, high=100, size=60),
        'val2': range(60)
    }
    df = pd.DataFrame(data=raw_data)

    df['val3'] = df['val2'].cumsum(axis=0)
    df['val4'] = df['val3'].rolling(window=2).sum()
    df['val5'] = pd.Series([df['val1'][0:i].tolist() for i in range(1,df['val1'].size)])
    df['val6'] = [np.std(x) for x in df['val5']]
    print(df.head(5))

    # lst = [range(0,10)];        # print(lst)
    # lst = list(range(0,10));    # print(lst)
    # lst1 = [*range(0,11)];      # print(lst1)

    return()

    # base = np.std(resp[0:3]) / np.mean(resp[0:3])

    for i in range(len(resp)):
        # print(i, end=" ")
        # if (int(i) == 0) or (int(i+1) >= len(resp)):
        #     continue
        # print(resp[i:i+5])

        for j in range(1,len(resp)):
            print(resp[i:i+j])
            std_03 = np.std(resp[i:i+j])/np.mean(resp[i:i+j])
            if std_03 >= base:
                print(std_03)
                continue

    # stdev = np.std(resp[i:i+5])
    # lower_stdev = np.std(resp[:i])
    # upper_stdev = np.std(resp[i+1:])
    # diff = lower_stdev - upper_stdev
    # rms = ((lower_stdev - upper_stdev)**2)**0.5

    # lst = [i, ref_stdev, lower_stdev, upper_stdev, diff, rms]
    # series = pd.Series(lst, index=colNames)    # (잘됨)
    # # print(series)
    # df = df.append(series, ignore_index=True)

    # ref.update({i:ref_std})    # dictionary에 추가할 때

    # print(df.head(20))

    # lst_ref = sorted(ref_std.items())     # 키를 기준으로 정렬
    # x1,y1 = zip(*lists_ref)             # unpack a list of pairs into two tuples




    # (1-2) Data Munging :: ㅇㅇㅇ
    # df.dropna(axis=0, inplace=True)  # NaN값이 있는 행 제거
    # df['col1'] = df['col1'].str.replace(pat='\s+', repl='')  # 'col1'에서 공백제거
    # df.drop(['col7', 'col8'], axis=1, inplace=True)
    # df['col1'] = pd.to_datetime(df['col1'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
    # df['col2'] = pd.to_numeric(df['col2'], errors='coerce')
    # df = df[['col1', 'col2', 'col3', 'col4']]
    # df.set_index(['col1', 'col2','col3'], inplace=True)
    # df = df.pivot_table('col3','col1','col2').resample('D',level=0).sum()

    # print(df.head(4))
    # print(df.iloc[0:3,0:10])
    # print(df.describe())
    # print(df.dtypes)

    # df.to_csv('./df_neat.csv', encoding='cp949')

    # print(df.loc[(df.index>='2000-1-1') & (df.index<='2000-1-31')])
    # print(df.query('col2>800'))    # (잘됨)
    # print(df[df['col2']>800])      # (잘됨) same result as the previous expression

    # (1-3) Plot
    # plot_01_STDEV(df)

    # plt.figure(num=1, figsize=(6,4))    # make the first figure
    # plt.subplot(2,1,1)                  # the first subplot in the first figure
    # plt.axhline(basis_stdev, linestyle="--", color="m", )
    # plt.plot(df["number"], df["ref_stdev"], "k.")
    # plt.plot(df["number"], df["lower_stdev"], "c.", alpha=0.4)
    # plt.plot(df["number"], df["upper_stdev"], "g.", alpha=0.4)
    # # plt.axis([0,100,-50,50])    # [Xmin, Xmax, Ymin, Ymax]

    # plt.subplot(2,1,2)                  # the second subplot in the first figure
    # plt.plot(df["number"], df["diff"], "r+", alpha=0.4)
    # plt.plot(df["number"], df["rms"], "b+", alpha=0.4)

    # # plt.figure(num=2, figsize=(6,4))        # make the second figure
    # # plt.subplot(1,1,1)                      # the first subplot in the second figure
    # # plt.plot(df["number"], df["rms"], "b+")
    # # plt.axis([0,100,-50,50])

    # # plt.figure(num=1)    # figure1 current; subplot(212) still current
    # # plt.subplot(211)     # make subplt(211) in figure1 current
    # plt.show()

    # (1-4) 참고자료
    # st = '2018-01-01'
    # en = '2018-06-30'
    # datecount = DateCount_ByWeekdayName(st, en)
    # print(datecount)

    return()


# (11) FUNC :: Plot DateFrame
def plot_01_STDEV(df):
    # (사례1) :: ".plot plots the index against every column"
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    df.plot(kind='line', ax=ax1)
    # df[['col1','col2']].plot(kind='line', ax=ax1)
    # df['col2'].groupby('col1').mean().sort_values().plot(kind='barh', ax=ax1)

    # (사례2) :: ".plot(x='col1') plots against a single specific column"
    # df['col2'].plot(x='col1')

    # (사례3) :: ".plot(x='col1',y='col2') plots one specific column against another specific column"
    # (Example_3-1)
    # fig, (ax1) = plt.subplots(nrows=1, ncols=1, figsize=(8,6))
    # df['col1'].plot(x=df.index, y='col2', ax=ax1)
    # df.groupby(by='col1').plot(x='col2', y='col3', ax=ax1, legend=True)

    # (Example_3-2)
    # fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(8,6), sharex=True)
    # df['col1'].plot(x=df.index, y='col9', ax=ax1, legend=True)
    # df['col2'].plot(x=df.index, y='col9', ax=ax2, legend=True)
    # df['col3'].plot(x=df.index, y='col9', ax=ax3, legend=True)
    # df['col4'].plot(x=df.index, y='col9', ax=ax4, legend=True)

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
