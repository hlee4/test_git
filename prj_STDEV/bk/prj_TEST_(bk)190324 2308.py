# -*- coding: utf-8 -*-
'''
    (설명부분)
'''
# (A) Import Modules
import datetime, glob
import matplotlib as mpl
# import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.style as style
# import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# (Matplotlib Style Settings)
style.use('fivethirtyeight')

# (Matplotlib Font Settings)
font_location = 'c:/Windows/Fonts/malgun.ttf'
# font_location = '/usr/share/fonts/truetype/msfonts/malgun.ttf'
font_name = mpl.font_manager.FontProperties(fname=font_location).get_name()
mpl.rcParams['font.family'] = font_name
mpl.rcParams['font.size'] = 12

# (numpy error)
np.seterr(all='ignore')

# (Pandas Display options)
pd.set_option('display.max_columns', 20)
# pd.set_option('max_colwidth', 18)
# pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.width', 100)
pd.set_option('display.unicode.east_asian_width', True)

# (Pandas Format of the Number)
# pd.set_option('display.float_format', '{:.2f}'.format)   # float format
# pd.set_option('display.float_format', '{:.2e}'.format)   # scientific format
# pd.set_option('display.float_format', '${:.2g}'.format)  # general format(적당히)
# pd.set_option('display.float_format', None)

# (B) Define Functions
# (1) FUNC :: main function
def main():
    # (1-1) Data Cleaning :: 단정한 데이터로 변환
    # func_DataCleaning('opt')
    # return()

    # (1-2) Load Data from Files
    files = glob.glob('./dn_files/**/*.csv', recursive=True)
    file = files[-1]
    print('> {}'.format(file))

    # df_orig = pd.read_excel(file, sheet_name=0, headers=True, skiprows=2, usecols=[1,2,3])
    df_orig = pd.read_csv(file, sep=',', skiprows=[1], nrows=50)
    print('> dataframe(orig)\n{}\n'.format(df_orig.head()))
    # return()

    # (1-3) Data Preprocessing
    df_neat = func_Preprocessing(df_orig)
    # print('> {}'.format(df_neat.head()))
    return()

    # (1-4) Data Analysis :: ㅇㅇㅇ
    # df_smr = func_StdMeanRatio(df_neat)
    # print(df_smr[10])

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

    # Plot1 :: ㅇㅇㅇ
    # plot_ㅇㅇㅇ()
    return()

    # (1-4) Data Analysis :: ㅇㅇㅇ

    # (9-1) 참고자료 :: 파이썬 리스트 사용하는 법
    # lst1 = [range(0,8)]
    # lst2 = list(range(0,8))
    # lst3 = [*range(0,8)]
    # print(f'> [range(0,8)]: {lst1}\n> list(range(0,8)): {lst2}\n> [*range(0,8)]: {lst3}')

    # (9-2) 참고자료 :: 일정기간 동안 각 요일별 날짜 수 계산
    # st = '2019-1-1'
    # en = '2019-3-2'
    # conv = func_DateCount_ByWeekdayName(st, en)
    # print(f'> from: {st} ~ to: {en} = {conv}')
    return()


# (11) Cleaning
# def func_DataCleaning(opt):


# (12) Preprocessing :: 일반주차차량 데이터 전처리 :: (잘됨)
def func_Preprocessing(df_orig):
    # (12-1) 기초데이터 설정
    # (아래는 기초데이터를 임의로 생성)
    # df = pd.DataFrame(data={
    #     'date': pd.date_range(start='2018-01-01', periods=50),
    #     'val0': np.random.randint(low=0, high=100, size=50),
    #     'val1': range(50,0,-1)})
    # (아래는 읽어온 데이터에서 기초데이터를 설정)
    df = pd.DataFrame(data={
        'date': df_orig['Date'],
        'val0': df_orig['Adj Close']})
    # print('> dataframe\n{}'.format(df.head()))
    # return()

    # (12-2) 기본함수 적용 사례
    df['val1'] = df['val0'].cumsum(axis=0)
    df['val2'] = df['val0'].rolling(window=2).sum()
    print('> dataframe\n{}\n'.format(df.head()))
    # return()

    # (12-3) 샘플로 첫 다섯자료의 표준편차를 평균으로 나누어 기준값으로 설정
    ref_sigma = pd.Series(df['val0'][0:5]).std() / pd.Series(df['val0'][0:5]).mean()
    print('> ref_sigma: {}\n'.format(ref_sigma))
    # return()

    unit_sigma = pd.Series(
      pd.Series(df['val0'][0:i]).std() / pd.Series(df['val0'][0:i]).mean()
       for i in range(1,10))
    print('> dataframe(unit_sigma)\n{}\n'.format(unit_sigma))
    # pf = df[5]
    # ref_sigma = np.std(ps:pf) / avg(ps:pf)
    # unit_sigma = np.std(pi:pj) / avg(pi:pj)
    return()


# (21) Analysis :: Calculate Std/Mean ratio
def func_StdMeanRatio_01(df):
    (21-1) 시작점 고정하되 하나씩 추가 :: (아래는 전부 잘됨)
    df['val01'] = pd.Series(df['val0'][0:i].tolist()
                           for i in range(1, df['val0'].size+1))
    df['val01'] = [df['val0'][0:i].tolist()
                   for i in range(1, df['val0'].size+1)]
    df['val02'] = [np.std(i) for i in df['val01']]
    df['val03'] = [np.mean(i) for i in df['val01']]
    df['val04'] = df['val02'] / df['val03']
    print(df[['val01','val02','val03','val04']].tail(5))
    return()

# (22) Analysis :: Calculate Std/Mean ratio
# def func_StdMeanRatio_02(df):
    # (2-4) 하나씩 추가하면서 시작점 변동 :: (아래는 전부 잘됨)
    # df['val11'] = [[df['val0'][0+j:i].tolist()
    #                for i in range(1+j, df['val0'].size+1)]
    #                for j in range(df['val0'].size)]                     # OK
    df['val11'] = pd.Series([df['val0'][0+j:i].tolist()
                            for i in range(1+j, df['val0'].size+1)]
                            for j in range(df['val0'].size))              # OK
    df['val12'] = pd.Series([np.std(i) for i in j] for j in df['val11'])   # OK
    df['val13'] = pd.Series([np.mean(i) for i in j] for j in df['val11'])  # OK
    df['val14'] = pd.Series([x/y
                            for x, y in zip(df['val12'][i], df['val13'][i])]
                            for i in range(df['val12'].size))              # OK
    # print(df[['val11','val12','val13','val14']].tail(5))
    # return()

    # fig, (ax1) = plt.subplots(nrows=1, ncols=1, figsize=(10,8))
    # df['val14'].apply(pd.Series).plot(kind='bar', ax=ax1)  # (잘됨)
    # plt.show()

    # fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(10,8))
    # df[['val12','val13']].unstack().apply(pd.Series).plot(kind='line', ax=ax1)  # (잘됨)
    # df['val12'].apply(pd.Series).plot(kind='line', ax=ax1)  # (잘됨)
    # df['val13'].apply(pd.Series).plot(kind='line', ax=ax2)  # (잘됨)
    # df['val14'].apply(pd.Series).plot(kind='bar', ax=ax3)   # (잘됨)
    # plt.show()
    return(df['val14'])

    # (2-5) 하나씩 추가하면서 시작점 변동 :: (아래는 전부 잘됨. 그런데 시간이 오래걸리네)
    # lst1 = [[df['val00'][0+j:i].tolist()
    #         for i in range(1+j, df['val00'].size+1)]
    #         for j in range(df['val00'].size)]        # (잘됨)
    # lst2 = [[np.std(x) for x in y] for y in lst1]    # (잘됨)
    # lst3 = [[np.mean(x) for x in y] for y in lst1]   # (잘됨)
    # lst4 = [[x/y for x,y in zip(lst2[i], lst3[i])]
    #           for i in range(len(lst2))]             # (잘됨)
    # print(lst1, '\n', lst2, '\n', lst3, '\n', lst4)
    # return()

    # fig, (ax1) = plt.subplots(nrows=1, ncols=1, figsize=(10,8))
    # ax1.plot(lst4[20], lst[15], lst4[10], lst4[5])
    # plt.show()
    # return()


# (11) Plot :: ㅇㅇㅇ
def plot_SMR(df):
    fig, (ax1) = plt.subplots(nrows=1, ncols=1, figsize=(10,8))
    # ddf.apply(pd.Series).plot(kind='line', ax=ax1)  # (잘됨)
    ax1.plot(df[10:20].apply(pd.Series))
    plt.show()

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
    # plt.tight_layout()
    # plt.show()
    # _ = fig.savefig('./output.png')


# (91) Func :: Date Count
def func_DateCount_ByWeekdayName(start, end):
    import collections
    weekdays = collections.Counter()
    st = datetime.datetime.strptime(start, '%Y-%m-%d')
    en = datetime.datetime.strptime(end, '%Y-%m-%d')
    for i in range((en - st).days+1):
        weekdays[(st+datetime.timedelta(i)).strftime("%a")] += 1
    print('> from: {} ~ to: {} = {}'.format(start, end, weekdays))
    return(weekdays)


# (92) Func :: Runtime Check
def RTCheck_func(orig_func):
    def wrapper(*args, **kwargs):
        import timeit
        st = timeit.default_timer()
        orig_func()
        en = timeit.default_timer()
        print('runtime is %f' % (en - st))
        return orig_func()
    return wrapper


# (Z) Run Code
if __name__ == '__main__':
    # print("helo~")
    main()
