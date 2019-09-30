# -*- coding: utf-8 -*-
'''
    (설명부분)
'''
# (A) Import Modules
import collections, datetime, glob
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
    # files = glob.glob('./dn_files/**/*.xls')
    # file = files[-1]
    # print(file)
    # df = pd.read_excel(file, sheet_name=0, header=0, usecols="A:O")
    # return()

    # (1-2) Data Munging
    # DataMunging_irregular()
    DataMunging_regular()

    # (1-3) Plot
    # plot_01_NumberOfOutgoingCars_ByVenue(df)
    # plot_02_NumberOfOutgoingCars_ByTime(df)
    # plot_03_NumberOfOutgoingCars_ByVenue_Weekday_Time(df)
    # plot_04_NumberOfOutgoingCars_ByVenueDay(df)
    # plot_05_NumberOfOutgoingCars_ByWeekday(df)   #(안됨)
    # plot_06_NumberOfOutgoingCars_ByPlate(df)
    # plot_07_EarnOfOutgoingCars_ByVenue(df)
    # plot_08_FareTimeSeries01_byVenue(df)
    # plot_09_FareTimeSeries02_byVenue(df)
    # plot_10_SeabornViolinPlot(df)

    # (1-4) 참고자료
    # weekdays = collections.Counter()
    # date_from = datetime.datetime(2018,4,20)
    # date_to = datetime.datetime(2018,4,21)
    # for i in range((date_to - date_from).days+1):
    #     weekdays[(date_from+datetime.timedelta(i)).strftime("%a")] += 1
    # print(weekdays)


# (2) Data Munging :: 일반차량(irregular)
def DataMunging_irregular():
    # (2-1) Concatenate Data :: 일반차량(irregular)
    # files = glob.glob('./dn_files/180817 ACC주차장 일반차량 입출차자료(170901~180731)_입차만/*입차.xlsx')
    # print(files)
    # df = pd.concat((pd.read_excel(file, sheet_name=0, header=0, skiprows=1, skipfooter=2, usecols='A:N') for file in files))
    # df.sort_values(by=['입차일시','주차권'], inplace=True)
    # df.reset_index(drop=True, inplace=True)
    # print(df.tail(20))
    # # df.to_csv('./df_neat_irregular.csv', index=False, encoding='cp949')
    # return()

    # (2-2) Load Data from Files
    file = './df_neat_irregular.csv'
    print(file)
    df = pd.read_csv(file, encoding='cp949')

    # (2-3) Dataframe Manupulation
    df['입차일시']= pd.to_datetime(df['입차일시'],format='%Y-%m-%d %H:%M:%S',errors='coerce')
    df['출차일시']= pd.to_datetime(df['출차일시'],format='%Y-%m-%d %H:%M:%S',errors='coerce')
    df['정산일시']= pd.to_datetime(df['정산일시'],format='%Y-%m-%d %H:%M:%S',errors='coerce')
    # df['날짜'] = df['입차일시'].dt.date
    # df['시간대'] = df['입차일시'].dt.hour
    # df['요일코드'] = df['입차일시'].dt.weekday
    # df['요일명'] = df['입차일시'].dt.weekday_name
    df['날짜'] = df['출차일시'].dt.date
    df['시간대'] = df['출차일시'].dt.hour
    df['요일코드'] = df['출차일시'].dt.weekday
    df['요일명'] = df['출차일시'].dt.weekday_name

    # print(df.head(20))
    # print(df.dtypes)
    # print(df.describe())
    # print(df.iloc[:,:])

    # (2-4) Data Munging :: 전체기간(170901~180731)동안 요일별 시간대별 입차량
    # df = df.groupby(['요일코드','요일명','시간대'])['주차권'].count().\
    #     reset_index(name='총입차량')
    df = df.groupby(['요일코드','요일명','시간대'])['주차권'].count().\
        reset_index(name='총출차량')
    df.set_index(['요일코드','요일명','시간대'], inplace=True)
    df = df.unstack(level=[0,1], fill_value=0)
    df.columns = df.columns.droplevel(level=[0,1])
    print(df)
    # df.to_csv('./df_result_irregular_(170901~180731).csv', encoding='cp949')
    return(df)

    # (2-5) Data Munging :: 전체기간(180421~180422)동안 시간대별 입차량
    # df.set_index(['입차일시'], inplace=True)
    df.set_index(['출차일시'], inplace=True)
    df = df[(df.index >= '2018-4-20') & (df.index < '2018-4-22')]
    df.reset_index(inplace=True)
    df = df.groupby(['요일코드','요일명','시간대'])['주차권'].count().\
        reset_index(name='총입차량')
    df.set_index(['요일코드','요일명','시간대'], inplace=True)
    df = df.unstack(level=[0,1], fill_value=0)
    df.columns = df.columns.droplevel(level=[0,1])
    print(df)
    # df.to_csv('./df_result_irregular_(180420~180421).csv', encoding='cp949')
    return(df)


# (3) Data Munging :: 정기차량(regular)
def DataMunging_regular():
    # (3-1) Concatenate Data :: 정기차량(regular)
    # files = glob.glob('./dn_files/180820 ACC주차장 정기차량 입출차기록(170901~180731)_입차만/*.xlsx')
    # print(files)
    # df = pd.concat((pd.read_excel(file, sheet_name=0, header=0, skiprows=13, skipfooter=8, usecols=[0,2,3,5,6,8,9,11,12,14,15]) for file in files))
    # df.sort_values(by=['입차일시','정기권번호'], inplace=True)
    # df.reset_index(drop=True, inplace=True)
    # # print(df.head(5))
    # # print(df.columns.tolist())
    # df.rename(columns={df.columns[6]:'출차 장비', df.columns[8]:'사용기간(시작)', \
    #     df.columns[9]:'사용기간(종료)', df.columns[10]:'종별'}, inplace=True)
    # print(df.head(10))
    # # print(df.columns.tolist())
    # # df.to_csv('./df_neat_regular.csv', index=False, encoding='cp949')
    # return()

    # (3-2) Load Data from Files
    file = './df_neat_regular.csv'
    print(file)
    df = pd.read_csv(file, encoding='cp949')
    # print(df.head(20))
    # return()

    # (3-3) Dataframe Manupulation
    df['입차일시'] = pd.to_datetime(df['입차일시'],format='%Y-%m-%d %H:%M:%S',errors='coerce')
    # df['날짜'] = df['입차일시'].dt.date
    # df['시간대'] = df['입차일시'].dt.hour
    # df['요일코드'] = df['입차일시'].dt.weekday
    # df['요일명'] = df['입차일시'].dt.weekday_name
    df['날짜'] = df['출차일시'].dt.date
    df['시간대'] = df['출차일시'].dt.hour
    df['요일코드'] = df['출차일시'].dt.weekday
    df['요일명'] = df['출차일시'].dt.weekday_name

    # print(df.head(20))
    # print(df.dtypes)
    # print(df.describe())
    # print(df.iloc[:,:])
    # return()

    # (3-4) Data Munging :: 전체기간(170901~180731)동안 요일별 시간대별 입차량
    # df = df.groupby(['요일코드','요일명','시간대'])['단체명'].count().\
    #     reset_index(name='총입차량')
    df = df.groupby(['요일코드','요일명','시간대'])['단체명'].count().\
        reset_index(name='총출차량')
    df.set_index(['요일코드','요일명','시간대'], inplace=True)
    df = df.unstack(level=[0,1], fill_value=0)
    df.columns = df.columns.droplevel(level=[0,1])
    print(df)
    # df.to_csv('./df_result_regular_(170901~180731).csv', encoding='cp949')
    return(df)

    # (3-5) Data Munging :: 전체기간(180421~180422)동안 시간대별 입차량
    # df.set_index(['입차일시'], inplace=True)
    df.set_index(['출차일시'], inplace=True)
    df = df[(df.index >= '2018-4-20') & (df.index < '2018-4-22')]
    df.reset_index(inplace=True)
    # df = df.groupby(['요일코드','요일명','시간대'])['단체명'].count().\
    #     reset_index(name='총입차량')
    df = df.groupby(['요일코드','요일명','시간대'])['단체명'].count().\
    reset_index(name='총출차량')
    df.set_index(['요일코드','요일명','시간대'], inplace=True)
    df = df.unstack(level=[0,1], fill_value=0)
    df.columns = df.columns.droplevel(level=[0,1])
    print(df)
    # df.to_csv('./df_result_regular_(180420~180421).csv', encoding='cp949')
    return(df)


# (11) Plot :: 출구별 출차대수 :: (잘됨)
def plot_01_NumberOfOutgoingCars_ByVenue(df):
    df = df.pivot_table('iID','정산일시','출구장비명칭').resample('1H',level=0).count()
    print(df.head(10))
    df.to_csv('./df_result.csv', encoding='cp949')

    fig,(ax1) = plt.subplots(nrows=1, ncols=1, figsize=(10,8), sharex=True)
    df['B주차장요금계산PC'].plot(kind='line', ax=ax1)
    ax1.set_xlim(['2018-01-01','2018-03-31'])
    # ax1.xaxis_date()
    # ax1.xaxis.set_major_locator(mdates.MonthLocator())
    # ax1.xaxis.set_minor_locator(mdates.DayLocator())

    # years = mdates.YearLocator()
    # months = mdates.MonthLocator()
    # yearsFmt = mdates.DateFormatter('%Y-%m-%d')

    # ax1.xaxis.set_major_locator(mdates.MonthLocator())
    # ax1.xaxis.set_major_formatter(mticker.FixedFormatter('%y\n%m'))
    # ax1.format_xdata = mdates.DateFormatter('%Y-%m-%d')
    # ax1.xaxis.set_minor_locator(mdates.DayLocator())
    # ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))  # (잘안됨)
    # ax1.xaxis.set_major_formatter(mticker.FixedFormatter('%y\n%m-%d'))
    _ = plt.legend(loc='best')
    _ = plt.tight_layout()
    _ = plt.show()
    # _ = fig.savefig('./output.png')


# (12) Plot :: 시간대별 출차량 조사 :: (잘됨)
def plot_02_NumberOfOutgoingCars_ByTime(df):
    df.set_index(['정산일시'], inplace=True)
    df.index = df.index.floor('4H')
    df = df['iID'].groupby([df.index]).count().reset_index(name='count')
    # df = df.groupby([df.index.hour])['iID'].count().reset_index(name='count')
    # df.set_index(['정산일시'], inplace=True)
    print(df.head(10))
    # df.to_csv('./df_neat.csv', encoding='cp949')

    fig,(ax1) = plt.subplots(nrows=1, ncols=1, figsize=(10,8), sharex=True)
    df['count'].plot(x='정산일시', kind='line', ax=ax1, color='b', alpha=0.5)
    # ax1.xaxis.set_major_locator(mdates.DayLocator())
    # ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
    # ax1.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MONDAY))
    # ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    # ax1.fmt_xdata = mdates.DateFormatter('%Y\n%m-%d')
    # ax1.xaxis.set_major_formatter(mticker.FixedFormatter(df.index.strftime('%Y\n%m-%d')))
    # ax1.xaxis.set_tick_params(rotation=30, labelsize=10)

    # ax1.xaxis_date()      # (Tell matplotlib to interpret the x-axis values as dates)
    # fig.autofmt_xdate()   # (make space for and rotate the x-axis tick labels)
    _ = plt.legend(loc='best')
    _ = plt.tight_layout()
    _ = plt.show()
    # _ = fig.savefig('./output.png')


# (13) Plot :: 요일별,시간대별 출차량 분석 :: (잘됨)
def plot_03_NumberOfOutgoingCars_ByVenue_Weekday_Time(df):
    fig,(ax1) = plt.subplots(nrows=1, ncols=1, figsize=(10,8), sharex=True)
    df.groupby(['요일코드','요일명'])['출입차량수'].plot(kind='bar', ax=ax1, alpha=0.5, stacked=True)

    # ax1.xaxis.set_major_locator(mticker.MaxNLocator(7))
    # ax1.xaxis.set_major_locator(mticker.AutoLocator())
    # ax1.xaxis.set_major_locator(mdates.DayLocator(bymonthday=2))
    # ax1.xaxis.set_major_formatter(mdates.AutoDateFormatter(mdates.AutoDateLocator()))
    ax1.xaxis.set_tick_params(rotation=30, labelsize=10)
    _ = plt.legend(loc='best')
    _ = plt.tight_layout()
    _ = plt.show()
    # _ = fig.savefig('./output.png')


# (14) Plot :: 출구별 차량번호별 주차요금 합계
def plot_04_NumberOfOutgoingCars_ByVenueDay(df):
    df = df.groupby(['출구장비명칭', '차량번호'], as_index=False)['주차요금'].sum()
    print(df.head(10))
    # df.to_csv('./df_neat.csv', encoding='cp949')

    fig,(ax1) = plt.subplots(nrows=1, ncols=1, figsize=(10,8), sharex=True)
    df.plot(kind='line', ax=ax1)
    _ = plt.legend(loc='best')
    _ = plt.tight_layout()
    _ = plt.show()
    # _ = fig.savefig('./output.png')


# (15) Plot :: 요일별,시간대별  출차대수 분석
def plot_05_NumberOfOutgoingCars_ByWeekday(df):
    df = df.groupby('weekday_name').get_group('Monday')
    print(df.head(10))
    # df.to_csv('./df_neat.csv', encoding='cp949')

    fig,(ax1) = plt.subplots(nrows=1, ncols=1, figsize=(10,8), sharex=True)
    df.plot(kind='line', ax=ax1)
    _ = plt.legend(loc='best')
    _ = plt.tight_layout()
    _ = plt.show()
    # _ = fig.savefig('./output.png')


# (16) Plot :: 차량별 출차횟수 분석 :: (잘됨)
def plot_06_NumberOfOutgoingCars_ByPlate(df):
    df = df.groupby(['차량번호'], as_index=False)['iID'].count()  # (잘됨)
    df.rename(columns={'iID':'count'}, inplace=True)
    print(df.head(10))
    # df.to_csv('./df_neat.csv', encoding='cp949')

    fig,(ax1) = plt.subplots(nrows=1, ncols=1, figsize=(10,8), sharex=True)
    df['count'].hist(bins=30, ax=ax1)
    _ = plt.legend(loc='best')
    _ = plt.tight_layout()
    _ = plt.show()
    # _ = fig.savefig('./output.png')


# (17) Plot :: 일자별,출구별 주차요금 OHLC
def plot_07_EarnOfOutgoingCars_ByWeekday(df):
    from mpl_finance import candlestick_ohlc

    df = df.pivot_table('주차요금','정산일시','출구장비명칭').resample('D',level=0).ohlc()
    print(df.head(10))
    # df.to_csv('./df_neat.csv', encoding='cp949')

    fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(10,8), sharex=False)
    candlestick_ohlc(ax1, df.open, df.high, df.low, df.close)
    _ = plt.legend(loc='best')
    _ = plt.tight_layout()
    _ = plt.show()
    # _ = fig.savefig('./output.png')


# (18) FUNC :: Plot DataFrame using matplotlib
def plot_08_FareTimeSeries01_byVenue(df):
    df.set_index(['정산일시'], inplace=True)
    df = df.groupby(['출구장비명칭'])['주차요금'].resample('4H').sum()
    print(df.head(10))

    # (사례1) :: ".plot plots the index against every column"
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    df['A주차장요금계산PC'].plot(kind='line', ax=ax1)
    df['B주차장요금계산PC'].plot(kind='line', ax=ax1)
    df['지하요금계산PC'].plot(kind='line', ax=ax1)
    df['버스요금계산PC'].plot(kind='line', ax=ax1)

    # ax1.xaxis.set_major_locator(mdates.DayLocator())
    # ax1.xaxis.set_major_locator(mdates.DayLocator(bymonthday=5, interval=10))
    # ax1.xaxis.set_major_formatter(mticker.NullFormatter())
    # ax1.xaxis.set_major_formatter(mticker.FixedFormatter('%y\n%m-%d'))
    # ax1.xaxis.set_tick_params(rotation=30, labelsize=10)

    _ = plt.legend(loc='best')
    _ = plt.tight_layout()
    _ = plt.show()
    # _ = fig.savefig('./output.png')


# (19) FUNC :: Plot DataFrame using matplotlib
def plot_09_FareTimeSeries02_byVenue(df):
    df.set_index(['정산일시'], inplace=True)
    df = df.groupby(['출구장비명칭'])['주차요금'].resample('D').sum()
    print(df.head(10))

    # (사례3) :: ".plot(x='col1',y='col2') plots one specific column against another specific column"
    fig,(ax1,ax2,ax3,ax4) = plt.subplots(nrows=4, ncols=1, figsize=(10,8), sharex=True)
    df['A주차장요금계산PC'].plot(x=df.index, ax=ax1, legend=True, color='red')
    df['B주차장요금계산PC'].plot(x=df.index, ax=ax2, legend=True, color='green')
    df['지하요금계산PC'].plot(x=df.index, ax=ax3, legend=True, color='blue')
    df['버스요금계산PC'].plot(x=df.index, ax=ax4, legend=True, color='yellow')
    _ = plt.legend(loc='best')
    _ = plt.tight_layout()
    _ = plt.show()
    # _ = fig.savefig('./output.png')


# (20) FUNC :: Plot DataFrame using seaborn
def plot_10_SeabornViolinPlot(df):
    # (2-1) Import Modules
    import seaborn as sns
    import matplotlib.pyplot as plt
    sns.set(style='whitegrid')

    # (2-3) Plot Graph
    fig,(ax1) = plt.subplots(nrows=1, ncols=1, figsize=(10,8), sharex=True)
    ax1 = sns.violinplot(data=df, palette='Set3')
    sns.despine(left=True, bottom=True)
    _ = plt.legend(loc='best')
    _ = plt.tight_layout()
    _ = plt.show()
    # _ = fig.savefig('./output.png')


# (Z) Run Code
if __name__ == '__main__':
    # print("helo~")
    main()
