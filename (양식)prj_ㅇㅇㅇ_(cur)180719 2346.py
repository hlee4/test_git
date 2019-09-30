# -*- coding: utf-8 -*-
'''
    (설명부분)
'''
# (A) Import Modules
import datetime, glob
import pandas as pd

# pd.set_option('display.expand_frame_repr', False)
# pd.set_option('display.width', 100)
# pd.set_option('display.unicode.east_asian_width', True)


# (B) Define Functions
# (1) FUNC :: main function
def main():
    # (1-0) comments
    print('helo')

    # (1-1) Load Data from Files
    # files = glob.glob('./dn_files/**/*.csv', recursive=True)
    # file = files[-1]
    # print(file)

    # df = pd.read_csv(file, sep=',', header=None)
    # df = pd.read_excel(file, sheet_name=0, headers=True, skiprows=2, usecols=[1,2,3], index_col=0)
    # print(df.head(4))

    # (1-2) Data Munging
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

    # (1-3) Plot Data
    # plot_df(df)

# (2) FUNC :: Plot DateFrame
def plot_df(df):
    # (2-1) Import Modules
    import matplotlib as mpl
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import matplotlib.style as style
    style.use('fivethirtyeight')

    # (2-2) Set Font Style
    font_location = 'c:/Windows/Fonts/malgun.ttf'
    # font_location = '/usr/share/fonts/truetype/msfonts/malgun.ttf'
    font_name = mpl.font_manager.FontProperties(fname=font_location).get_name()
    mpl.rcParams['font.family'] = font_name
    mpl.rcParams['font.size'] = 10

    # (2-3) Plot Graph
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
    # fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(8,6))
    # df['col1'].plot(x=df.index, y='col2', ax=ax1)
    # df.groupby(by='col1').plot(x='col2', y='col3', ax=ax1, legend=True)

    # (Example_3-2)
    # fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(8,6), sharex=True)
    # df['col1'].plot(x=df.index, y='col9', ax=ax1, legend=True)
    # df['col2'].plot(x=df.index, y='col9', ax=ax2, legend=True)
    # df['col3'].plot(x=df.index, y='col9', ax=ax3, legend=True)
    # df['col4'].plot(x=df.index, y='col9', ax=ax4, legend=True)

    # (x-axis ticker formatter settings)
    # ax1.xaxis.set_major_formatter(mticker.NullFormatter())
    # ax1.xaxis.set_major_formatter(mdates.AutoDateFormatter(mdates.AutoDateLocator()))
    # ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    # ax1.xaxis.set_tick_params(rotation=30, labelsize=10)

    # (x-axis ticker locator settings)
    # ax1.xaxis.set_major_locator(mticker.NullLocator())
    # ax1.xaxis.set_major_locator(mticker.AutoLocator())
    # ax1.xaxis.set_major_locator(mticker.MaxNLocator(4))
    # ax1.xaxis.set_major_locator(mdates.YearLocator())
    # ax1.xaxis.set_minor_locator(mdates.MonthLocator())
    # ax1.xaxis.set_minor_locator(mdates.DayLocator())
    # ax1.xaxis.set_minor_locator(mdates.HourLocator(interval=6))

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


# (Z) Run Code
if __name__ == '__main__':
    main()
