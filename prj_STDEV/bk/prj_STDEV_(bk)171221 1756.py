# -*- coding: utf-8 -*-
'''
  (설명부분)
'''
## (1) Import Libraries
import datetime
import numpy as np
import pandas as pd
import pandas_datareader.data as pdr
# import pprint
import matplotlib.pyplot as plt
from matplotlib import style
style.use('fivethirtyeight')
# style.use('ggplot')
# style.use('xkcd')


## (2) Define Functions
## (2-1) Func :: Define Main Function
def main():
    # (2-1-1) Get Data
    company_name = 'HHI'
    company_code = '009540'

    # START = '2016-01-01';    END = '2016-12-31'
    # print('START = %s, END = %s' % (START, END))

    START = datetime.date.today() - datetime.timedelta(180);  START =  START.strftime('%Y-%m-%d')
    END = datetime.date.today() - datetime.timedelta(1);      END = END.strftime('%Y-%m-%d')
    print('START = %s, END = %s' % (START, END))

    # file_name = Download_Data(company_name, company_code, START, END)
    file_name = './out_HHI_(170624~171220).csv'

    df_STOCK = pd.read_csv(file_name)
    print(df_STOCK.head(5))

    # Verify_Data(df_STOCK)


    # (2-1-2) Set Tolerance
    n_row = len(df_STOCK['Close'])
    std_base = np.std(df_STOCK['Close'][0:10])
    std_compare




## (0-0) Func :: 금액에 따라 허용편차를 조정할 수 있도록
def Tolerance(df):
    # funds = 10000000    # 투자금 1000만원
    # affordable_risk = 0.02
    # tolerance = funds * affordable_risk    # 감내할 수 있는 변동률

    print('tolerance = %10.2f' % (tolerance))

    ## (4-3) Set Tolerance
    df_STDEV = pd.DataFrame()
    #df_STDEV = pd.DataFrame(columns=['STDEV'])
    #df_STDEV = pd.DataFrame({'DATE':[], 'STDEV':[]})
    #print(df_STDEV)

    lst_date = [];  lst_stdev = []    # Empty List 생성
    lst_OP = [];  lst_CL = [];  lst_HI = [];  lst_LO = []    # Empty List 생성

    Si = 0

    for Ei in range(1, n_row+1):
        p_mean = np.mean(df['Close'][Si:Ei])
        #print('p_mean = %10.2f' % (p_mean))
        p_stdev = np.std(df['Close'][Si:Ei])
        #print('stdev = %10.2f' % (p_stdev))
        #lst_stdev.append(p_stdev)    # 리스트에 추가

        p_tolerance = p_stdev * ( mean / p_mean );    #print('p_tolerance = %10.2f' % (p_tolerance))

        if (p_tolerance >= tolerance):
            print('%d ~ %d : %6.2f' % (Si, Ei, p_stdev))
            lst_date.append(df.index[Si])    # 리스트에 추가
            lst_stdev.append(p_stdev)
            lst_OP.append(df['Close'][Si])
            lst_CL.append(df['Close'][Ei])
            lst_HI.append(df['Close'][Si:Ei].max())
            lst_LO.append(df['Close'][Si:Ei].min())
            Si = Ei + 1

    df_STDEV['DATE'] = lst_date    # 데이터프레임에 추가
    df_STDEV['STDEV'] = lst_stdev
    df_STDEV['OP'] = lst_OP;    df_STDEV['HI'] = lst_HI;    df_STDEV['LO'] = lst_LO;    df_STDEV['CL'] = lst_CL
    df_STDEV.set_index(df_STDEV['DATE'], inplace=True)    # re-Set Index
    df_STDEV.drop(['DATE'], axis=1, inplace=True)         # Delete Column
    df_STDEV = df_STDEV[['OP','HI','LO','CL','STDEV']]    # re-Order Column
    print(df_STDEV.head(5))
    #print(df_STDEV)



## (2-2) Func :: Get Data from WEB
def Download_Data(company_name, company_code, START, END):
    # df_stock = pdr.DataReader('%s.KS' % (company_code), 'yahoo', START, END)
    df_stock = pdr.get_data_yahoo('%s.KS' % (company_code), START, END)
    # df_stock = pdr.DataReader('KRX:%s' % (company_code), 'google', START, END)
    # df_stock = pdr.get_data_google('KRX:%s' % (company_code), START, END)
    # df_stock = df_stock[df_stock['Volume'] > 0]
        #>> 거래량이 없는 날 제거(공휴일 등) : 이건 해석단계에서 제거하는 것으로 할 것
    print(df_stock.tail(5))

    START = START[2:].replace('-','');    END = END[2:].replace('-','')
    file_name = 'out_' + company_name +'_('+ START +'~'+ END + ').csv'
    df_stock.to_csv(file_name, encoding='cp949')

    return(file_name)


## (2-3) Verify Data
def Verify_Data(df):
    print(df.iloc[[0,-1]])
    # print(df.dtypes)
    print(df.describe())
    print(df.quantile([0.25, 0.5, 0.75]))

    sum = np.sum(df['Close']);      print('sum = %10.4f' % (sum))
    mean = np.mean(df['Close']);    print('mean = %10.4f' % (mean))
    stdev = np.std(df['Close']);    print('stdev = %10.4f' % (stdev))


## (3) Process data
## (3-1) Find Covariance & Correlation (잘됨)
def FN_corr(df):
    data_x = df['Close']
    data_y = df['Volume']
    #>> Covariance [공분산] : 2개 변수의 상관정도
    data_cov = data_x.cov(data_y);    print('[Covariance] %s' % (data_cov))
    #>> Correlation [상관] : 2개 변수간 어떤 선형적 관계가 있는지
    data_corr = data_x.corr(data_y);    print('[Correlation] %s' % (data_corr))

## (3-2) ADF(Augmented Dickey-Fuller) TEST (잘됨)
def FN_ADF(df):
    # 평균회귀 모델 적용을 위해 시계열인지 아닌지 판별하는 식. 즉, 랜덤워크인지 아닌지 판단
    import statsmodels.tsa.stattools as ts
    adf_result = ts.adfuller(df['Close'])
    # (설명) 1st:검정통계량(Test Statistic), 2nd:p-value, 3rd:(??), 4th:데이터 갯수,
    # 5th:가설검정을 위한 1%,5%,10% 기각값(Critical Value), 6th:(???)
    pprint.pprint(adf_result)

## (3-3) Hurst Exponent TEST
def FN_Hurst(df):
    def get_hurst_exponent(df):
        # 기하브라운운동(GBM) 보다 천천히 값이 퍼져나가는지 확인
        # (정상과정은 평균과 표준편차가 일정해서 랜덤워크보다 천천히 확산됨)
        lags = range(2, 100)
        ts = np.log(df)
        tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        result = poly[0] * 2.0
        return result

    data_x = df['Close']
    data_y = df['Volume']
    hurst_df_Cl = get_hurst_exponent(data_x)
    hurst_df_Vo = get_hurst_exponent(data_y)
    print('[Hurst Exponent] Closed=%s' % (hurst_df_Cl))
    print('Hurst Exponent : Closed=%s, Volume=%s' % (hurst_df_Cl, hurst_df_Vo))

## (3-4) Half-life TEST  :: 평균회귀 모델을 적용할 수 있는지 확인
def FN_HalfLife(df):
    def get_half_life(df):    # 비교대상 중 half-life 수치가 클수록 평균회귀 성향이 희박하다는 의미
        price = pd.Series(df)
        lagged_price = price.shift(1).fillna(method='bfill')
        delta = price - lagged_price
        beta = np.polyfit(lagged_price, delta, 1)[0]
        half_life = (-1 * np.log(2) / beta)
        return half_life

    data_x = df['Close']
    data_y = df['Volume']
    half_life_df_Cl = get_half_life(data_x)
    half_life_df_Vo = get_half_life(data_y)
    print('[Half_Life] Close=%s, Volume=%s' % (half_life_df_Cl, half_life_df_Vo))


## (2-1) Draw Graph(1) :: Draw simple graph
def DrawGraph1():
    ax = df_STDEV['CL'].plot(lw=2)
    plt.show()

## (2-2) Draw Graph(2) :: Draw single graph
def DrawGraph2():
    plt.figure(figsize=(12,8))
    plt.subplot(1,1,1)

    #plt.hist(df['Volume'], bins=100)    # [histtype='step', fill=None]
    #plt.scatter(df.index, df['Close'])
    #plt.plot(df_STDEV.index, df_STDEV['STDEV'], 'b-')    # [X, Y, Style]
    plt.plot(df_STDEV.index, df_STDEV['CL'], 'r-')    # [X, Y, Style]
    plt.plot(df.index, df['Close'], 'g-')    # [X, Y, Style]
    plt.legend(loc='upper left')    # [loc='best', 'upper left', 'center left', ..., fancybox=True, shadow=True]

    xcoords = df_STDEV.index
    for xc in xcoords:
        plt.axvline(x=xc, linewidth=1)    # linestyle='-', color='g'

    xmin = df_STDEV.index.min() - datetime.timedelta(15)
    xmax = df_STDEV.index.max() + datetime.timedelta(15)
    #xmin = df.index.min() - datetime.timedelta(15)
    #xmax = df.index.max() + datetime.timedelta(15)
    plt.xlim((xmin,xmax))

    #plt.title(company_name)
    plt.ylabel('Price')    # [fontsize=14, color='red']
    plt.yscale('log')    # [linear, log, symlog, logit]
    #plt.grid(True)
    #plt.text()    # [60, .025, r'$\mu=100,\ \sigma=15$']
    #plt.annotate('쓰고싶은말', xy=(2, 1), xytext=(3, 1.5), arrowprops=dict(facecolor='black', shrink=0.05))

    plt.show()
    #plt.savefig('proj_Graph02.jpg', dpi=300, format='jpg')

## (5-3) Draw Graph(3) :: Draw multiple graphs in One Page
def DrawGraph3():
    fig = plt.figure(figsize=(24,16));

    ax1 = plt.subplot2grid((4,4),(0,0), rowspan=3, colspan=4)
    #ax1 = fig.add_subplot(2,1,1)
    ax1.plot(df.index, df['Close'])
    #ax1.hist(df['Volume'], bins=100)
    ax1.set_title(company_name)
    ax1.set_ylabel('Close')
    #ax1.set_ylabel('Volume')
    #xmin1 = df.index.min() - datetime.timedelta(15)
    #xmax1 = df.index.max() + datetime.timedelta(15)
    #ax1.set_xlim((xmin1,xmax1))

    ax2 = plt.subplot2grid((4,4),(3,0), rowspan=1, colspan=4)
    #ax2 = fig.add_subplot(2,1,2)
    ax2.bar(df.index, df['Volume'])
    #ax2.scatter(df.index, df['Close'])
    ax2.set_title(company_name)
    ax2.set_ylabel('Volume')
    #xmin2 = df.index.min() - datetime.timedelta(15)
    #xmax2 = df.index.max() + datetime.timedelta(15)
    #ymin2 = df['Volume'].min();
    #ymax2 = df['Volume'].max()
    #ax2.set_xlim((xmin2,xmax2))

    #ax3 = fig.add_subplot(2,1,1)
    #ax3.plot(df.index, df['Volume'], 'bo')    # [X, Y, Style]

    plt.show()
    #plt.gcf().get_size_inches(15,8)
    #plt.savefig('proj_Graph03.jpg', dpi=300, format='jpg')

## (5-4) Draw Graph(4)  :: OHLC plot(잘됨)
def DrawGraph4():
    import matplotlib.finance as matfin
    import matplotlib.ticker as ticker

    fig = plt.figure(figsize=(12,8))    # Create Empty Object with specific size
    ax = fig.add_subplot(1,1,1)

    #matfin.candlestick2_ohlc(ax, df['Open'], df['High'], df['Low'], df['Close'], width=0.5, colorup='red', colordown='blue')
    matfin.candlestick2_ohlc(ax, df_STDEV['OP'], df_STDEV['HI'], df_STDEV['LO'], df_STDEV['CL'], width=0.5, colorup='red', colordown='blue')

    #day_list = []
    #name_list = []
    #for i, day in enumerate(df.index):
    #for i, day in enumerate(df_STDEV['DATE']):
    #    if (day.day == 1):    # 월요일마다 'day.dayofweek == 0'
    #        day_list.append(i)
    #        name_list.append(day.strftime('%m-%d'))    # 요일은 (%a)
    #ax.xaxis.set_major_locator(ticker.FixedLocator(day_list))
    #ax.xaxis.set_major_formatter(ticker.FixedFormatter(name_list))

    #plt.title(company_name)
    #plt.xlabel('Time');        #plt.ylabel('Price')
    xmin = df_STDEV.index.min() - datetime.timedelta(15)
    xmax = df_STDEV.index.max() + datetime.timedelta(15)
    #xmin = df.index.min() - datetime.timedelta(15)
    #xmax = df.index.max() + datetime.timedelta(15)
    #ymin = df['Close'].min();    ymax = df['Close'].max()
    #plt.xlim((xmin,xmax))              # set ONLY X-axis range
    #plt.ylim((ymin,ymax))              # set ONLY Y-axis range
    #plt.axis([xmin,xmax,ymin,ymax])    # set Axes [xmin,xmax,ymin,ymax]
    #plt.axis('equal')                  # set Axes [xmin,xmax,ymin,ymax]
    plt.grid(True)

    plt.show()
    #plt.savefig('proj_Graph04.jpg', dpi=300, format='jpg')

## (5-5) Draw Graph(5)  :: autocorrelation plot(잘됨)
def DrawGraph5():
    from pandas.tools.plotting import scatter_matrix, autocorrelation_plot
    fig, axes = plt.subplots(2,1,figsize=(24,16))
    df['Close'].plot(ax=axes[0], lw=1, style='b-')
    autocorrelation_plot(df['Close'], ax=axes[1], lw=1)
    plt.show()

## (5-6) Draw Graph(6)  :: bootstrap plot
def DrawGraph6():
    from pandas.tools.plotting import bootstrap_plot
    # fig, axes = plt.subplots(2,1,figsize=(24,16))
    # df['Volume'].plot(ax=axes[0], lw=1, style='b-')
    bootstrap_plot(df['Close'], size=10, samples=500, color='grey', lw=1)
    plt.show()

## (5-7) Draw Graph(7) :: Histogram (잘됨)
def DrawGraph7(df):
    plt.figure(figsize=(24,16))
    (n, bins, patched) = plt.hist(df['Open'])
    plt.axvline(df['Open'].mean(), color='red')
    plt.show()
    for index in range(len(n)) :
        print('Bin = %0.f, Frequency = %0.f' % (bins[index], n[index]))

## (5-8) Draw Graph(8) :: 산점도 행렬 (잘됨)
def DrawGraph8(df):
    from pandas.tools.plotting import scatter_matrix
    scatter_matrix(df[['Open', 'High', 'Low', 'Close']], alpha=0.2, figsize=(25,16), diagonal='kde')
    plt.show()

## (5-9) Draw Graph(9)
def DrawGraph9(df):
    df[['Open', 'High', 'Low', 'Close', 'Adj Close']].plot(figsize=(25,16), kind='box')
    plt.show()


## (0) RUN
if __name__ == '__main__':
    main()
