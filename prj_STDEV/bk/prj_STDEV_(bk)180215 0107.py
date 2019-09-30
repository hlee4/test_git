# -*- coding: utf-8 -*-
'''
  (설명부분)
'''
## (A) Import Libraries
import datetime
import glob
import numpy as np
import pandas as pd
import pandas_datareader.data as pdr
# import pprint
import matplotlib.pyplot as plt
from matplotlib import style
style.use('fivethirtyeight')
# style.use('ggplot')
# style.use('xkcd')


## (B) Define Functions
## (1) Func :: Define Main Function
def main():
    # (1-1) Get Data from Web and Save to file
    company_name = 'HHI'
    company_code = '009540'

    today = datetime.date.today()
    START = (today - datetime.timedelta(180)).strftime('%Y-%m-%d')
    END   = (today - datetime.timedelta(1)).strftime('%Y-%m-%d')
    # START = '2016-01-01';    END = '2016-12-31'
    print('> START = %s, END = %s' % (START, END))

    # df = Download_Data(company_name, company_code, START, END)

    # (1-2) Load Data from Files
    files = glob.glob("out_HHI_*.xlsx")
    for file in files:
        df = pd.read_excel(file)
        print(df.head(5))
        # Verify_Data(df)

    # (1-3) Set Personal Info
    init_funds = 100000
    base_stdev =
    if df["Volume" == NaN]:
        print(df["Volume" == NaN])
        continue

    # (2-1-2) Set Tolerance
    # n_row = len(df.Close[df_STOCK.index<5])
    # print('n_row : %s' % (n_row))
    # std_base = np.std(df['Close'][0:23])
    # print('std_base : %s' % (std_base))
    # prt = df.Close[df.index<5]
    # print(prt)
    # std_compare = np.std(df['Close'][0:23])



## (2) Func :: 금액에 따라 허용편차를 조정할 수 있도록
def Tolerance(df):
    # funds = 10000000    # 투자금 1000만원
    # affordable_risk = 0.02
    # tolerance = funds * affordable_risk    # 감내할 수 있는 변동률

    print('tolerance = %10.2f' % (tolerance))

    ## (4-3) Set Tolerance
    # df_STDEV = pd.DataFrame()
    colNames = ['DATE', 'STDEV']
    df_STDEV = pd.DataFrame(columns=colNames)
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
    df = pdr.DataReader('%s.KS' % (company_code), 'yahoo', START, END)
    # df = pdr.get_data_yahoo('%s.KS' % (company_code), START, END)
    # df = pdr.DataReader('KRX:%s' % (company_code), 'google', START, END)
    # df = pdr.get_data_google('KRX:%s' % (company_code), START, END)
    # 거래량 없는 날 제거(공휴일 등) : 이건 해석단계에서 제거할 것
    # df = df_stock[df_stock['Volume'] > 0]
    # print(df.tail(5))

    START = START[2:].replace('-','');    END = END[2:].replace('-','')
    file_name = 'out_' + company_name +'_('+ START +'~'+ END + ').xlsx'
    df.to_excel(file_name, encoding='cp949')
    return(df)

## (2-3) Func :: Verify Data
def Verify_Data(df):
    print(df.iloc[[0,-1]])
    # print(df.dtypes)
    print(df.describe())
    print(df.quantile([0.25, 0.5, 0.75]))

    sum = np.sum(df['Close']);      print('sum = %10.4f' % (sum))
    mean = np.mean(df['Close']);    print('mean = %10.4f' % (mean))
    stdev = np.std(df['Close']);    print('stdev = %10.4f' % (stdev))


## (3) Func :: Process data
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





## (0) RUN
if __name__ == '__main__':
    main()
