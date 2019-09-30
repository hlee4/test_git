# -*- coding: utf-8 -*-
'''
    (설명부분)
'''
# (A) Import Modules
import datetime, hashlib, glob
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.style as style
import numpy as np
import pandas as pd

# (Matplotlib Style Settings)
style.use('fivethirtyeight')
# style.use('ggplot')

# (Matplotlib Font Settings)
font_location = 'c:/Windows/Fonts/malgun.ttf'
# font_location = '/usr/share/fonts/truetype/msfonts/malgun.ttf'
font_name = mpl.font_manager.FontProperties(fname=font_location).get_name()
mpl.rcParams['font.family'] = font_name
mpl.rcParams['font.size'] = 8

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
# (01) FUNC :: main function
def main():
    # (1-1) Data Cleaning :: 단정한 데이터로 변환
    # DataCleaning('occasional')    # {일반주차: 'occasional', 정기주차: 'regular'}
    # DataCleaning('regular')
    # return()

    # (1-2) Load Data from File
    files_occ = glob.glob('./dn_files/190109*일반*(180101~181231)*/*일반차량*.csv')
    file_occ = files_occ[-1]
    print('> {}'.format(file_occ))
    # df = pd.read_excel(file_occ, sheet_name=0, header=0, usecols='A:O')
    # df = pd.read_csv(file_occ, nrows=5000, engine='python')
    # (C parser에 문제가 있어서 engine='python'으로 설정)
    df_occ = pd.read_csv(file_occ, engine='python')
    # print('> Dataframe\n{}\n'.format(df_occ.head()))

    files_reg = glob.glob('./dn_files/190218*정기*(180101~181231)*/*정기차량*.csv')
    file_reg = files_reg[-1]
    print('> {}'.format(file_reg))
    # df = pd.read_excel(file_reg, sheet_name=0, header=0, usecols='A:O')
    # df = pd.read_csv(file_reg, nrows=5000, engine='python')
    df_reg = pd.read_csv(file_reg, engine='python')
    # print('> Dataframe\n{}\n'.format(df_reg.head()))
    # return()

    # (1-3) Data Preprocessing
    df_occ = Preprocessing_occ(df_occ)
    df_reg = Preprocessing_reg(df_reg)
    # return()

    # (1-4) Data Analysis :: 주차장 체류시간 분석 :: (잘됨)
    # dur_occ = func_Duration_occ(df_occ)
    # # print('> 체류시간별 체류규모(일반)\n{}\n'.format(dur_occ.head(20)))
    # # dur_occ.to_csv('./ACC_duration_occ.csv', encoding='cp949')
    # # return()

    # dur_occ = dur_occ.droplevel(level=0)
    # # dur_sel = dur_occ.iloc[:, ((dur_occ.columns.get_level_values(0)=='A주차장요금계산PC') & (dur_occ.columns.get_level_values(1)=='현금수입') & (dur_occ.columns.get_level_values(2)=='count'))]
    # # dur_sel = dur_occ.xs(key=['지하요금계산PC','주차요금','count'], axis=1, level=[0,1,2])
    # dur_sel = dur_occ.xs(key=['지하요금계산PC','sum'], axis=1, level=[0,2])
    # print(f'> 체류시간별 체류규모(선택)\n{dur_sel.head(20)}\n')
    # # dur_occ.to_csv('./ACC_duration_occ_sel.csv', encoding='cp949')
    # plot_Duration(dur_sel)
    # return()

    # (1-5) Data Analysis :: 시간대별 체류규모 분석 :: (잘됨)
    cycle = '1H'  # 원하는 주기로 설정 (시간대는 시작값을 말함)

    residue_occ = func_Residue_occ(df_occ, cycle)
    # print(f'> 체류대수(일반)\n{residue_occ.head(20)}\n')
    # fname_occ = './ACC_residue_occ_' + '(' + cycle + ').csv'
    # residue_occ.to_csv(fname_occ, encoding='cp949')

    residue_reg = func_Residue_reg(df_reg, cycle)
    # print(f'> 체류대수(정기)\n{residue_reg.head(20)}\n')
    # fname_reg = './ACC_residue_reg_' + '(' + cycle + ').csv'
    # residue_reg.to_csv(fname_reg, encoding='cp949')

    residue_all = residue_occ.add(residue_reg, fill_value=0)
    # print(f'> 체류대수(합계)\n{residue_all.head(20)}\n')
    # fname_all = './ACC_residue_all_' + '(' + cycle + ').csv'
    # residue_all.to_csv(fname_all, encoding='cp949')

    opt = 'MMS'  # {'MMS': (max,mean,std), 'ohlc': (open,high,low,close)}
    residue_cal = func_Residue_cal(residue_all, opt)
    # print(f'> 체류대수(완료)\n{residue_cal.head(20)}\n')
    # print('> 체류대수(완료)\n{}'.format(residue_cal.query('장소 == "지하체류량"').head(20)))
    # fname_cal = './ACC_residue_cal_' + opt + '(' + cycle + ').csv'
    # residue_cal.to_csv(fname_cal, encoding='cp949')

    # # 그래프1: 3차원(월별,요일별,시간대별)으로 산출한 버전 :: 잘됨
    # # residue_sel = residue_cal.xs(key=['지하체류량', 1, 'Fri'], axis=0, level=[0,2,3])  # 샘플
    # resi_3d = residue_cal.xs(key=['지하체류량'], axis=0, level=[0])
    # # print(f'> 체류대수(3D_all)\n{resi_3d.head(10)}\n')
    # resi_3d = resi_3d.xs(key=['max'], axis=1)
    # # print(f'> 체류대수(3D_max)\n{resi_3d.head(10)}\n')
    # # resi_2d = resi_3d.groupby(level=[0,1], axis=0).agg(['max', 'mean', 'std', ('+2sigma', lambda x: np.mean(x)+2*np.std(x, ddof=1)), ('+3sigma', lambda x: np.mean(x)+3*np.std(x, ddof=1))])  #참고
    # # resi_2d = resi_2d.xs(key=['max'], axis=1, level=[0])
    # # print(f'> 체류대수(2D_all)\n{resi_2d.head(10)}\n')
    # # fname_sel = './ACC_residue_sel_' + opt + '(' + cycle + ').csv'
    # # residue_3d.to_csv(fname_sel, encoding='cp949')
    # plot_Residue_MMS_3D(resi_3d, opt1='월별', opt2='plt')  # [(opt1: 요일별,월별), (opt2: plt,sns)]
    # return()

    # # 그래프2-1: 2차원(월별,요일별,시간대별 중 2개)으로 축소한 버전 :: 잘됨
    # resi_3d = residue_cal.xs(key=['지하체류량'], axis=0, level=[0])
    # resi_3d = resi_3d.xs(key=['max'], axis=1)
    # resi_2d = resi_3d.groupby(level=[0,1], axis=0).agg(['max', 'mean', 'std', ('+2sigma', lambda x: np.mean(x)+2*np.std(x, ddof=1)), ('+3sigma', lambda x: np.mean(x)+3*np.std(x, ddof=1))])
    # resi_2d = resi_2d.xs(key=['max'], axis=1, level=[0])
    # print(f'> 체류대수(2D)_종일\n{resi_2d.head(10)}\n')
    # resi_am = resi_3d.xs(key=[slice(datetime.time(7,0,0),datetime.time(13,0,0))], axis=0, level=[2])
    # resi_am = resi_am.xs(key=['max'], axis=1)
    # resi_am = resi_am.groupby(level=[0,1], axis=0).agg(['max', 'mean', 'std', ('+2sigma', lambda x: np.mean(x)+2*np.std(x, ddof=1)), ('+3sigma', lambda x: np.mean(x)+3*np.std(x, ddof=1))])
    # resi_am = resi_am.xs(key=['max'], axis=1, level=[0])
    # print(f'> 체류대수(2D)_오전\n{resi_am.head(10)}\n')
    # resi_pm = resi_3d.xs(key=[slice(datetime.time(14,0,0),datetime.time(23,0,0))], axis=0, level=[2])
    # resi_pm = resi_pm.xs(key=['max'], axis=1)
    # resi_pm = resi_pm.groupby(level=[0,1], axis=0).agg(['max', 'mean', 'std', ('+2sigma', lambda x: np.mean(x)+2*np.std(x, ddof=1)), ('+3sigma', lambda x: np.mean(x)+3*np.std(x, ddof=1))])
    # resi_pm = resi_pm.xs(key=['max'], axis=1, level=[0])
    # print(f'> 체류대수(2D)_오후\n{resi_pm.head(10)}\n')
    # return()

    # 그래프2-2: 2차원(월별,요일별,시간대별 중 2개)으로 축소한 버전(바로 위와 동일) :: 잘됨
    resi_3d = residue_cal.xs(key=['지하체류량'], axis=0, level=[0])
    resi_3d = resi_3d.xs(key=['max'], axis=1)
    resi_2d = resi_3d.groupby(level=[0,1], axis=0).agg(['max', 'mean', 'std', ('+2sigma', lambda x: np.mean(x)+2*np.std(x, ddof=1)), ('+3sigma', lambda x: np.mean(x)+3*np.std(x, ddof=1))])
    resi_2d = resi_2d.xs(key=['max'], axis=1, level=[0])
    print(f'> 체류대수(2D)_종일\n{resi_2d.head(10)}\n')
    # resi_am = resi_3d.loc[pd.IndexSlice[:,:,datetime.time(7,0,0):datetime.time(13,0,0)],'max']
    # resi_am = resi_am.groupby(level=[0,1], axis=0).agg(['max', 'mean', 'std', ('+2sigma', lambda x: np.mean(x)+2*np.std(x, ddof=1)), ('+3sigma', lambda x: np.mean(x)+3*np.std(x, ddof=1))])
    # print(f'> 체류대수(2D)_오전\n{resi_am.head(10)}\n')
    # resi_pm = resi_3d.loc[pd.IndexSlice[:,:,datetime.time(14,0,0):datetime.time(23,0,0)],'max']
    # resi_pm = resi_pm.groupby(level=[0,1], axis=0).agg(['max', 'mean', 'std', ('+2sigma', lambda x: np.mean(x)+2*np.std(x, ddof=1)), ('+3sigma', lambda x: np.mean(x)+3*np.std(x, ddof=1))])
    # print(f'> 체류대수(2D)_오후\n{resi_pm.head(10)}\n')
    plot_Residue_MMS_2D(resi_2d, opt1='요일별', opt2='plt')  # [(opt1: 요일별,월별), (opt2: plt,sns)]
    return()

    # (1-6) Data Analysis :: 매월 주차장별 요일별 출차량 :: (잘됨)
    # df_out = (df.groupby(['출차장비', df['출차일시'].dt.weekday,
    #           pd.Grouper(key='출차일시', freq='1M')])['주차권'].count())
    # df_out.index.names = ['출차장비', '출차요일', '출차일시']
    # df_out = df_out.unstack(level=[0,1], fill_value=0)
    # print('> 매월 주차장별 요일별 출차량\n{}'.format(df_out.head(20)))
    # print('> 월요일 출차량\n{}'.format(df_out.xs(key=0, level=1, axis='columns')))
    # print('> 화요일 출차량\n{}'.format(df_out.xs(key=1, level=1, axis='columns')))
    # return()

    # (1-7) Data Analysis :: 매일오전 주차장별 요일별 출차량 :: (잘됨)
    # df_am = df[df['출차일시'].dt.time <= datetime.time(12,0,0)]
    # df_am = df_am.pivot_table(index=df_am['출차일시'].dt.weekday, columns='출차장비',
    #                           values='주차권', aggfunc=len, fill_value=0)
    # wdays = {0:'Monday', 1:'Tuesday', 2:'Wednesday', 3:'Thursday',
    #          4:'Friday', 5:'Saturday', 6:'Sunday'}
    # # wdays = {0:'Mon', 1:'Tue', 2:'Wed', 3:'Thu', 4:'Fri', 5:'Sat', 6:'Sun'}
    # df_am.index = df_am.index.map(wdays.get)
    # print('> {}\n{}'.format('매일오전 주차장별 요일별 출차대수', df_am.T))
    # return()

    # (1-8) Data Analysis :: (새로 만들고 싶은 내용은 여기에 작성)
    # ㅇㅇㅇ

    # (1-10) Data Analysis :: Trivial
    # opt = 3  # 주차권이 3회이상 반복되는 자료를 찾을 때
    # func_Overlap(df_occ, opt)
    # return()

    # (9-1) 참고자료 :: 일정기간 동안 각 요일별 날짜 수 계산
    # import collections
    # weekdays = collections.Counter()
    # date_from = datetime.datetime(2018,4,20)
    # date_to = datetime.datetime(2018,4,21)
    # for i in range((date_to - date_from).days+1):
    #     weekdays[(date_from+datetime.timedelta(i)).strftime("%a")] += 1
    # print(weekdays)
    return()


# (11) Cleaning :: Combine several files into one file
def DataCleaning(opt):
    if opt == 'occasional':
        # (11-1) Combine several files into one file for OccasionalCustomer :: 잘됨
        files = glob.glob('./dn_files/190109*일반*180101~181231*/월별파일/*.xlsx')
        df = pd.concat(pd.read_excel(file, sheet_name=0, skiprows=1, header=0,
                       usecols='A:O', index_col='주차권') for file in files)
        colnames = df.columns.values.tolist()
        colnames = [col.replace(' ', '') for col in colnames]
        df.columns = colnames
        # df.to_excel('2018년 일반차량 입출차기록.xlsx', sheet_name='일반권 차량')
        df.to_csv('2018년 일반차량 입출차기록.csv', encoding='cp949')
        print('> OK! Finished!')
    elif opt == 'regular':
        # (11-2) Combine several files into one file for RegularCustomer :: 잘됨
        files = glob.glob('./dn_files/190218*정기*180101~181231*/월별파일/*.xlsx')
        df = pd.concat(pd.read_excel(file, sheet_name=0, skiprows=5, header=0,
                       usecols='A:L', index_col='정기권번호') for file in files)
        colnames = df.columns.values.tolist()
        colnames = [col.replace(' ', '') for col in colnames]
        df.columns = colnames
        df.rename(columns={'Unnamed:8': '이용시작일', 'Unnamed:10': '이용종료일'},
                  inplace=True)
        df.drop(columns=['사용기간'], inplace=True)
        # df.to_excel('2018년 정기차량 입출차기록.xlsx', sheet_name='정기권 차량')
        df.to_csv('2018년 정기차량 입출차기록.csv', encoding='cp949')
        print('> OK! Finished!')
    else:
        print('> !!!ERROR: Check Your Option!!!')
        return()


# (12) Preprocessing :: 일반주차차량 데이터 전처리 :: (잘됨)
def Preprocessing_occ(df):
    df['입차일시'] = pd.to_datetime(
        df['입차일시'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
    df['출차일시'] = pd.to_datetime(
        df['출차일시'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
    df['체류시간'] = df['출차일시'] - df['입차일시']

    # (차량번호 암호화) :: 잘됨
    # df['차량번호'] = df['차량번호'].str.encode('utf-8')
    # df['차량번호'] = [hashlib.md5(val).hexdigest() for val in df['차량번호']]

    # (요일표기_ver1)
    # wdays = {0:'Mon', 1:'Tue', 2:'Wed', 3:'Thu', 4:'Fri', 5:'Sat', 6:'Sun'}
    # df['입차요일'] = df['입차일시'].dt.weekday.map(wdays.get)
    # df['출차요일'] = df['출차일시'].dt.weekday.map(wdays.get)

    # (요일표기_ver2)
    # weekdays = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    # df['입차요일'] = pd.Categorical(df['입차일시'].dt.weekday_name,
    #                                 categories=weekdays, ordered=True)
    # df['출차요일'] = pd.Categorical(df['출차일시'].dt.weekday_name,
    #                                 categories=weekdays, ordered=True)

    # print(f'> Dataframe\n{df.head(10)}\n')
    # print(df.groupby('출차장비').describe().T)
    return(df)


# (13) Preprocessing :: 정기주차차량 데이터 전처리 :: (잘됨)
def Preprocessing_reg(df):
    df['입차일시'] = pd.to_datetime(
        df['입차일시'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
    df['출차일시'] = pd.to_datetime(
        df['출차일시'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
    # print(f'> 입차일시 Nan 개수: {df["입차일시"].isnull().sum()}')  # Nan 값의 개수
    # print(f'> 출차장비 Nan 개수: {df["출차장비"].isnull().sum()}')  # Nan 값의 개수

    df['입차일시'].fillna((df['출차일시']-datetime.timedelta(hours=9)), inplace=True)
    # print(f'> 입차일시 Nan 개수: {df["입차일시"].isnull().sum()}')  # Nan 값의 개수
    # print(f'> 출차장비 Nan 개수: {df["출차장비"].isnull().sum()}')  # Nan 값의 개수

    df['입차장비'].fillna('불명', inplace=True)
    df['출차장비'].fillna('불명', inplace=True)
    # print(f'> 입차장비 Nan 개수: {df["입차장비"].isnull().sum()}')  # Nan 값의 개수
    # print(f'> 출차장비 Nan 개수: {df["출차장비"].isnull().sum()}')  # Nan 값의 개수

    df['체류시간'] = df['출차일시'] - df['입차일시']
    # print(f'> 체류시간 Nan 개수: {df["체류시간"].isnull().sum()}')  # Nan 값의 개수

    # (차량번호 암호화) :: 잘됨
    # df['차량번호'] = df['차량번호'].str.encode('utf-8')
    # df['차량번호'] = [hashlib.md5(val).hexdigest() for val in df['차량번호']]

    # (요일표기_ver1)
    # wdays = {0:'Mon', 1:'Tue', 2:'Wed', 3:'Thu', 4:'Fri', 5:'Sat', 6:'Sun'}
    # df['입차요일'] = df['입차일시'].dt.weekday.map(wdays.get)
    # df['출차요일'] = df['출차일시'].dt.weekday.map(wdays.get)

    # (요일표기_ver2)
    # weekdays = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    # df['입차요일'] = pd.Categorical(df['입차일시'].dt.weekday_name,
    #                                 categories=weekdays, ordered=True)
    # df['출차요일'] = pd.Categorical(df['출차일시'].dt.weekday_name,
    #                                 categories=weekdays, ordered=True)

    # print(f'> Dataframe\n{df.head(10)}\n')
    # print(df.groupby('출차장비').describe().T)
    return(df)


# (21) Analysis :: 주차장별 체류시간 분석 :: (잘됨)
def func_Duration_occ(df):
    df.set_index(['체류시간'], inplace=True)
    # print('> Target Dataframe\n{}\n'.format(df.head()))

    # (바로 아래는 잘됨)
    df_agg = pd.DataFrame()
    cats_names = ['cat1', 'cat2', 'cat3', 'cat4', 'cat5', 'cat6', 'cat7',
                  'cat8', 'cat9', 'cat10', 'cat11', 'cat12', 'cat13', 'cat14',
                  'cat15']
    cats = {'cat1': ['-1:00:00', '00:00:00', '<0min', 1],
            'cat2': ['00:00:00', '00:30:00', '0~30min', 2],
            'cat3': ['00:30:00', '01:00:00', '30min~1hr', 3],
            'cat4': ['01:00:00', '02:00:00', '1hr~2hr', 4],
            'cat5': ['02:00:00', '03:00:00', '2hr~3hr', 5],
            'cat6': ['03:00:00', '04:00:00', '3hr~4hr', 6],
            'cat7': ['04:00:00', '05:00:00', '4hr~5hr', 7],
            'cat8': ['05:00:00', '06:00:00', '5hr~6hr', 8],
            'cat9': ['06:00:00', '09:00:00', '6hr~9hr', 9],
            'cat10': ['09:00:00', '12:00:00', '9hr~12hr', 10],
            'cat11': ['12:00:00', '18:00:00', '12hr~18hr', 11],
            'cat12': ['18:00:00', '24:00:00', '18hr~24hr', 12],
            'cat13': ['24:00:00', '2 days', '1day~2day', 13],
            'cat14': ['2 days', '7 days', '2day~7day', 14],
            'cat15': ['7 days', '360 days', '>=7day', 15]}

    for name in cats_names:
        df_tmp = df[(df.index >= cats[name][0]) & (df.index < cats[name][1])]
        # df_tmp = df[(df.index >= cats[name][0]) & (df.index < cats[name][1]) &
        #             (df['요금종별'] != '회차차량')]
        df_tmp = df_tmp.groupby(['출차장비'])['주차요금', '할인금액', '수익금', '현금수입', '카드수입'].agg(['count', 'sum'])
        df_tmp['timezone'] = cats[name][2]
        df_tmp['tz_order'] = cats[name][3]
        # print('> from: {}\n{}\n'.format(cats[name][2], df_tmp.head()))
        df_agg = df_agg.append(df_tmp)

    df_agg.set_index([df_agg.index, 'tz_order', 'timezone'], inplace=True)
    df_agg = df_agg.unstack(level=0)
    df_agg = df_agg.reorder_levels([2,0,1], axis=1)
    df_agg.columns.names = ['출차장비', '구분1', '구분2']
    df_agg = df_agg.reindex(columns=['A주차장요금계산PC', 'B주차장요금계산PC',
                            '지하요금계산PC', '버스요금계산PC'], level=0)
    df_agg = df_agg.reindex(columns=['주차요금', '할인금액', '수익금',
                            '현금수입', '카드수입'], level=1)
    df_agg.fillna(value=0, inplace=True)
    # print('> Dataframe Aggregated\n{}\n'.format(df_agg.head(50)))
    # df_agg.to_csv('df_Duration(occ).csv', encoding='cp949')
    return(df_agg)

    # (바로 아래도 잘 되지만 위 것을 쓸 것)
    # df_agg = pd.DataFrame()
    # # slots = pd.to_timedelta(['-1:00:00', '00:00:00', '00:30:00', '01:00:00',
    # #                          '02:00:00', '03:00:00', '04:00:00', '05:00:00',
    # #                          '300days'])
    # slots = pd.to_timedelta(['00:00:00','00:30:00','01:00:00','02:00:00'])
    # colnames = ['<0min', '0~30min', '30min~1hr', '1hr~2hr', '2hr~3hr',
    #             '4hr~5hr', '>=5hr', '>=5day']

    # for st, en in zip(slots[0:-1], slots[1:]):
    #     df_tmp = df.loc[(df.index >= st) & (df.index < en)].copy()
    #     # df_tmp = df[(df.index >= st) & (df.index < en) & (df['요금종별'] != '회차차량')].copy()
    #     df_tmp['st'] = st / np.timedelta64(1, 'h')  # 체류시간 시점
    #     df_tmp['en'] = en / np.timedelta64(1, 'h')  # 체류시간 종점
    #     df_tmp = df_tmp.groupby(['출차장비', 'st', 'en'])['주차요금', '할인금액', '수익금', '현금수입', '카드수입'].agg(['count', 'sum'])
    #     # print('> from: {}    > to: {}\n{}\n'.format(st, en, df_tmp.head()))
    #     df_agg = df_agg.append(df_tmp)

    # df_agg.sort_index(inplace=True)
    # df_agg = df_agg.unstack(level=0)
    # df_agg = df_agg.reorder_levels([0,2,1], axis=1)
    # df_agg = df_agg.T
    # df_agg.sort_index(inplace=True)
    # print('> Dataframe Aggregated\n{}\n'.format(df_agg.head(50)))

    # df_agg.to_csv('df_Duration(occ).csv', encoding='cp949')
    # return(df_agg)


# (22) Analysis :: 일반주차 시간대별 체류량 추출 :: (잘됨)
def func_Residue_occ(df, cycle):
    df_in = df.pivot_table(index='입차일시', columns='출차장비', values='차량번호',
                           aggfunc=len, fill_value=0)  # np.size,'count',len 같음
    df_out = df.pivot_table(index='출차일시', columns='출차장비', values='차량번호',
                            aggfunc=len, fill_value=0)
    df_tmp = pd.merge(df_in, df_out, how='outer', left_index=True,
                      right_index=True, suffixes=('|입차', '|출차'))
    df_tmp = df_tmp.resample(cycle).sum()
    # print('> 시간대별 입출차대수\n{}\n'.format(
    #       df_tmp[['지하요금계산PC|입차', '지하요금계산PC|출차']].tail(20)))
    # df_tmp.to_csv('df_residue_TEST01.csv', encoding='cp949')
    # return()

    df_cum = pd.DataFrame()
    df_cum['A체류량'] = df_tmp['A주차장요금계산PC|입차'] - df_tmp['A주차장요금계산PC|출차']
    df_cum['B체류량'] = df_tmp['B주차장요금계산PC|입차'] - df_tmp['B주차장요금계산PC|출차']
    df_cum['버스체류량'] = df_tmp['버스요금계산PC|입차'] - df_tmp['버스요금계산PC|출차']
    df_cum['지하체류량'] = df_tmp['지하요금계산PC|입차'] - df_tmp['지하요금계산PC|출차']
    # df_cum.dropna(inplace=True)
    # print('> 해당시간대 체류대수\n{}\n'.format(df_cum.tail(20)))
    # df_cum.to_csv('df_residue_TEST02.csv', encoding='cp949')
    # return()

    df_cum = df_cum.cumsum()
    df_cum = df_cum[df_cum.index >= datetime.datetime(2018, 1, 1)]
    # print('> 시간대별 누적체류대수\n{}\n'.format(df_cum.tail(20)))
    # df_cum.to_csv('df_residue_TEST03.csv', encoding='cp949')
    return(df_cum)


# (23) Analysis :: 정기주차 시간대별 체류량 추출 :: (잘됨)
def func_Residue_reg(df, cycle):
    df_in = df.pivot_table(index='입차일시', columns='출차장비', values='차량번호',
                           aggfunc=len, fill_value=0)  # np.size,'count',len 같음
    df_out = df.pivot_table(index='출차일시', columns='출차장비', values='차량번호',
                            aggfunc=len, fill_value=0)
    df_tmp = pd.merge(df_in, df_out, how='outer', left_index=True,
                      right_index=True, suffixes=('|입차', '|출차'))
    df_tmp = df_tmp.resample(cycle).sum()
    # print('> 시간대별 입출차대수\n{}\n'.format(
    #       df_tmp[['지하요금계산PC|입차', '지하요금계산PC|출차']].tail(20)))
    # df_tmp.to_csv('df_residue_TEST11.csv', encoding='cp949')
    # return()

    df_cum = pd.DataFrame()
    df_cum['A체류량'] = df_tmp['A주차장요금계산PC|입차'] - df_tmp['A주차장요금계산PC|출차']
    df_cum['B체류량'] = df_tmp['B주차장요금계산PC|입차'] - df_tmp['B주차장요금계산PC|출차']
    df_cum['버스체류량'] = df_tmp['버스요금계산PC|입차'] - df_tmp['버스요금계산PC|출차']
    df_cum['지하체류량'] = df_tmp['지하요금계산PC|입차'] - df_tmp['지하요금계산PC|출차']
    # df_cum.dropna(inplace=True)
    # print('> 해당시간대 체류대수\n{}\n'.format(df_cum.tail(20)))
    # df_cum.to_csv('df_residue_TEST12.csv', encoding='cp949')
    # return()

    df_cum = df_cum.cumsum()
    df_cum = df_cum[df_cum.index >= datetime.datetime(2018, 1, 1)]
    # print('> 시간대별 누적체류대수\n{}\n'.format(df_cum.tail(20)))
    # df_cum.to_csv('df_residue_TEST13.csv', encoding='cp949')
    return(df_cum)


# (24) Analysis :: 정기주차+일반주차 시간대별 체류량 분석 :: (잘됨) :: orig
def func_Residue_cal(df, opt):
    if opt == 'MMS':
        # (Min, Max, Mean 결과값을 보고싶을 때) :: 잘됨
        df_cal = df.groupby([df.index.month, df.index.weekday, df.index.time]).agg(['max', 'mean', 'std', ('+2sigma', lambda x: np.mean(x)+2*np.std(x, ddof=1)), ('+3sigma', lambda x: np.mean(x)+3*np.std(x, ddof=1))])
        # print('> 체류대수(MMS)\n{}\n'.format(df_cal.tail(20)))
        # df_cal.to_csv('df_residue_TEST04.csv', encoding='cp949')

        df_cal.index.names = ['월', '요일', '시간대']
        df_cal.columns.names = ['장소', '구분']
        df_cal = df_cal.stack(level=0)
        df_cal = df_cal.reorder_levels(order=[3,0,1,2], axis=0)
        df_cal = df_cal.reindex(index=['A체류량','B체류량','지하체류량','버스체류량'], level=0)
        df_cal = df_cal.reindex(columns=['max','mean','std','+2sigma','+3sigma'])
        # wdays = {0:'Mon', 1:'Tue', 2:'Wed', 3:'Thu', 4:'Fri', 5:'Sat', 6:'Sun'}
        # df_cal.rename(mapper=wdays, axis='index', level=3, inplace=True)
        # print('> 체류대수(MMS completed)\n{}\n'.format(df_cal.tail(20)))
        # df_cal.to_csv('df_residue_TEST05.csv', encoding='cp949')
        return(df_cal)
    elif opt == 'ohlc':
        # (OHLC 결과값을 보고싶을 때) :: 잘됨
        df_cal = df.groupby([df.index.month, df.index.weekday, df.index.time]).ohlc()
        # print('> 체류대수(ohlc)\n{}\n'.format(df_cal.tail(20)))
        # df_cal.to_csv('df_residue_TEST06.csv', encoding='cp949')

        df_cal.index.names = ['월', '요일', '시간대']
        df_cal.columns.names = ['장소', '구분']
        df_cal = df_cal.stack(level=0)
        df_cal = df_cal.reorder_levels(order=[3,0,1,2], axis=0)
        df_cal = df_cal.reindex(index=['A체류량','B체류량','지하체류량','버스체류량'], level=0)
        df_cal = df_cal.reindex(columns=['open','high','low','close'])
        # wdays = {0:'Mon', 1:'Tue', 2:'Wed', 3:'Thu', 4:'Fri', 5:'Sat', 6:'Sun'}
        # df_cal.rename(mapper=wdays, axis='index', level=3, inplace=True)
        # print('> 체류대수(ohlc completed)\n{}\n'.format(df_cal.tail(20)))
        # df_cal.to_csv('df_residue_TEST07.csv', encoding='cp949')
        return(df_cal)
    else:
        print('> !!!ERROR: Check Your Option!!!')
        return()


# (31) Analysis :: Trivial
def func_Overlap(df_occ, n):
    # (31-1) 주차권 번호가 같은 차량정보 :: (잘됨)
    df_dup = df_occ['주차권'].value_counts()
    print(f'주차권 정보가 {n}회 이상 같은 자료의 수\n{df_dup[df_dup>n]}\n')
    return()


# (51) Plot :: 시간대별 체류시간 플롯 :: 잘됨
def plot_Duration(df):
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    df.plot(kind='line', ax=ax1)
    label_group_bar(ax1, df)
    # df.plot(kind='line', ax=ax1)

    # ax1.xaxis.set_major_locator(mdates.DayLocator())
    # ax1.xaxis.set_major_locator(mdates.DayLocator(bymonthday=5, interval=10))
    # ax1.xaxis.set_major_formatter(mticker.NullFormatter())
    # ax1.xaxis.set_major_formatter(mticker.FixedFormatter('%y\n%m-%d'))
    # ax1.xaxis.set_tick_params(rotation=30, labelsize=10)

    _ = plt.legend(loc='best')
    _ = plt.tight_layout()
    _ = plt.show()
    # _ = fig.savefig('./output.png')


# (52) Plot :: 시간대별 체류규모 플롯(MMM) :: 잘됨
def plot_Residue_MMS_3D(df, opt1, opt2):    # .plot plots the index against every column
    import matplotlib.pyplot as plt
    print('> Dataframe\n{}\n'.format(df.head(10)))

    if opt1 == '월별' and opt2 == 'plt':
        # (52-1) plot :: 월별 > 요일별 플롯
        df = df.unstack(level=0)   # [level=0: 요일별]
        df.columns = df.columns.droplevel(level=0)
        df = df.reset_index()
        print('> Dataframe for plot\n{}\n'.format(df.head(10)))

        fig, (axes) = plt.subplots(nrows=3, ncols=4, figsize=(10,8), sharex=True, sharey=True)
        df.pivot(index='시간대', columns='요일', values=1).plot(ax=axes[0,0], title='1월', lw=2)
        df.pivot(index='시간대', columns='요일', values=2).plot(ax=axes[0,1], title='2월', lw=2)
        df.pivot(index='시간대', columns='요일', values=3).plot(ax=axes[0,2], title='3월', lw=2)
        df.pivot(index='시간대', columns='요일', values=4).plot(ax=axes[0,3], title='4월', lw=2)
        df.pivot(index='시간대', columns='요일', values=5).plot(ax=axes[1,0], title='5월', lw=2)
        df.pivot(index='시간대', columns='요일', values=6).plot(ax=axes[1,1], title='6월', lw=2)
        df.pivot(index='시간대', columns='요일', values=7).plot(ax=axes[1,2], title='7월', lw=2)
        df.pivot(index='시간대', columns='요일', values=8).plot(ax=axes[1,3], title='8월', lw=2)
        df.pivot(index='시간대', columns='요일', values=9).plot(ax=axes[2,0], title='9월', lw=2)
        df.pivot(index='시간대', columns='요일', values=10).plot(ax=axes[2,1], title='10월', lw=2)
        df.pivot(index='시간대', columns='요일', values=11).plot(ax=axes[2,2], title='11월', lw=2)
        df.pivot(index='시간대', columns='요일', values=12).plot(ax=axes[2,3], title='12월', lw=2)

        plt.xticks([datetime.time(0,0,0), datetime.time(2,0,0), datetime.time(4,0,0),
                    datetime.time(6,0,0), datetime.time(8,0,0), datetime.time(10,0,0),
                    datetime.time(12,0,0), datetime.time(14,0,0), datetime.time(16,0,0),
                    datetime.time(18,0,0), datetime.time(20,0,0), datetime.time(22,0,0)])

        # plt.xticks([0,1,2,3,4,5,6], ['월','화','수','목','금','토','일'])  # 당연히 샘플임
        # axes[0,0].set_xticks(df.index.get_level_values(level=1))
        # axes[0,0].set_xticklabels(df.index.get_level_values(level=1))
        # fig.autofmt_xdate(rotation=60)  # x축 글자 각도 조정

        plt.suptitle('2018년도 외곽(지하)주차장 통계')
        plt.legend(loc='upper left')
        plt.tight_layout()
        plt.show()
        # _ = fig.savefig('./output.png')
    elif opt1 == '요일별' and opt2 == 'plt':
        # (52-2) plot :: 요일별 > 월별 플롯
        df = df.unstack(level=1)  # [level=1: 월별]
        df.columns = df.columns.droplevel(level=0)
        df = df.reset_index()
        print('> dataframe for plot\n{}\n'.format(df.head(10)))

        fig, (axes) = plt.subplots(nrows=2, ncols=4, figsize=(10,8), sharex=True, sharey=True)
        df.pivot(index='시간대', columns='월', values=0).plot(ax=axes[0,0], title='월요일', lw=2)
        df.pivot(index='시간대', columns='월', values=1).plot(ax=axes[0,1], title='화요일', lw=2)
        df.pivot(index='시간대', columns='월', values=2).plot(ax=axes[0,2], title='수요일', lw=2)
        df.pivot(index='시간대', columns='월', values=3).plot(ax=axes[0,3], title='목요일', lw=2)
        df.pivot(index='시간대', columns='월', values=4).plot(ax=axes[1,0], title='금요일', lw=2)
        df.pivot(index='시간대', columns='월', values=5).plot(ax=axes[1,1], title='토요일', lw=2)
        df.pivot(index='시간대', columns='월', values=6).plot(ax=axes[1,2], title='일요일', lw=2)

        plt.xticks([datetime.time(0,0,0), datetime.time(2,0,0), datetime.time(4,0,0),
                    datetime.time(6,0,0), datetime.time(8,0,0), datetime.time(10,0,0),
                    datetime.time(12,0,0), datetime.time(14,0,0), datetime.time(16,0,0),
                    datetime.time(18,0,0), datetime.time(20,0,0), datetime.time(22,0,0)])

        plt.suptitle('2018년도 외곽(지하)주차장 통계')
        plt.legend(loc='upper left')
        plt.tight_layout()
        plt.show()
        # _ = fig.savefig('./output.png')
    elif opt2 == 'sns':
        # (52-3) plot using Seaborn :: (잘됨)
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set(style='whitegrid')

        fig, (ax1) = plt.subplots(nrows=1, ncols=1, figsize=(10,8), sharex=True, sharey=True)
        # ax = sns.violinplot(data=df[opt1[1]], palette='Set3')  # 월별 평균값
        ax = sns.boxplot(data=df[opt1[0]], palette='Set3')  # 월별 최댓값
        # ax = sns.boxplot(by=ddd, data=df[opt1[0]], palette='Set3')  # 요일별 평균값
        sns.despine(left=True, bottom=True)

        plt.legend(loc='best')
        plt.tight_layout()
        plt.show()
        # fig.savefig('./output.png')
    else:
        print('> !!!ERROR: Check Your Option!!!')
        return()


# (53) Plot :: 시간대별 체류규모 플롯(MMM) :: 잘됨
def plot_Residue_MMS_2D(df, opt1, opt2):    # .plot plots the index against every column
    import matplotlib.pyplot as plt
    print('> Dataframe\n{}\n'.format(df.head(10)))

    if opt1 == '월별' and opt2 == 'plt':
        # (53-1) plot :: 월별 > 요일별 플롯
        print('> Dataframe for plot\n{}\n'.format(df.head(20)))

        fig, (axes) = plt.subplots(nrows=3, ncols=4, figsize=(10,8), sharex=True, sharey=True)
        ## (여기부터 수정)
        df.loc[pd.IndexSlice[1,:],:].plot(ax=axes[0,0], title='1월', lw=2)
        df.loc[pd.IndexSlice[2,:],:].plot(ax=axes[0,1], title='2월', lw=2)
        df.loc[pd.IndexSlice[3,:],:].plot(ax=axes[0,2], title='3월', lw=2)
        df.loc[pd.IndexSlice[4,:],:].plot(ax=axes[0,3], title='4월', lw=2)
        df.loc[pd.IndexSlice[5,:],:].plot(ax=axes[1,0], title='5월', lw=2)
        df.loc[pd.IndexSlice[6,:],:].plot(ax=axes[1,1], title='6월', lw=2)
        df.loc[pd.IndexSlice[7,:],:].plot(ax=axes[1,2], title='7월', lw=2)
        df.loc[pd.IndexSlice[8,:],:].plot(ax=axes[1,3], title='8월', lw=2)
        df.loc[pd.IndexSlice[9,:],:].plot(ax=axes[2,0], title='9월', lw=2)
        df.loc[pd.IndexSlice[10,:],:].plot(ax=axes[2,1], title='10월', lw=2)
        df.loc[pd.IndexSlice[11,:],:].plot(ax=axes[2,2], title='11월', lw=2)
        df.loc[pd.IndexSlice[12,:],:].plot(ax=axes[2,3], title='12월', lw=2)

        plt.xticks([0,1,2,3,4,5,6], ['월','화','수','목','금','토','일'])
        # fig.autofmt_xdate(rotation=60)  # x축 글자 각도 조정

        _ = plt.legend(loc='upper left')
        _ = plt.tight_layout()
        _ = plt.show()
        # _ = fig.savefig('./output.png')
    elif opt1 == '요일별' and opt2 == 'plt':
        # (53-2) plot :: 요일별 > 월별
        print('> dataframe for plot\n{}\n'.format(df.head(20)))

        fig, (axes) = plt.subplots(nrows=2, ncols=4, figsize=(10,8), sharex=True, sharey=True)
        df.loc[pd.IndexSlice[:,0],:].plot(ax=axes[0,0], title='월요일', lw=2)
        df.loc[pd.IndexSlice[:,1],:].plot(ax=axes[0,1], title='화요일', lw=2)
        df.loc[pd.IndexSlice[:,2],:].plot(ax=axes[0,2], title='수요일', lw=2)
        df.loc[pd.IndexSlice[:,3],:].plot(ax=axes[0,3], title='목요일', lw=2)
        df.loc[pd.IndexSlice[:,4],:].plot(ax=axes[1,0], title='금요일', lw=2)
        df.loc[pd.IndexSlice[:,5],:].plot(ax=axes[1,1], title='토요일', lw=2)
        df.loc[pd.IndexSlice[:,6],:].plot(ax=axes[1,2], title='일요일', lw=2)

        plt.xticks([1,2,3,4,5,6,7,8,9,10,11,12], ['1월','2월','3월','4월','5월','6월','7월','8월','9월','10월','11월','12월'])
        # fig.autofmt_xdate(rotation=60)  # x축 글자 각도 조정

        _ = plt.legend(loc='upper left')
        _ = plt.tight_layout()
        _ = plt.show()
        # _ = fig.savefig('./output.png')
    elif opt == 'sns':
        # (53-3) plot using Seaborn :: (잘됨)
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set(style='whitegrid')

        fig, (ax1) = plt.subplots(nrows=1, ncols=1, figsize=(10, 8), sharex=True, sharey=True)
        # ax = sns.violinplot(data=df[opt1[1]], palette='Set3')  # 월별 평균값
        # ax = sns.boxplot(data=df[opt1[0]], palette='Set3')  # 월별 최댓값
        # ax = sns.boxplot(by=ddd, data=df[opt1[0]], palette='Set3')  # 요일별 평균값
        sns.despine(left=True, bottom=True)

        _ = plt.legend(loc='best')
        _ = plt.tight_layout()
        _ = plt.show()
        # _ = fig.savefig('./output.png')
    else:
        print('> !!!ERROR: Check Your Option!!!')
        return()


# (54) Plot :: 시간대별 체류규모 플롯(OHLC) :: 안됨
def plot_Residue_ohlc(df):
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from mpl_finance import candlestick_ohlc

    df = df.droplevel(level=0, axis=0)
    df = df.droplevel(level=0, axis=1)
    df.reset_index(inplace=True)
    print(df.dtypes)
    print(df)
    df['시간대(시작값)'] = df['시간대(시작값)'].apply(mdates.date2num)
    print(df)
    return()

    fig, (ax1) = plt.subplots(nrows=1, ncols=1, figsize=(10,8), sharex=False)
    # candlestick_ohlc(ax=ax1, quotes=zip(mdates.date2num(df.index.to_pydatetime()),df['open'],df['high'],df['low'],df['close']))
    candlestick_ohlc(ax=ax1, quotes=df.values)
    _ = plt.legend(loc='best')
    _ = plt.tight_layout()
    _ = plt.show()
    # _ = fig.savefig('./output.png')

    # import plotly.plotly as py
    # trace =go.Ohlc(x=df.index.get_level_values(1),
    #                open=df['open'],
    #                high=df['high'],
    #                low=df['low'],
    #                close=df['close'])
    # data = [trace]
    # py.iplot(data, filename='simple_ohlc')


# (Z) Run Code
if __name__ == '__main__':
    # print("helo~")
    main()
