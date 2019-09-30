# -*- coding: utf-8 -*-
'''
    (설명부분)
'''
# (A) Import Modules
import glob, json, requests, time, pprint
import pandas as pd
from bs4 import BeautifulSoup
# import lxml
import ath_Geocoding as ath

# (Pandas Display options)
pd.set_option('display.max_columns', 20)
# pd.set_option('max_colwidth', 18)
# pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.width', 100)
pd.set_option('display.unicode.east_asian_width', True)


# (B) Define Functions
# (1) Main function
def main():
    # (1-1) 지오코딩 :: (잘됨)
    files = glob.glob('./dn_files/01_한글주소*.xlsx')
    file = files[-1]
    print('> {}'.format(file))
    # df = pd.read_excel(file, sheet_name=0, header=0, usecols='A:D', nrows=10, encoding='cp949')
    df = pd.read_excel(file, sheet_name=0, header=0, usecols='A:D', encoding='cp949')
    print('> {}\n'.format(df.head()))

    df_addrs = Munging_addr(df)

    lst_coords = []
    for index, row in df_addrs.iterrows():
        if (row[2] == '(불명)'):
            coord = ConvertToGcode(index, row, row[3], 'ROAD')
            print('> {}\n'.format(coord))
            lst_coords.append(coord)
        else:
            coord = ConvertToGcode(index, row, row[2], 'PARCEL')
            print('> {}\n'.format(coord))
            lst_coords.append(coord)

    df_coords = pd.DataFrame(lst_coords, columns=['구분', '업소명', '주소타입', '주소', 'lon', 'lat'])
    print('> {}\n'.format(df_coords.head(20)))
    # df_coords.to_csv('./out_Coords_cp949.csv', encoding='cp949')  # (Encoding Err)
    df_coords.to_csv('./out_Coords_utf8.csv', encoding='utf-8')
    return()

    # (1-2) 리버스지오코딩 :: (잘됨)
    files = glob.glob('./dn_files/02_지오코드*.xlsx')
    file = files[-1]
    print('> {}'.format(file))
    df = pd.read_excel(file, sheet_name=0, header=0, usecols='A:F', nrows=10, encoding='cp949')
    # df = pd.read_excel(file, sheet_name=0, header=0, usecols='A:F', encoding='cp949')
    print('> {}\n'.format(df.head()))

    df_coords = Munging_code(df)
    df_coords = df_coords.astype(str)

    lst_addrs = []
    for index, row in df_coords.iterrows():
        addr = ConvertToRevGcode(index, row)
        lst_addrs.append(addr)

    df_addrs = pd.DataFrame(lst_addrs, columns=['구분', '업소명', 'lon', 'lat', '주소'])
    print('> {}\n'.format(df_addrs.head(20)))
    # df_addrs.to_csv('./out_Addrs_cp949.csv', encoding='cp949')
    # df_addrs.to_csv('./out_Addrs_utf8.csv', encoding='utf-8')
    return()


# (2) Data Munging
def Munging_addr(df):
    # df['소재지'] = df['소재지'].str.split(pat=',').str[0]  # 잘됨
    # df['소재지'] = df.iloc[:,2].str.split(pat=',').str[0]  # 잘됨
    return(df)

# (3) Data Munging
def Munging_code(df):
    # df['소재지'] = df['소재지'].str.split(pat=',').str[0]  # 잘됨
    # df['소재지'] = df.iloc[:,2].str.split(pat=',').str[0]  # 잘됨
    return(df)


# (4) Geocoding :: (잘됨)
def ConvertToGcode(index, row, addr, addr_type):
    # (4-1) Set URL
    base = 'http://api.vworld.kr/req/address?service=address'
    opt1 = '&request=getCoord'
    opt2 = '&key=' + ath.api_key  # HJL's Only
    opt3 = '&format=json'         # [json, xml]
    opt4 = '&type=' + addr_type        # [PARCEL(지번주소), ROAD(도로명주소)]
    opt5 = '&address=' + addr
    # option6 = '&crs=' +         # [EPSG:4326(WGS84,기본값), EPSG:5179(UTM-K)]
    url = base + opt1 + opt2 + opt3 + opt4 + opt5
    print(f'> [{index}]  url: {url}')

    # # (4-2) Get XML result :: (잘됨)
    # try:
    #     RESP = requests.get(url)
    #     # print('> Address: ' + addr)
    #     # print('> Status Code: ' + str(RESP.status_code))
    #     resp = RESP.text.strip()
    # except Exception as err:
    #     print('> !!!!!!  ERROR  !!!!!!')
    #     print('> ERROR!!: ' + err)
    #     print('> Status Code: ' + str(RESP.status_code))

    # # (Parse XML)
    # soup = BeautifulSoup(resp, 'lxml-xml')    # lxml-XML parser
    # isOK = soup.find('status').text.strip()
    # # print(soup.prettify())

    # # (Extract data from XML)
    # if (isOK == 'OK'):
    #     lon = soup.find('x').text.strip()    # 경도
    #     lat = soup.find('y').text.strip()    # 위도
    #     print('> 구분: {}, 업소명: {}, 주소타입: {}, 주소: {}, 경도: {}, 위도: {}/n'.format(
    #           row[0], row[1], addr_type, addr, lon, lat))
    #     return([row[0], row[1], addr_type, addr, lon, lat])
    # elif (isOK == 'NOT_FOUND'):
    #     print('> === ERROR: Wrong Address!! ===')
    #     return([row[0], row[1], addr_type, addr, 0, 0])
    # else:
    #     print('> === ERROR: Unknown!! ===')
    #     return([row[0], row[1], addr_type, addr, -1, -1])

    # time.sleep(50)

    # (4-3) Get JSON result :: (잘됨)
    try:
        RESP = requests.get(url)
        # print('> Address: ' + addr)
        # print('> Status Code: ' + str(RESP.status_code))
        resp = RESP.text.strip()
        resp = json.loads(resp)
    except Exception as err:
        print('> !!!!!!  ERROR  !!!!!!')
        print('> ERROR!!: ' + err)
        print('> Status Code: ' + str(RESP.status_code))

    # (Parse json)
    # pprint.pprint(resp)
    isOK = resp['response']['status']

    # (Extract data from json)
    if (isOK == 'OK'):
        lon = resp['response']['result']['point']['x']
        lat = resp['response']['result']['point']['y']
        print('> 구분: {}, 업소명: {}, 주소타입: {}, 주소: {}, 경도: {}, 위도: {}'.format(
              row[0], row[1], addr_type, addr, lon, lat))
        return([row[0], row[1], addr_type, addr, lon, lat])
    elif (isOK == 'NOT_FOUND'):
        print('> === ERROR: Wrong Address!! ===')
        return([row[0], row[1], addr_type, addr, 0, 0])
    else:
        print('> === ERROR: Unknown!! ===')
        return([row[0], row[1], addr_type, addr, -1, -1])

    time.sleep(50)


# (5) Reverse GeoCoding :: (잘됨)
def ConvertToRevGcode(index, row):
    # (5-1) Set URL
    base = 'http://api.vworld.kr/req/address?service=address'
    opt1 = '&request=getAddress'
    opt2 = '&key=' + ath.api_key  # HJL's Only
    opt3 = '&format=json'         # [json, xml]
    opt4 = '&type=PARCEL'         # [PARCEL(지번주소), ROAD(도로명주소), BOTH]
    opt5 = '&point=' + row['lon'] + ',' + row['lat']   # x(경도), y(위도)
    # opt6 = '&crs=' +            # [EPSG:4326(WGS84,기본값), EPSG:5179(UTM-K)]
    url = base + opt1 + opt2 + opt3 + opt4 + opt5
    # print(f'> [{index}]  url: {url}')

    # # (5-2) Get XML result :: (잘됨)
    # try:
    #     RESP = requests.get(url)
    #     # print('> Status Code: ' + str(RESP.status_code))
    #     resp = RESP.text.strip()
    # except Exception as err:
    #     print('> !!!!!!  ERROR  !!!!!!')
    #     print('> ERROR!!: ' + err)
    #     print('> Status Code: ' + str(resp.status_code))

    # # (Parse XML)
    # soup = BeautifulSoup(resp, 'lxml-xml')    # lxml-XML parser
    # isOK = soup.find('status').text.strip()
    # # print(soup.prettify())

    # # (Extract data from xml)
    # if (isOK == 'OK'):
    #     addr = soup.find('text').text.strip()    # 전체주소
    #     # print(f'> 주소: {addr} ')
    #     # print('> 구분: {}, 업소명: {}, 경도: {}, 위도: {}, 주소: {}'.format(
    #     #       row[0], row[1], row['lon'], row['lat'], addr))
    #     return([row[0], row[1], row['lon'], row['lat'], addr])
    # elif (isOK == 'NOT_FOUND'):
    #     print('> === ERROR: Wrong Address!! ===')
    #     return([row[0], row[1], row['lon'], row['lat'], 0])
    # else:
    #     print('> === ERROR: Unknown!! ===')
    #     return([row[0], row[1], row['lon'], row['lat'], -1])

    # time.sleep(50)

    # (5-3) Get JSON result  :: (잘됨)
    try:
        RESP = requests.get(url)
        # print('> Address: ' + addr)
        # print('> Status Code: ' + str(RESP.status_code))
        resp = RESP.text.strip()
        resp = json.loads(resp)
    except Exception as err:
        print('> !!!!!!  ERROR  !!!!!!')
        print('> ERROR!!: ' + err)
        print('> Status Code: ' + str(RESP.status_code))

    # (Parse json)
    # pprint.pprint(resp)
    isOK = resp['response']['status']

    # (Extract data from json)
    if (isOK == 'OK'):
        addr = resp['response']['result'][0]['text']
        # print(f'> 주소: {addr}')
        # print('> 구분: {}, 업소명: {}, 경도: {}, 위도: {}, 주소: {}'.format(
        #       row[0], row[1], row['lon'], row['lat'], addr))
        return([row[0], row[1], row['lon'], row['lat'], addr])
    elif (isOK == 'NOT_FOUND'):
        print('> === ERROR: Wrong Address!! ===')
        return([row[0], row[1], row['lon'], row['lat'], 0])
    else:
        print('> === ERROR: Unknown!! ===')
        return([row[0], row[1], row['lon'], row['lat'], -1])

    time.sleep(50)


# (Z) Run Code
if __name__ == '__main__':
    main()
