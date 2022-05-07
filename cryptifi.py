import math
from datetime import datetime
from pycoingecko import CoinGeckoAPI
import pandas as pd
from math import floor, ceil
import sqlite3
import cryptifi as st
from binance.client import Client
import ffn
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
#import ccxt
import talib as ta
#import ccxt.async_support as ccxt



API_KEY = 'ggEM79Vvjd6QeHn32mgA4MkKXKrU3vN3mVNSn66pHPgAdEmtF2qdApyqmpZnScpJ'
API_SECRET = 'ZxK3NSAO9MAa0ExWVlqqMtsiMiKQ4WecxCFXIo93X1zXlBR6scKcV4Mc1l2PxL7Y'

client = Client(API_KEY, API_SECRET)

connection = sqlite3.connect('crypto.db')
cursor = connection.cursor()

def ohlcCryphist(ticker,interval,start):
    column = ['date', 'open', 'high', 'low', 'close','volume', 'close_time', 'quote_vol', 'trades', 'buy_base', 'buy_quote', 'ignore']
    data = pd.DataFrame(client.get_historical_klines(ticker, interval,start), columns=column)
    data[['open', 'high', 'low', 'close','volume']] = data[['open', 'high', 'low', 'close','volume']].apply(pd.to_numeric, axis=1)
    data['date'] = data['date']/1000
    data['datetime'] = pd.to_datetime(data['date'], unit='s')
    data = data.set_index('datetime')
    return data



def annual_rets(allReturns):
  allReturns = allReturns.copy()
  df = allReturns.resample('Y').agg({'close': lambda df: df[-1]})
  #print(df)
  df = df.pct_change()
  return df.dropna()

def quat_rets(allReturns):
  allReturns = allReturns.copy()
  df = allReturns.resample('Q').agg({'close': lambda df: df[-1]})
  df = df.pct_change()
  return df.dropna()

def monthly_rets(allReturns):
  allReturns = allReturns.copy()
  df = allReturns.resample('M').agg({'close': lambda df: df[-1]})
  df = df.pct_change()
  return df.dropna()



prices = pd.DataFrame(client.get_all_tickers())
prices = prices[prices['symbol'].str.contains("USDT")]
tickers = list(prices['symbol'])
ticks = tickers[:100]
prices.head()
symbols = prices['symbol']


st.set_page_config(layout="wide")
cg = CoinGeckoAPI()

option = st.sidebar.selectbox("Navgiate", ('Home', 'Coins','Scanner'))

if option =="Home":
    st.title('Top Cryptocurrencies')
    def top_coins_by_market_cap(cg,  currency='usd', limit=500):
        try:
            top_len = limit
            per_page = 250 if top_len >= 250 else top_len
            pages = ceil(top_len/per_page)
            top_coins = []
            for i in range(pages):
                page_num = i+1
                res = cg.get_coins_markets(
                    order='market_cap_desc',
                    vs_currency=currency,
                    per_page=per_page,
                    page=page_num
                )
                top_coins.extend(res)
            return top_coins
        except Exception as e:
            print('top_coins_by_market_cap', 'Error in CoinGeckoAPI '+str(e))
            return []

    data = pd.DataFrame(top_coins_by_market_cap(cg))
    data = data[data['market_cap']>100000000]
    data.drop(columns=['roi'],inplace=True)
    data.set_index('name',inplace=True)
    img = list(data['image'])
    imgs = []

    # for i in img:
    #     imgs.append(st.image(i,width=20))
    # data['image'] = imgs

    col1, col2 = st.columns(2)
    #col1.dataframe(data)
    data['price_change_percentage_24h'] = data['price_change_percentage_24h']
    #data.dropna(inplace=True)
    col1.subheader('Top Gainers 24H')
    top = data.sort_values(by=['price_change_percentage_24h'],ascending=False).head(10)
    #print(top.columns)
    col1.dataframe(top[['symbol','current_price','price_change_percentage_24h',]])

    col2.subheader('Top Losers 24H')
    bot = data.sort_values(by=['price_change_percentage_24h']).head(10)
    #print(bot.columns)
    col2.dataframe(bot[['current_price','price_change_percentage_24h',]])

    data = data.sort_values(by='market_cap_rank')


    # fig = plt.figure(figsize=(10, 4))
    # my_values=list(data['market_cap'][:100])

    # cmap = matplotlib.cm.Blues  
    # mini=min(my_values)
    # maxi=max(my_values)
    # norm = matplotlib.colors.Normalize(vmin=mini, vmax=maxi)
    # colors = [cmap(norm(value)) for value in my_values]
    # squarify.plot(sizes=data['market_cap'], label=data['symbol'], alpha=0.8 )
    # st.title("Monthly Returns (%)")
    # st.pyplot(fig, use_container_width=True)

    st.dataframe(data)

    fig = px.scatter(data, x="market_cap", y="price_change_percentage_24h",
            size="market_cap",color='price_change_percentage_24h',
                hover_name="symbol", log_x=True, size_max=60)
    #fig.update_xaxes(type='category')
    fig.update_layout(height=700,template="plotly_white")

    st.plotly_chart(fig, use_container_width=True)




    ids = data['id']
    st.sidebar.title('Options')
# option = st.sidebar.selectbox("Select",ids)

# print(option)
#for id in ids[:1]:
#if option =='Bitcoin':
if option == "Coins":
    symbol = st.sidebar.text_input("Symbol", value='bitcoin', max_chars=None, key=None, type='default')
    coin = symbol
    # st.sidebar.text_input("Symbol",value='bitcoin')
    # st.sidebar.selectbox('Select the Crypto', symbols )

    def access_index(obj,idx):
        try:
            return obj[idx] if idx in obj else None
        except:
            return None

    def get_crypto_category(categories):
        try:
            return categories[-1] if categories[-1] else categories[-2]
        except Exception as e:
            return None

    currency = 'usd'


    id = coin
    data = (cg.get_coin_by_id(id))
    data = {**data, **data['market_data']}
    data.pop('market_data',None)
    entry = {
            'code': id,
            'symbol': access_index(data, 'symbol'),
            'name': access_index(data, 'name'),
            'category': get_crypto_category(access_index(data, 'categories')),
            # Not required
            #'description': access_index(access_index(data, 'description'), 'en'),
            # Not required
            #'image': access_index(access_index(data, 'image'), 'thumb'),
            #'homepage': None,  # Not required
            'genesis_date': access_index(data, 'genesis_date'),
            'sentiment_votes_up_percentage': access_index(data, 'sentiment_votes_up_percentage'),
            'sentiment_votes_down_percentage': access_index(data, 'sentiment_votes_down_percentage'),
            'market_cap_rank':	access_index(data, 'market_cap_rank'),
            # 'coingecko_rank':	access_index(data,'coingecko_rank'),  # Not required
            # 'coingecko_score': access_index(data,'coingecko_score'),  # Not required
            # 'developer_score': access_index(data,'developer_score'),  # Not required
            # 'community_score':	access_index(data,'community_score'),  # Not required
            # 'liquidity_score':	access_index(data,'liquidity_score'),  # Not required
            # 'public_interest_score': access_index(data,'public_interest_score'),  # Not required
            'current_price': access_index(access_index(data, 'current_price'), currency),
            'ath': 	access_index(access_index(data, 'ath'), currency),
            'ath_change_percentage': access_index(access_index(data, 'ath_change_percentage'), currency),
            'ath_date': access_index(access_index(data, 'ath_date'), currency),
            'atl': 	access_index(access_index(data, 'atl'), currency),
            'atl_change_percentage': 	access_index(access_index(data, 'atl_change_percentage'), currency),
            'atl_date': 	access_index(access_index(data, 'atl_date'), currency),
            'market_cap': access_index(access_index(data, 'market_cap'), currency),
            'market_cap_change_24h': access_index(data, 'market_cap_change_24h'),
            'market_cap_change_percentage_24h':	access_index(data, 'market_cap_change_percentage_24h'),
            'market_cap_change_24h_in_currency': 	access_index(access_index(data, 'market_cap_change_24h_in_currency'), currency),
            'market_cap_change_percentage_24h_in_currency': access_index(access_index(data, 'market_cap_change_percentage_24h_in_currency'), currency),
            'fully_diluted_valuation': 	access_index(access_index(data, 'fully_diluted_valuation'), currency),
            'total_volume': access_index(access_index(data, 'total_volume'), currency),
            'high_24h': access_index(access_index(data, 'high_24h'), currency),
            'low_24h': 	access_index(access_index(data, 'low_24h'), currency),
            'price_change_24h': access_index(data, 'price_change_24h')
            }
    entry_2  = {  'price_change_percentage_24h': 	access_index(data, 'price_change_percentage_24h'),
            'price_change_percentage_7d': access_index(data, 'price_change_percentage_7d'),
            'price_change_percentage_14d': access_index(data, 'price_change_percentage_14d'),
            'price_change_percentage_30d': access_index(data, 'price_change_percentage_30d'),
            'price_change_percentage_60d':	access_index(data, 'price_change_percentage_60d'),
            'price_change_percentage_200d': access_index(data, 'price_change_percentage_200d'),
            'price_change_percentage_1y': access_index(data, 'price_change_percentage_1y'),
            'price_change_24h_in_currency': access_index(access_index(data, 'price_change_24h_in_currency'), currency),
            'price_change_percentage_1h_in_currency':	access_index(access_index(data, 'price_change_percentage_1h_in_currency'), currency),
            'price_change_percentage_24h_in_currency': access_index(access_index(data, 'price_change_percentage_24h_in_currency'), currency),
            'price_change_percentage_7d_in_currency':	access_index(access_index(data, 'price_change_percentage_7d_in_currency'), currency),
            'price_change_percentage_14d_in_currency': access_index(access_index(data, 'price_change_percentage_14d_in_currency'), currency),
            'price_change_percentage_30d_in_currency': access_index(access_index(data, 'price_change_percentage_30d_in_currency'), currency),
            'price_change_percentage_60d_in_currency': access_index(access_index(data, 'price_change_percentage_60d_in_currency'), currency),
            'price_change_percentage_200d_in_currency':	access_index(access_index(data, 'price_change_percentage_200d_in_currency'), currency),
            'price_change_percentage_1y_in_currency':	access_index(access_index(data, 'price_change_percentage_1y_in_currency'), currency),
            'total_supply': access_index(data, 'total_supply'),
            'max_supply':	access_index(data, 'max_supply'),
            'circulating_supply': access_index(data, 'circulating_supply'),
            #'last_updated': access_index(data, 'last_updated'),
        }
    def get_crypto_perf_metrics(data, riskfreerate):
        details = (data.calc_stats())
        details.set_riskfree_rate(riskfreerate)
        metrics = ["start", "end", "total_return", 'best_year', 'worst_year', 'yearly_sharpe', 'yearly_sortino', 'cagr', 'max_drawdown',
                'mtd', 'three_month', 'six_month', 'ytd', 'one_year', 'three_year', 'five_year',  'ten_year', 'incep']
        #companies = list(data.columns)
        result = pd.DataFrame(metrics)
        result.columns = ["Description"]
        
        temp = {}
        for Symbol in metrics:
            to_do = "details."+Symbol
            # print(to_do)
            temp[Symbol] = eval(to_do)
        result = pd.DataFrame.from_dict(temp,orient='index')
        #result["Description"].map(temp)
        #print(result)
        #result = result.set_index("Description")
        percentages = ["total_return", 'best_year', 'worst_year', 'cagr', 'max_drawdown', 'mtd', 'three_month',
                    'six_month', 'ytd', 'one_year', 'three_year', 'five_year']
        result.loc[percentages] = result.loc[percentages] * 100
        ratios = ["total_return", 'best_year', 'worst_year', 'yearly_sharpe',
                'yearly_sortino', 'cagr', 'max_drawdown']
        returns = ['mtd', 'three_month', 'six_month', 'ytd',
                'one_year', 'three_year', 'five_year']

        asset_ratios = result.loc[ratios]
        asset_returns = result.loc[returns]
        asset_ratios = asset_ratios.reset_index()
        asset_returns = asset_returns.reset_index()
        asset_ratios["Description"] = ["Total Returns", "Best Year", "Worst Year",
                                    'Sharpe Ratio', 'Sortino Ratio', 'CAGR', 'Max Drawdown']
        asset_returns["Description"] = ['MTD', 'Three month', 'Six month', 'YTD',
                                        'One year', 'Three year', 'Five year']
        asset_ratios = asset_ratios.set_index("Description")
        asset_returns = asset_returns.set_index("Description")
        asset_ratios = asset_ratios.drop(columns='index')
        asset_returns = asset_returns.drop(columns='index')
        asset_ratios = asset_ratios.dropna(how='any', axis=0)
        asset_returns = asset_returns.dropna(how='any', axis=0)
        asset_ratios.columns=['Metric']
        asset_ratios['Metric'] = asset_ratios['Metric'].astype(float).round(2)
        asset_returns.columns=['Returns %']
        asset_returns['Returns %'] = asset_returns['Returns %'].astype(float).round(2)
        return asset_ratios, asset_returns


    def monthly_pivot(ohlc):
        returns = monthly_rets(ohlc['close'])
        values = returns['close'].tolist()
        values = [str(round(value*100,2)) for value in values]
        returns['close'] = values
        #returns = returns.reset_index().set_index('Date')
        returns['Year'] = returns.index.strftime('%Y')
        returns['Month'] = returns.index.strftime('%b')
        months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", 
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        returns['Month'] = pd.Categorical(returns['Month'], categories=months, ordered=True)
        #returns['Quarter'] = np.where(returns.Month == "Dec","Q4",np.where(returns.Month == "Mar","Q1",np.where(returns.Month == "Sep","Q3","Q2")))
        #print(returns)
        returns = returns.pivot('Year', 'Month', 'close').fillna(0)
        #returns = returns.sort_values(by='Month')
        returns = returns.astype(float)
        #returns['Ticker'] = ticker
        #returns = returns.reset_index().set_index('Ticker')

        df = returns.copy()
        #print(df)
        return df

    def price_index(ohlc):
        rets = ohlc['close'].pct_change().dropna()
        pi = rets.to_price_index(start=100)
        return pi

    def dd(ohlc):
        ddown = ohlc['close'].to_drawdown_series()
        #dets = ohlc['close'].ffn.drawdown_details()
        return ddown

    # def arb():
    #     binance = ccxt.binance()
    #     bitbns = ccxt.bitbns()
    #     kraken = ccxt.kraken()
    #     bitfinex= ccxt.bitfinex()
    #     bitstamp = ccxt.bitstamp()
    #     exchanges = [binance,bitbns,kraken,bitfinex,bitstamp]
    #     prices = []
    #     for ex in exchanges:
    #         ticker = ex.fetch_ticker(symbol[:3].upper()+'/USDT')
            
    #         prices.append(ticker['last'])
    #     v = np.array(prices)
    #     x = (v[:, np.newaxis] - v[np.newaxis, :])
        
    #     arb = pd.DataFrame(x,columns = exchanges)
    #     arb['ex'] = exchanges
    #     arb = arb.set_index('ex')
    #     #print(arb)
    #     return arb.head()

    symbol =  access_index(data, 'symbol').upper()+'USDT'
    ohlc = ohlcCryphist(symbol,client.KLINE_INTERVAL_1DAY,"1 Jan,2018")
    ohlc['date'] = ohlc.index.date
    perf_ratios = get_crypto_perf_metrics(ohlc['close'],0.02)[0]
    perf_ratios = perf_ratios.astype(str)
    perf_rets = get_crypto_perf_metrics(ohlc['close'],0.02)[1]
    perf_rets = perf_rets.astype(str)
    monthly = monthly_pivot(ohlc)
    price_ind = price_index(ohlc)
    ddown = dd(ohlc)
    #arb_ = arb()
    #ddown_details = dd(ohlc)[1]

    fig = go.Figure(data=[go.Candlestick(x=ohlc.index,
                    open=ohlc['open'],
                    high=ohlc['high'],
                    low=ohlc['low'],
                    close=ohlc['close'],
                    name=symbol)])

    fig.update_xaxes(type='category')
    fig.update_layout(height=700,template="simple_white")

    #st.plotly_chart(fig, use_container_width=True)

    col1 = st.columns(2)
    #st.write(ohlc)
    df = pd.read_json('https://api.binance.com/api/v3/ticker/24hr')
    df = df[df['symbol']==symbol]
    price = df['prevClosePrice'].iloc[0]
    change = df['priceChangePercent'].iloc[0]
    #access_index(access_index(data, 'current_price'), currency)
    name = access_index(data, 'name').upper()
    #st.title(name)
    img =  access_index(access_index(data, 'image'), 'large')
    col1[0].image(img,width=100)
    #col1[1].write(name)
    #st.image(img,width=100)
    #cola[1].write(price)
    st.metric(label=symbol, value=price, delta=(change))
    #col1.metric(df.symbol,price,df.priceChangePercent)
    #st.subheader(price)


    st.plotly_chart(fig, use_container_width=True)
    data = pd.DataFrame.from_dict(entry, orient="index").reset_index()
    data = data.astype(str)
    data2 = pd.DataFrame.from_dict(entry_2, orient="index").reset_index()
    data2 = data2.astype(str)

    cols =  st.columns(2)
    cols[0].title('General Info')
    cols[0].dataframe(data)
    cols[1].title('Price Change %')
    cols[1].dataframe(data2)


    cols2 =  st.columns(2)
    cols2[0].title('Perf Metrics')
    cols2[0].dataframe(perf_ratios)
    cols2[1].title('Trailing Returns')
    cols2[1].dataframe(perf_rets)

    fig_2 = plt.figure(figsize=(10, 4))
    sns.heatmap(monthly,cmap="RdYlGn",annot=True)
    #fig_2.update_layout(height=700,template="simple_white")
    st.title("Monthly Returns (%)")
    st.pyplot(fig_2, use_container_width=True)

    st.title("$100 invested on {} would now be {}".format(ohlc['date'].iloc[0],np.round(price_ind.iloc[-1],2)))
    st.line_chart(price_ind)

    st.title("Underwater Plot")
    st.line_chart(ddown)
    #st.pyplot(fig_2, use_container_width=True)

    # print(arb_)
    # st.dataframe(arb_)
    # st.dataframe(perf_rets)
    #st.write(entry)


if option =="Scanner":
    option = st.sidebar.selectbox("Select Scanner", ('52 Week High', '52 Week low','Rsi','MACD Crossover', 'Up by atlest'))

    cursor.execute("""
    SELECT *  FROM daily_prices
    """)
    rows = cursor.fetchall()
    data = pd.DataFrame(rows,columns=['date','open','high','low','close','volume','symbol'])
    data.set_index('date',inplace=True)
    symbols = list(set(data['symbol']))

    
    if option =="52 Week High":
        ticks = []
        for symbol in symbols:
            try:
                datas = data[data['symbol']==symbol][-365:]
                max_price = max(datas['close'][:-1])
                last_price = datas['close'].iloc[-1] 
                #st.write(symbol,max_price,last_price)
                if last_price > max_price:
                    ticks.append(symbol)          
            except:
                pass
        st.write(ticks)

    if option =="52 Week low":
        lowticks = []
        for symbol in symbols:
            try:
                data = data[data['symbol']==symbol][-365:]
                min_price = min(data['close'][:-1])
                last_price = data['close'].iloc[-1] 
                if last_price < min_price:
                    lowticks.append(symbol)  
                    st.write(symbol,last_price)        
            except:
                pass
        st.write(lowticks)

    if option =="Rsi":
        condition = st.sidebar.selectbox("Select Threshold", ('Greater than', 'Lower than'))
        threshold = st.sidebar.text_input("Enter Threshold", value=50, max_chars=None, key=None, type='default')
        threshold = np.float64(threshold)
        for symbol in symbols:
            try:
                datas = data[data['symbol']==symbol]
                datas['rsi'] = ta.RSI(datas['close'],timeperiod=14)
                last = datas['rsi'].iloc[-1] 
                if condition=="Greater than" and last>threshold:
                    st.write(symbol,last)  
                if condition=="Lower than" and last<threshold:
                    st.write(symbol,last)            
            except:
                pass

    if option =="MACD Crossover":
        condition = st.sidebar.selectbox("Select Condition", ('Crosses below', 'Crosses above'))
        for symbol in symbols[:100]:
            try:
                datas = data[data['symbol']==symbol]
                datas['macd'] = ta.MACD(datas['close'],fastperiod=12, slowperiod=26, signalperiod=9)[0]
                datas['macdsig'] = ta.MACD(datas['close'],fastperiod=12, slowperiod=26, signalperiod=9)[1]
                datas['macdhis'] = ta.MACD(datas['close'],fastperiod=12, slowperiod=26, signalperiod=9)[2]
                #st.write(symbol)
                if condition=="Crosses below" and (datas['macd'].iloc[-2]>datas['macdsig'].iloc[-2] and datas['macd'].iloc[-1]<datas['macdsig'].iloc[-1]):
                    st.write(symbol)  
                if condition=="Crosses above" and (datas['macd'].iloc[-2]<datas['macdsig'].iloc[-2] and datas['macd'].iloc[-1]>datas['macdsig'].iloc[-1]):
                    st.write(symbol)             
            except:
                pass
        
    if option == 'Up by atlest':
        threshold = st.sidebar.text_input("Enter Threshold %", value=2, max_chars=None, key=None, type='default')
        threshold = np.float64(threshold)
        for symbol in symbols:
            try:
                datas = data[data['symbol']==symbol]
                last = datas['close'].iloc[-1] 
                prev = datas['close'].iloc[-2] 
                if (last-prev)/prev > threshold/100:
                    st.write(symbol,last)               
            except:
                pass



        
            


