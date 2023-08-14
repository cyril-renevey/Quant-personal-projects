import yfinance as yf
import numpy as np
import pandas as pd
import scipy.stats as stats
import datetime as dt
from copy import deepcopy
from scipy.integrate import quad
from scipy.optimize import minimize
from scipy.optimize import fsolve

# define a function that test is a specific date is a business day
def is_BD(date):
    return bool(len(pd.bdate_range(date, date)))

def BS_call(S, K, T, r, sigma):
    """
    Calculate the theoretical price of a European call option using the Black-Scholes formula.

    Parameters:
        S (float or np.ndarray): Current stock price
        K (float or np.ndarray): Strike price of the option
        T (float or np.ndarray): Time to expiration (in years)
        r (float): Risk-free rate (annualized, expressed as a decimal)
        sigma (float): Volatility of the stock (annualized, expressed as a decimal)

    Returns:
        float or np.ndarray: Theoretical price of the call option
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    call_price = S * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2)
    return call_price

def BS_put(S, K, T, r, sigma):
    """
    Calculate the theoretical price of a European put option using the Black-Scholes formula.

    Parameters:
        S (float or np.ndarray): Current stock price
        K (float or np.ndarray): Strike price of the option
        T (float or np.ndarray): Time to expiration (in years)
        r (float): Risk-free rate (annualized, expressed as a decimal)
        sigma (float): Volatility of the stock (annualized, expressed as a decimal)

    Returns:
        float or np.ndarray: Theoretical price of the put option
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    put_price = K * np.exp(-r * T) * stats.norm.cdf(-d2) - S * stats.norm.cdf(-d1)
    return put_price

class Market:
    def __init__(self,stocks_list: list,start_date,end_date):
        
        self.tickers = [i.replace('^','') for i in stocks_list]

        tempo_stocks = yf.download(tickers=stocks_list,start=start_date,end=end_date+dt.timedelta(days=1),interval='1d').Close
        tempo_stocks_hourly = yf.download(tickers=stocks_list,start=start_date,end=end_date+dt.timedelta(days=1),interval='1h'\
                                          ,ignore_tz=True).Close.pct_change().dropna()
        tempo_vol = tempo_stocks_hourly.groupby(pd.Grouper(freq='1B')).std().dropna()*np.sqrt(1440/60*252)

        dict_tempo={i.replace('^',''): pd.DataFrame(data = np.vstack([tempo_stocks[i].values,tempo_vol[i].values]).T, \
                                    index = tempo_stocks.index,columns = ['price','histVol']) for i in stocks_list}
        self.rate = \
                yf.download(tickers='^IRX',start=start_date,end=end_date+dt.timedelta(days=1),interval='1d')[['Close']]/100
        self.stocks = dict_tempo

        self.date_range = tempo_stocks.index.tolist()
        self.start = self.date_range[0]
        self.end = self.date_range[-1]

        dict_options = {}
        for x in self.tickers:
            df_ticker = pd.DataFrame(columns = ['date','lastPrice','strike','strikeDist','expiration','DTE',\
           'callVolume','callPrice','callIV','putPrice','putVolume','putIV','C_BID','C_ASK'])
            
            for y in {x.strftime("%Y%m") for x in self.date_range}:
                data=pd.read_csv('option_data/'+x+'_eod_'+y+'.txt', sep=',',low_memory=False)

                data.columns=[x.replace('[','').replace(']','').replace(' ','') for x in data.columns]
                data=data[['QUOTE_DATE','UNDERLYING_LAST','STRIKE','STRIKE_DISTANCE','EXPIRE_DATE','DTE',\
                           'C_VOLUME','C_LAST','C_IV','P_LAST','P_VOLUME','P_IV','C_BID','C_ASK']]

                data=data.replace(' ','0.00')
                for i in [x for x in data.columns if "DATE" not in x]:
                    data[i]=data[i].astype('float')

                data=data[((data.C_VOLUME>10) | (data.P_VOLUME>10)) & (data.DTE>0)\
                            & (data.STRIKE.isin(set(data.STRIKE.values.round())))]
                data=data.loc[data.groupby(data.QUOTE_DATE).idxmin().STRIKE_DISTANCE]

                data['EXPIRE_DATE']=[pd.to_datetime(x) for x in data.EXPIRE_DATE]
                data['QUOTE_DATE']=[pd.to_datetime(x) for x in data.QUOTE_DATE]

                data=data.loc[data.groupby(data.QUOTE_DATE).idxmin().DTE.values]

                data.columns=df_ticker.columns

                df_ticker = pd.concat([df_ticker,data])
            tempo_df = df_ticker.set_index('date')
            dict_options[x] = tempo_df[(tempo_df.index<=end_date) & (tempo_df.index>=start_date) \
                                & (tempo_df.index.isin(tempo_stocks.index))][['strike','DTE','callPrice','callIV','putPrice','putIV']]
            
            dict_options[x]['callIV'] = [fsolve(lambda z:BS_call(self.stocks[x].price[time],dict_options[x].strike[time],dict_options[x].DTE[time]/365,\
                        self.rate.loc[time],z)-dict_options[x].callPrice[time],0.5)[0] for time in dict_options[x].index]
            dict_options[x]['putIV'] = [fsolve(lambda z:BS_put(self.stocks[x].price[time],dict_options[x].strike[time],dict_options[x].DTE[time]/365,\
                        self.rate.loc[time],z)-dict_options[x].putPrice[time],0.5)[0] for time in dict_options[x].index]
            
        self.options = dict_options


class Position:
    """
    Defines the instantaneous position of a portfolio

    Attributs:
        market (Market): market on which the position is defined
        stock_pos (list): [money account, # of stock 1, # of stock 2, etc.]
        call_pos (list):[call_stock_1, etc.] where call_stock_1=[[strike, DTE, # of call1],[strike, DTE, # of call2],...]
        put_pos (list): same as call
        time (timestamp): instantaneous time of the position

    methods:
        value(self): return the value of the position
        step(self): evolves the position by one business day and return the new position
    """
    def __init__(self, market: Market,time,stock_pos=[0,0],call_pos = [],put_pos = []):
        self.market = market
        self.time = time
        self.stock_pos = stock_pos
        self.call_pos = call_pos
        self.put_pos = put_pos

    def value(self):
        money_value = self.stock_pos[0]
        stock_value = (np.array([y.loc[self.time].price for x,y in self.market.stocks.items()])*np.array(self.stock_pos[1:])).sum()
        
        a = self.market
        call_value = sum([sum([j[2]*BS_call(a.stocks[a.tickers[i]].loc[self.time].price,\
                    j[0], j[1]/365, a.rate.loc[self.time].Close, \
                    a.options[a.tickers[i]].loc[self.time].callIV) for j in self.call_pos[i]]) for i in range(len(self.call_pos))])
        put_value = sum([sum([j[2]*BS_put(a.stocks[a.tickers[i]].loc[self.time].price,\
                    j[0], j[1]/365, a.rate.loc[self.time].Close, \
                    a.options[a.tickers[i]].loc[self.time].putIV) for j in self.put_pos[i]]) for i in range(len(self.put_pos))])
        
        return money_value+stock_value+call_value+put_value
        
    def step(self):
        if self.time < self.market.end:   

            self.stock_pos[0] = self.stock_pos[0]*(1+self.market.rate.Close[self.time])**(1/252)
            
            self.time = self.time+dt.timedelta(days=1)
            diff_day=1
            
            while not self.time in self.market.date_range:
                self.time = self.time+dt.timedelta(days=1)
                diff_day=diff_day+1
                
            self.call_pos = [[(np.array(y)-[0,diff_day,0]).tolist() for y in x] for x in self.call_pos]
            self.put_pos = [[(np.array(y)-[0,diff_day,0]).tolist() for y in x] for x in self.put_pos]
            
            for i in range(len(self.call_pos)):
                stock=self.market.tickers[i]
                ST=self.market.stocks[stock].price[self.time]
                
                self.stock_pos[0]=self.stock_pos[0]+sum([y[2]*np.maximum(0,ST-y[0]) for y in self.call_pos[i] if y[1]<= 0])
                self.call_pos[i]=[y for y in self.call_pos[i] if y[1]> 0]

                self.stock_pos[0]=self.stock_pos[0]+sum([y[2]*np.maximum(0,y[0]-ST) for y in self.put_pos[i] if y[1]<= 0])
                self.put_pos[i]=[y for y in self.put_pos[i] if y[1]> 0]

class Portfolio:
    def __init__(self,pos_ini,end_date,strategy = lambda x,param: x,param_ini=[]):
        self.pos_ini = pos_ini
        self.date_range = [x for x in self.pos_ini.market.date_range if ((x>=self.pos_ini.time) & (x<=end_date))]
        self.start = self.date_range[0]
        self.end = self.date_range[-1]
        self.strategy = strategy
        
        tempo_portfolio=[deepcopy(pos_ini)]
        while pos_ini.time<self.end:
            pos_ini.step()
            pos_ini=strategy(pos_ini,param_ini)
            new_pos=deepcopy(pos_ini)
            tempo_portfolio.append(new_pos)
        self.portfolio=tempo_portfolio
        
        self.value = pd.DataFrame(data=[i.value() for i in self.portfolio],index=self.date_range,columns = ['portfolio_value'])
        self.value.index.name='date'

    def compare(self,benchmark):
        comparison=pd.concat([self.value/self.value.loc[self.start], benchmark.value/benchmark.value.loc[benchmark.start]], axis=1)
        comparison.columns=['our_strat','benchmark']
        comparison.plot()

    def is_self_financing(self):
        portfolio = deepcopy(self.portfolio)
        for x in portfolio:
            x.step()
        return [self.portfolio[i].value()-portfolio[i-1].value() for i in range(1,len(portfolio))]
