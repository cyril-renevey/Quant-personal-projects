import yfinance as yf
import numpy as np
import pandas as pd
import scipy.stats as stats
import datetime as dt
from copy import deepcopy
from scipy.integrate import quad
from scipy.optimize import minimize
from scipy.optimize import fsolve

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
    """
    Defines the market composed of stocks and at the money, shortest maturity european options 

    Attributs:
        tickers (list): list of the stock tickers
        rate (dataframe): the risk free rate with respect to date
        stocks (dict of dataframes): dictionnary with stock tickers as keys and the dataframe of the corresponding stock. 
        the dataframe includes the daily close price and annualized volatility calculated with hourly data.
        date_range (list): list of dates 
        start (timestamp): initial time
        end (timestamp): final time
        options (dict of dataframes): dictionnary with stock tickers as keys and the dataframe of the corresponding options. 
        the dataframe includes the following options data ('strike','DTE','callPrice','callIV','putPrice','putIV') 
        and only the at the money and shortest maturity options are included. Only a limited time: jan to jun 2013 included.

    methods:
        none
    """
    def __init__(self,stocks_list,start_date,end_date,include_options=False):
        
        self.tickers = [i.replace('^','') for i in stocks_list]

        column_values=['Close','Volume']
        tempo_stocks = yf.download(tickers=stocks_list,start=start_date,end=end_date+dt.timedelta(days=1),interval='1d')[column_values]
        self.date_range = tempo_stocks.index.tolist()
        
        dict_tempo={i.replace('^',''): pd.DataFrame(data=np.vstack([tempo_stocks[x][i].values\
                                for x in column_values]).T, index=self.date_range,columns=column_values)\
                                for i in stocks_list}
        
        self.rate = \
                yf.download(tickers='^IRX',start=start_date,end=end_date+dt.timedelta(days=1),interval='1d')[['Close']]/100
        self.stocks = dict_tempo

        self.start = self.date_range[0]
        self.end = self.date_range[-1]
        if include_options:
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
                    data=data.loc[data.groupby(data.QUOTE_DATE).idxmin(numeric_only=True).STRIKE_DISTANCE]

                    data['EXPIRE_DATE']=[pd.to_datetime(x) for x in data.EXPIRE_DATE]
                    data['QUOTE_DATE']=[pd.to_datetime(x) for x in data.QUOTE_DATE]

                    data=data.loc[data.groupby(data.QUOTE_DATE).idxmin(numeric_only=True).DTE.values]

                    data.columns=df_ticker.columns

                    df_ticker = pd.concat([df_ticker,data])
                tempo_df = df_ticker.set_index('date')
                dict_options[x] = tempo_df[(tempo_df.index<=end_date) & (tempo_df.index>=start_date) \
                                & (tempo_df.index.isin(tempo_stocks.index))][['strike','DTE','callPrice','callIV','putPrice','putIV']]
            
                dict_options[x]['callIV'] = [fsolve(lambda z:BS_call(self.stocks[x].Close[time],dict_options[x].strike[time],dict_options[x].DTE[time]/365,\
                        self.rate.loc[time],z)-dict_options[x].callPrice[time],0.5)[0] for time in dict_options[x].index]
                dict_options[x]['putIV'] = [fsolve(lambda z:BS_put(self.stocks[x].Close[time],dict_options[x].strike[time],dict_options[x].DTE[time]/365,\
                        self.rate.loc[time],z)-dict_options[x].putPrice[time],0.5)[0] for time in dict_options[x].index]
            
            self.options = dict_options


class Position:
    """
    Defines the instantaneous position of a portfolio

    Attributs:
        market (Market): market on which the position is defined
        stock_pos (dict of floats): {ticker stock #1: # of stock #1, etc}
        money (float): money account value
        call_pos (dict of lists of lists of int/float):{ticker stock #1:[[strike, DTE, # of call1],[strike, DTE, # of call2],...], etc}
        put_pos (dict of lists of lists of int/float): same as call
        time (timestamp): instantaneous time of the position

    methods:
        value(self): return the value of the position
        step(self): evolves the position by one business day and return the new position
    """
    def __init__(self, market,time,money_account,stock_pos,call_pos,put_pos):
        self.market = market
        while (time not in market.date_range) & (time<=market.end):
            time = time + dt.timedelta(days=1)
        self.time = time
        self.stock_pos = stock_pos
        self.money = money_account
        self.call_pos = call_pos
        self.put_pos = put_pos

    def value(self):
        a = self.market
        stock_value = (np.array([y.Close[self.time] for y \
                                 in a.stocks.values()])*np.array(list(self.stock_pos.values()))).sum()
        call_value = sum([sum([j[2]*BS_call(a.stocks[i].Close[self.time],\
                    j[0], j[1]/365, a.rate.Close[self.time], \
                    a.options[i].callIV[self.time]) for j in self.call_pos[i]]) for i in a.tickers])
        put_value = sum([sum([j[2]*BS_put(a.stocks[i].Close[self.time],\
                    j[0], j[1]/365, a.rate.Close[self.time], \
                    a.options[i].putIV[self.time]) for j in self.put_pos[i]]) for i in a.tickers])
        
        return self.money+stock_value+call_value+put_value
        
    def step(self):
        if self.time < self.market.end:   

            r = self.market.rate.Close[self.time]
            
            self.time = self.time+dt.timedelta(days=1)
            diff_day=1
            while not self.time in self.market.date_range:
                self.time = self.time+dt.timedelta(days=1)
                diff_day=diff_day+1

            self.money = self.money*(1+r)**(diff_day/365)
            self.call_pos = {name: [(np.array(y)-[0,diff_day,0]).tolist() for y in pos] for name,pos in self.call_pos.items()}
            self.put_pos = {name: [(np.array(y)-[0,diff_day,0]).tolist() for y in pos] for name,pos in self.put_pos.items()}
            
            for stock in self.market.tickers:
                ST=self.market.stocks[stock].Close[self.time]
                
                self.money=self.money+sum([y[2]*np.maximum(0,ST-y[0]) for y in self.call_pos[stock] if y[1]<= 0])
                self.call_pos[stock]=[y for y in self.call_pos[stock] if y[1]> 0]

                self.money=self.money+sum([y[2]*np.maximum(0,y[0]-ST) for y in self.put_pos[stock] if y[1]<= 0])
                self.put_pos[stock]=[y for y in self.put_pos[stock] if y[1]> 0]

class Portfolio:
    """
    Defines a portfolio which is a list of positions. Each positions are connected using a function strategy that describes
    how to re-alocated the positions at each step.

    Attributs:
        pos_ini (position): initial position
        date_range (list of timestamp): the date range spanning the portfolio's history
        start (timestamp): initial time
        end (timestamp): final time
        strategy (function): the strategy that is applied at each time step
        portfolio (list of positions): the history of the portfolio
        value (dataframe): dataframe of the values of the positions wrt dates

    methods:
        compare(self,benchmark): returns a plot of the benchmark and self portfolio's values, benchmark.value and the self.value
        is_self_financing(self): returns the list of additional money that needs to be invested at each time. 
                                 if it is a list of values very close to zero, the portfolio is self financing.
    """
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
