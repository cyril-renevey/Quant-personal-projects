import yfinance as yf
import numpy as np
import os
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

        if include_options:
            dict_options = {}
            
            for tick in deepcopy(stocks_list):
                if tick+'_2013.csv' in os.listdir('./option_data/df_SandP500_2013'):
                    data=pd.read_csv('./option_data/df_SandP500_2013/'+tick+'_2013.csv',\
                                     sep=',',low_memory=False,infer_datetime_format=True,index_col='quote_date')
                    data.index = pd.to_datetime(data.index)
                    if data[data.type=='put'].empty | data[data.type=='call'].empty:
                        stocks_list.remove(tick)
                    else:
                        dict_options[tick]=data[(data.index<=end_date) & (data.index>=start_date)]
                        dict_options[tick]=dict_options[tick].drop_duplicates(keep='first')
                else:
                    stocks_list.remove(tick)
            self.options=dict_options
            
            
        self.tickers = stocks_list

        dict_stock={tick:pd.read_csv('./option_data/df_SandP500_2013/'+tick+'_stock_2013.csv',\
                                     sep=',',low_memory=False,index_col='date') for tick in stocks_list}
        for tick in dict_stock:
            dict_stock[tick].index = pd.to_datetime(dict_stock[tick].index)
            dict_stock[tick]=dict_stock[tick][(dict_stock[tick].index<=end_date) & (dict_stock[tick].index>=start_date)]
            dict_stock[tick]=dict_stock[tick][['close','volume']]
        
        self.date_range = dict_stock[stocks_list[0]].index.tolist()
        
        self.rate = \
                yf.download(tickers='^IRX',start=start_date,end=end_date+dt.timedelta(days=1),interval='1d')[['Close']]/100
        self.rate=self.rate.rename(columns={'Close':'close'})
        
        self.stocks = dict_stock

        self.start = self.date_range[0]
        self.end = self.date_range[-1]
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
        stock_value = (np.array([y.close[self.time] for y \
                                 in a.stocks.values()])*np.array(list(self.stock_pos.values()))).sum()

        calls={x:a.options[x][a.options[x].type=='call'] for x in a.tickers}
        calls={x:calls[x].loc[min(calls[x].index.tolist(), key=lambda x: abs(x - self.time))] for x in calls}
        puts={x:a.options[x][a.options[x].type=='put'] for x in a.tickers}
        puts={x:puts[x].loc[min(puts[x].index.tolist(), key=lambda x: abs(x - self.time))] for x in puts}

        def near_strike(i,k):
            return min(calls[i].strike.tolist(), key=lambda x: abs(x - k))
        def near_dte(i,T,k):
            return min(calls[i][calls[i].strike==near_strike(i,k)].DTE.tolist(),key=lambda x: abs(x-T))   
        call_value = sum([sum([j[2]*BS_call(a.stocks[i].close[self.time],j[0], j[1]/365, a.rate.close[self.time], \
                    calls[i][(calls[i].strike==near_strike(i,j[0])) & (calls[i].DTE==near_dte(i,j[1],j[0]))].implied_volatility[0]\
                             ) for j in self.call_pos[i]]) for i in a.tickers])

        def near_strike(i,k):
            return min(puts[i].strike.tolist(), key=lambda x: abs(x - k))
        def near_dte(i,T,k):
            return min(puts[i][puts[i].strike==near_strike(i,k)].DTE.tolist(),key=lambda x: abs(x - T))
        put_value = sum([sum([j[2]*BS_put(a.stocks[i].close[self.time],j[0], j[1]/365, a.rate.close[self.time], \
                    puts[i][(puts[i].strike==near_strike(i,j[0])) & (puts[i].DTE==near_dte(i,j[1],j[0]))].implied_volatility[0]\
                             ) for j in self.put_pos[i]]) for i in a.tickers])
        
        return self.money+stock_value+call_value+put_value
        
    def step(self):
        if self.time < self.market.end:   

            r = self.market.rate.close[self.time]
            
            self.time = self.time+dt.timedelta(days=1)
            diff_day=1
            while not self.time in self.market.date_range:
                self.time = self.time+dt.timedelta(days=1)
                diff_day=diff_day+1

            self.money = self.money*(1+r)**(diff_day/365)
            self.call_pos = {name: [(np.array(y)-[0,diff_day,0]).tolist() for y in pos] for name,pos in self.call_pos.items()}
            self.put_pos = {name: [(np.array(y)-[0,diff_day,0]).tolist() for y in pos] for name,pos in self.put_pos.items()}
            
            for stock in self.market.tickers:
                ST=self.market.stocks[stock].close[self.time]
                
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
