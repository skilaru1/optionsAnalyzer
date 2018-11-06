# Popular Technical Indicators
import numpy as np
import pandas as pd
import quandl
import matplotlib.pyplot as plt


quandl.ApiConfig.api_key = 'Y5rrxri7wsuNyXE6c-mj'
brkb = quandl.get('WIKI/BRK_B')

def vecSMA(stock, ndays):
    SMA = []
    for i in range(0, len(stock)-ndays):
        x = stock[i:i+ndays].mean()
        SMA.append(x)
    return SMA

def vecEMA(stock, ndays):
    ewma = pd.Series.ewm
    fwd = ewma(stock,span=ndays).mean() # take EWMA in fwd direction
    bwd = ewma(stock[::-1],span=ndays).mean() # take EWMA in bwd direction
    EMA = np.vstack(( fwd, bwd[::-1] )) # lump fwd and bwd together
    EMA = np.mean(EMA, axis=0 ) # average
    return EMA

def MACD(stock, short, long):
    return np.subtract(vecEMA(stock, short),vecEMA(stock, long))

def MACD_Hist(stock, short, long, signal):
    return np.subtract(vecEMA(stock, signal), MACD(stock, short, long))

def RSI(stock, period):
    change = np.diff(stock)
    first = change[0:(period-1)]
    avgGain = [np.mean(first[first > 0])]
    avgLoss = [np.mean(first[first <= 0])]
    for x in change[period:]:
        print(change[0])
        if x > 0:
            avgGain = np.append(avgGain, (avgGain[-1]*(period-1) + x)/period)
            avgLoss = np.append(avgLoss, avgLoss[-1])
        else:
            avgLoss = np.append(avgLoss, (avgLoss[-1]*(period-1) + x)/period)
            avgGain = np.append(avgGain, avgGain[-1])
    rs = avgGain / abs(avgLoss)
    return (100 - (100/(1 + rs)))

def CCI(stock, periods):
    typicalPrice = (stock['Adj. High'] + stock['Adj. Low'] + stock['Adj. Close']) / 3
    typicalPriceSMA = vecSMA(typicalPrice, periods)
    meanDev = sum(abs((typicalPrice - typicalPriceSMA))) / periods
    return (typicalPrice - typicalPriceSMA) / (0.15 * meanDev)

def BollingerBands(stock, periods=20):
    middleBand = np.array(vecSMA(stock, 20))
    stdDev = []
    for x in range(0, len(stock)-periods):
        stdDev = np.append(stdDev, np.std(stock[x:periods+x]))
    upperBand = middleBand + (stdDev * 2)
    lowerBand = middleBand - (stdDev * 2)
    x = np.vstack((lowerBand, middleBand, upperBand))
    return x

bmacd = MACD(brkb, 100, 300)
plt.plot(bmacd)

plt.show()

# News Parsing
# import requests
# from bs4 import BeautifulSoup
# import nltk
# from nltk.tokenize import sent_tokenize
# from nltk.tokenize import word_tokenize
#
# # nltk.download('all')
#
# news = requests.get("https://seekingalpha.com/news/3404942-berkshire-hathaway-q3-beats-consensus-investment-gain-solid-operations")
#
# soup = BeautifulSoup(news.content, 'html.parser')
#
# head = str(soup.find('title'))
# print(head)
# head_tokenize_list = word_tokenize(head)
# print(head_tokenize_list)
