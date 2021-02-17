# trading_bot
This repo I am trying to create an algorithmic stock trader, specifically, swing trading. 

First, I have some jupyter notebooks that I used to build a classification model using TPOT. 
I build a different model for each stock I am interested in trading. The classification is binary and classifies buy/no-buy.

The training data is the previous 45 closing prices, the 20 day rolling average, the 10 day rolling average, and a binary at each 
day saying if the 10 day is below or above the 20 day. 

The label is 1, if the stock should be a buy (Label a buy if the stock price in the following three days had a high 
greater than or equal to 5% higher from the open price on day 46.)

you run buy_nobuy_v2.py with the correct version number to determine if the model recommends you should buy a stock or not. 

The models are created from the notebook official_theory_adding_crossover.ipynb


