import pandas as pd
import pickle
import json
from sklearn import  metrics
from sklearn import svm
from datetime import datetime, timedelta
import yfinance as yf
import argparse
import os
import numpy as np

parser = argparse.ArgumentParser(description='Evaluate features for ITC')
#parser.add_argument('company_ticker', type=str, help='what ticker is this for')
parser.add_argument('model_version', type=str, help='what model version is this')



def calculating_average_crossover(prev_closes):
    my_df = pd.DataFrame(prev_closes)
    twenty = my_df.T.rolling(window = 20, axis = 1).mean().iloc[:,-26:].values[0]
    ten = my_df.T.rolling(window = 10, axis = 1).mean().iloc[:,-26:].values[0]
    
    twenty_over_ten = []
    for i in range(0,26):
        if twenty[i]>ten[i]:
            twenty_over_ten.append(0)
        else:
            twenty_over_ten.append(1)
    return twenty, ten, twenty_over_ten

def company_data(ticker):
    days_to_subtract = 90
    end_date = (datetime.today())
    start_date = end_date-timedelta(days=days_to_subtract)

    end_date = end_date.strftime('%Y-%m-%d')
    start_date = start_date.strftime('%Y-%m-%d')
    print (end_date)
    
    company = yf.Ticker(ticker)
    historical_df = company.history(period='1d', interval = '1d', start = start_date)
    return historical_df


def data_for_model(ticker):
    historical_df = company_data(ticker)
    open_prices = historical_df['Open'].values
    close_prices = historical_df['Close'].values
    high_prices = historical_df['High'].values
    low_prices = historical_df['Low'].values
    
    
    prev_closes = close_prices[-45:]
    prev_high = high_prices[-45:]
    prev_open = open_prices[-45:]
    prev_low = low_prices[-45:]
    
    # get the averages and crossovers
    twenty_average, ten_average, cross_over = calculating_average_crossover(prev_closes)
    
    model_df = pd.DataFrame(prev_closes)
    model_df = model_df.T
    for index in range(len(prev_closes)):
        model_df['open_{}'.format(index)] = prev_open[index]
        model_df['high_{}'.format(index)] = prev_high[index]
        model_df['low_{}'.format(index)] = prev_low[index]
    
    for index in range(len(twenty_average)):
                model_df['twenty_average_{}'.format(index+19)] = twenty_average[index]
                model_df['ten_average_{}'.format(index+19)] = ten_average[index]
                model_df['crossover_{}'.format(index+19)] = cross_over[index]
        
    return model_df


def buy_nobuy(ticker, model_version):
    # Check to see the model is good
    with open('./{}/data/{}/purchase_results.json'.format(model_version, ticker)) as json_file:
        results = json.load(json_file)
    if 'Purchases' in results:
        return "Test set no purchases"
    elif results['hold_strategy_roi'] < .035:
        return "Bad ROI"
    
    print ("Rate of purchases that had to be holds: {}".format(len(results['days_held'])/results['purchases']))
    if len(results['days_held']) > 0:
        print ("Average number of days held: {}".format(np.average(results['days_held'])))
    model_df = data_for_model(ticker)
    
    # Load in the model
    model = pickle.load(open('./{}/data/{}/model.sav'.format(model_version, ticker), 'rb'))
    with open('./{}/data/{}/meta_data.json'.format(model_version, ticker)) as json_file:
                meta = json.load(json_file)
    prediction = model.predict_proba(model_df)
    print ("prediction: {}".format(prediction))
    print ("threshold: {}".format(meta['threshold']))
    if prediction[:,1]> meta['threshold']:
        return "buy"
    else:
        return "no_buy"

def main(model_version):
    for pin in os.listdir('./{}/data/'.format(model_version)):
        print (pin)
        if pin == '.DS_Store':
            continue
        elif not os.path.exists('./{}/data/{}/model.sav'.format(model_version, pin)):
            print ('{} did not have a model'.format(pin))
            print ('\n')
        else:
            print ("{}: ".format(pin) + buy_nobuy(pin, model_version))
            print ('\n')

if __name__ == "__main__":
    args = parser.parse_args()
    main(args.model_version)
    #buy_nobuy(args.company_ticker, args.model_version)