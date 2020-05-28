import pandas as pd
import pickle
import json
from sklearn import  metrics
from sklearn import svm
from datetime import datetime, timedelta
import yfinance as yf
import argparse
import os

parser = argparse.ArgumentParser(description='Evaluate features for ITC')
#parser.add_argument('company_ticker', type=str, help='what ticker is this for')
parser.add_argument('model_version', type=str, help='what model version is this')

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
    
    model_df = pd.DataFrame(prev_closes)
    model_df = model_df.T
    for index in range(len(prev_closes)):
        model_df['open_{}'.format(index)] = prev_open[index]
        model_df['high_{}'.format(index)] = prev_high[index]
        model_df['low_{}'.format(index)] = prev_low[index]
        
    return model_df


def buy_nobuy(ticker, model_version):
    model_df = data_for_model(ticker)
    
    # Load in the model
    clf2 = pickle.load(open('./{}/data/{}/svm_model.sav'.format(model_version, ticker), 'rb'))
    with open('./{}/data/{}/meta_data.json'.format(model_version, ticker)) as json_file:
                meta = json.load(json_file)
    prediction = clf2.predict_proba(model_df)
    print (prediction)
    if prediction[:,1]> meta['threshold']:
        return "buy"
    else:
        return "no_buy"

def main(model_version):
    for pin in os.listdir('./{}/data/'.format(model_version)):
        if pin =='.DS_Store':
            continue
        elif not os.path.exists('./{}/data/{}/svm_model.sav'.format(model_version, pin)):
            print ('{} did not have a model'.format(pin))
        else:
            print ("{}: ".format(pin) + buy_nobuy(pin, model_version))

if __name__ == "__main__":
	args = parser.parse_args()
	main(args.model_version)
	#buy_nobuy(args.company_ticker, args.model_version)



