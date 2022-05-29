
import torch 
import pandas as pd
import argparse
from transformers import BertForSequenceClassification
from bert_train import bert_run
def main():
    parser = argparse.ArgumentParser(description='Run Experiments for English & Spanish Data', prefix_chars='-+')

    parser.add_argument('--trainData', dest='location_train', default='../dataset/', 
                            help='The directory for retrieving the train data') 
    
    parser.add_argument('--tweetContent_train', dest='text_col_train', default='tweet_text', 
                            help='The column name for the tweet content') 

    parser.add_argument('--tweetLabel', dest='id_col_train', default='content_behavior', 
                            help='The column name for the tweet label in train file') 
 
    parser.add_argument('--testData', dest='location', default='../dataset/', 
                            help='The directory for retrieving the test data') 
    
    parser.add_argument('--outputFile', dest='out', default='results.csv',
                            help='The output file')
    
    parser.add_argument('--tweetContent', dest='text_col', default='content', 
                            help='The column name for the tweet content') 

    parser.add_argument('--tweetID', dest='id_col', default='tweet_ids', 
                            help='The column name for the tweet id')  


    args = parser.parse_args()
    # model = BertForSequenceClassification.from_pretrained('bert_models')
    model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased', num_labels = 2)
    # model.load_state_dict(torch.load('/home/xw27/dataset/hiv_tweet/hiv-actionability/baseline/model.pth.tar'))
    bert_run(args.location_train, args.text_col_train, args.id_col_train, args.location, args.out, args.text_col, args.id_col, model)

main()
