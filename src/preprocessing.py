import pandas as pd
import numpy as np
import nltk
import re
from emoji import demojize
import demoji
import contractions
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords

def emoji_converter(text):
    ##Convert emojis to text (ex: :) --> :smiling_face:)
    emoji_dict = demoji.findall(text)
    for k, v in emoji_dict.items():
        text = text.replace(k, ' :'+'_'.join(v.split())+': ')
    return text

def preprocess_tweets(text):
    #Apply tweet tokenizer
    tk = TweetTokenizer()
    tokens = tk.tokenize(text)
    return ' '.join(tokens)

def clean_tweet_for_transformers(tweet):
    if type(tweet) == np.float:
        return ""
    ##Set label @USER for mentions
    temp = re.sub("@[A-Za-z0-9_]+","@USER", tweet)
    ## Discard # symbol (this is not done in the preprocessing of data that have been used for BERTweet training, 
    # however I oibtained better result by omitting them)
    #temp = re.sub("#","", temp)
    ##Set label for URLs
    temp = re.sub('http\S+', 'HTTPURL', temp)
    ##Apply function to convert emojis
    #temp = emoji_converter(temp)
    temp = ''.join([demojize(token) for token in temp])
    ##Tokenize string with TweetTokenizer
    temp = preprocess_tweets(temp)
    return temp


def deep_clean_tweet(tweet):
    #Avoid float issues
    if type(tweet) == np.float:
        return ""
    #Substitute mentions with '@USER'
    temp = re.sub("@[A-Za-z0-9_]+","@USER", tweet)
    #Discard # from hashtags
    #temp = re.sub("#","", temp)
    #Convert emojis into text
    temp = emoji_converter(temp)
    #Assign 'HTTPURL' to links
    temp = re.sub(r'http\S+', 'HTTPURL', temp)
    #Discard numbers
    temp = re.sub(r'\d+', '', temp)
    #Discard special characters
    temp = re.sub('[()!?]', ' ', temp)
    temp = re.sub('\[.*?\]',' ', temp)
    temp = re.sub(r'[^\w\s<>@]'," ", temp)
    #Split string and substitute contractions with uncontracted form
    temp = temp.split()
    temp = [contractions.fix(word) for word in temp]
    #Discard stopwords
    temp = [w for w in temp if not w in stopwords.words('english')]
    #Rejoin tweets and apply final tokenization
    temp = " ".join(word for word in temp)
    temp = preprocess_tweets(temp)
    return temp