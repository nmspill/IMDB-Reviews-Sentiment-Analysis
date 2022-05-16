import re
import nltk
import pandas as pd
from nltk.tokenize import RegexpTokenizer
import math


# nltk.download('averaged_perceptron_tagger')
# nltk.download('stopwords')
# nltk.download('punkt')

def get_scores(adj_list, pos_list, neg_list):
    scores = {'Positive': 0, 'Negative': 0}
    for pos_tuple in adj_list:
        word, tag = pos_tuple
        if word in pos_list:
            scores['Positive'] += 1
        if word in neg_list:
            scores['Negative'] += 1
    return scores

def get_words(path):
    a_file = open(path, 'r')
    lines = a_file.read()
    list_of_lists = lines.splitlines()
    a_file.close()
    return list_of_lists

def get_adj_list(str):
    text = nltk.word_tokenize(str)
    adj = []
    for pos_tuple in nltk.pos_tag(text):
        word, tag = pos_tuple
        if tag == 'JJ' or tag == 'JJR' or tag == 'JJS':
            adj.append(pos_tuple)
    return adj

def get_word_count(str):
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(str)
    return round(math.log(len(tokens)), 3)

def get_exclamation_point(str):
    if re.search('\!', str):
        return 1
    return 0

def get_no(str):
    if re.search('(?:^|\W)no(?:$|\W)', str, re.IGNORECASE):
        return 1
    return 0

positive_words = get_words('data\\positive words.txt')
negative_words = get_words('data\\negative words.txt')
reviews = pd.read_csv('data\\IMDB Dataset.csv')

positive_word_count = []
negative_word_count = []
word_count = []
exclamation_point = []
no = []
review_type = []

for index, row in reviews.iterrows():
    adj_list = get_adj_list(row['review'])
    positive_word_count.append(get_scores(adj_list, positive_words, negative_words)['Positive'])
    negative_word_count.append(get_scores(adj_list, positive_words, negative_words)['Negative'])
    word_count.append(get_word_count(row['review']))
    exclamation_point.append(get_exclamation_point(row['review']))
    no.append(get_no(row['review']))


    if row['sentiment'] == 'positive':
        review_type.append('1')
    elif row['sentiment'] == 'negative':
        review_type.append('0')

data_dict = {'positive adj count': positive_word_count, 'negative adj count': negative_word_count,'word count': word_count, 'contains !': exclamation_point, "contains 'no'": no, 'sentiment': review_type} 

data = pd.DataFrame(data_dict)
data.to_csv('features.csv', encoding='utf-8')
