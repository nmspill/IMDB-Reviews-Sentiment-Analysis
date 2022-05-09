from re import X
import nltk
import pandas as pd

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


positive_words = get_words('data\\positive words.txt')
negative_words = get_words('data\\negative words.txt')
reviews = pd.read_csv('data\IMDB Dataset.csv')

positive_word_count = []
negative_word_count = []
review_type = []

for index, row in reviews.iterrows():
    adj_list = get_adj_list(row['review'])
    positive_word_count.append(get_scores(adj_list, positive_words, negative_words)['Positive']) 
    positive_word_count.append(get_scores(adj_list, positive_words, negative_words)['Negative'])

    if row['sentiment'] == 'positive':
        review_type.append('1')
    elif row['sentiment'] == 'negative':
        review_type.append('0')

data_dict = {'positive adj count': positive_word_count, 'negative adj count': negative_word_count, 'sentiment': review_type} 

data = pd.DataFrame(data_dict)
data.to_csv('features.csv', encoding='utf-8')
