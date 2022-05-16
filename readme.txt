Authors:
    - Nolan Spillane
    - Patrick Cappello

How to Run:
    - Install the depedencies below
    - Run 'clean data.py' in order to generate the features used to train the models (results are already generated and can be found in data\features.csv)
    - Run 'logistic regression.py' to train the Logistic Regression model and evaluate using 10 fold cross validation
    - Run 'naiveBayes.py' to train the Multinomial Naive Bayes model and evaluate using 10 fold cross validation

Dependecies: 
    - matplotlib 3.3.4
    - pandas 1.2.1
    - numpy 1.19.5
    - nltk 3.7
    - scikit-learn 0.24.1

Resources Used:

IMDB Data: https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews?resource=download


Positive and Negative Words:

Positive Words: https://gist.github.com/mkulakowski2/4289437
Negative Words: https://gist.github.com/mkulakowski2/4289441

This file and the papers can all be downloaded from 
;    http://www.cs.uic.edu/~liub/FBS/sentiment-analysis.html
;
; If you use this list, please cite one of the following two papers:
;
;   Minqing Hu and Bing Liu. "Mining and Summarizing Customer Reviews." 
;       Proceedings of the ACM SIGKDD International Conference on Knowledge 
;       Discovery and Data Mining (KDD-2004), Aug 22-25, 2004, Seattle, 
;       Washington, USA, 
;   Bing Liu, Minqing Hu and Junsheng Cheng. "Opinion Observer: Analyzing 
;       and Comparing Opinions on the Web." Proceedings of the 14th 
;       International World Wide Web conference (WWW-2005), May 10-14,  
;       2005, Chiba, Japan.
;
; Notes: 
;    1. The appearance of an opinion word in a sentence does not necessarily  
;       mean that the sentence expresses a positive or negative opinion. 
;       See the paper below:
;
;       Bing Liu. "Sentiment Analysis and Subjectivity." An chapter in 
;          Handbook of Natural Language Processing, Second Edition, 
;          (editors: N. Indurkhya and F. J. Damerau), 2010.
;
;    2. You will notice many misspelled words in the list. They are not 
;       mistakes. They are included as these misspelled words appear 
;       frequently in social media content. 
;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

