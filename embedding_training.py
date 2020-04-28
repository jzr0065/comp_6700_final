# import modules & set up logging
import gensim, logging
import codecs
import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
def read():
    filename = "5G.txt"
    text = codecs.open(filename, "r", "utf-8").read()
    sent_tokens = sent_tokenize(text)

    # print(sent_tokens)
    df = pd.DataFrame(sent_tokens, columns=['sent'])
    df['clean_sent'] = df['sent'].apply(word_tokenize)

    stopWords = set(stopwords.words('english'))
    punctuation = [',','.','?','(',')']

    token_list = list()
    #remove the stop words and punctuations
    for index, row in enumerate(df['clean_sent']):
        word_filter = list()
        for item in row:
            if item not in stopWords and item not in punctuation:
                item = item.lower()
                word_filter.append(item)
        df['clean_sent'][index] = " ".join(word_filter)
        token_list.append(word_filter)
    df['tokens'] = token_list
    return df

if __name__ == '__main__':
    df = read()
    print(df['clean_sent'][0])
    print(df['tokens'])

    sentence = list()
    for index, row in enumerate(df['tokens']):
        sentence.append(row)

    # train word2vec on the two sentences
    model = gensim.models.Word2Vec(sentence, min_count=1, size=50, window=5, workers=4)
    print('the embedding for 5g:',model.wv['5g'])
    model.save("./embedding_model_5g")
