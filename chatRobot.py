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

def remove_punctuation(line):
    #line = str(line)
    if line.strip()=='':
        return ''
    rule = re.compile(u"[^a-zA-Z0-9\u4E00-\u9FA5]")
    line = rule.sub('',line)
    return line


def response(user_response, df):
    robo_response = ''
    cut_sent = df.clean_sent.values.tolist()
    cut_sent.append(user_response)
    tfidf = TfidfVectorizer()

    tfidf_vec = tfidf.fit_transform(cut_sent)

    cos_sim = cosine_similarity(tfidf_vec[-1], tfidf_vec)
    idx = cos_sim.argsort()[0][-2]
    flat = cos_sim.flatten()
    flat.sort()
    req_tfidf = flat[-2]

    if (req_tfidf == 0):
        robo_response = robo_response + "sorry, I don't get your meanings"
        return (robo_response)
    else:
        robo_response = robo_response + df.sent.values[idx]
        return (robo_response)

def chatBox(user_input):
    filename = "5G.txt"
    text = codecs.open(filename, "r", "utf-8").read()
    sent_tokens = sent_tokenize(text)

    # print(sent_tokens)
    df = pd.DataFrame(sent_tokens, columns=['sent'])
    df['clean_sent'] = df['sent'].apply(word_tokenize)

    stopWords = set(stopwords.words('english'))
    punctuation = [',','.','?']

    #remove the stop words and punctuations
    for index, row in enumerate(df['clean_sent']):
        word_filter = list()
        for item in row:
            if item not in stopWords and item not in punctuation:
                word_filter.append(item)
        df['clean_sent'][index] = " ".join(word_filter)



    user_response = user_input
    user_response = word_tokenize(user_response)
    word_filter = list()
    for word in user_response:
        if word not in stopWords and word not in punctuation:
            word_filter.append(word)
    user_response = " ".join(word_filter)
    bot_response = response(user_response, df)
    return bot_response







