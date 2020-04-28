import numpy as np
from typing import Iterable, List
from gensim.models.keyedvectors import BaseKeyedVectors
from sklearn.decomposition import PCA
from embedding_training import read
import gensim
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity

def TF_build(token_list):
    """
    :param token_list: 2-d list
    :return: TF dict
    """
    voc_dic = dict()
    for row in token_list:
        for word in row:
            if word not in voc_dic:
                voc_dic[word] = 0
            voc_dic[word] += 1
    # print(voc_dic)
    factor = 1.0 / sum(voc_dic.values())
    for k in voc_dic:
        voc_dic[k] = voc_dic[k] * factor

    return voc_dic

def build_embedding(embedding_name, token_list, voc_dict, alpha= 1e-3):
    # load the embedding model
    model = gensim.models.word2vec.Word2Vec.load(embedding_name)
    length = len(token_list)
    sentence_emb = np.zeros((length, 50))
    for index, row in enumerate(token_list):
        for word in row:
            sentence_emb[index] += model.wv[word] * (alpha / (alpha + voc_dict[word]))

    # calculate the PCA component
    pca = PCA(n_components= 50)
    pca.fit(sentence_emb)
    u = pca.components_[0]  # the PCA vector
    u = np.multiply(u, np.transpose(u))  # u x uT

    for index, row in enumerate(sentence_emb):
        sub = np.multiply(u, row)
        sentence_emb[index] = row - sub

    return sentence_emb

def input_tokenlize(input):
    user_response = word_tokenize(input)
    word_filter = list()
    stopWords = set(stopwords.words('english'))
    punctuation = [',', '.', '?']
    for word in user_response:
        if word not in stopWords and word not in punctuation:
            word_filter.append(word)
    return word_filter

def chatBox(input, sentence, voc_dict, df):

    input = input_tokenlize(input)
    sentence.append(input)
    # print(input)
    input_embedding = build_embedding('embedding_model_5g', sentence, voc_dict)
    scores = cosine_similarity(input_embedding[:-1], np.reshape(input_embedding[-1], (1, -1)))
    index = np.argmax(scores)
    return df['sent'][index]