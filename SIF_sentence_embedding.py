import numpy as np
from typing import Iterable, List
from gensim.models.keyedvectors import BaseKeyedVectors
from sklearn.decomposition import PCA
from embedding_training import read
import gensim
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from utils import load_model, evaluation

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

def build_embedding(embedding_name, token_list, voc_dict, length_k, alpha= 1e-3, ):
    # load the embedding model
    model = load_model(embedding_name)
    length = len(token_list)
    sentence_emb = np.zeros((length, 100))
    for index, row in enumerate(token_list):
        for word in row:
            try:
                sentence_emb[index] += model.wv[word] * (alpha / (alpha + voc_dict[word]))
            except:
                continue

    # calculate the PCA component
    pca = PCA(n_components= 100)
    pca.fit(sentence_emb[:length_k])
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

if __name__ == '__main__':

    df_context = pd.read_pickle('squad_context.pkl')
    df_questions = pd.read_pickle('squad_question.pkl')
    df_answers = pd.read_pickle('squad_answer.pkl')

    sentence = list()
    for index, row in enumerate(df_context['tokens_context']):
        sentence.append(row)
    length = len(sentence)
    voc_dict = TF_build(sentence)
    # print(voc_dict)
    # print(sum(voc_dict.values()))
    # print(sentence_embedding)
    F1_score = 0

    for i in range(len(df_questions['tokens_questions'])):
        input = df_questions['tokens_questions'][i]
        sentence.append(input)
    # print(input)
    input_embedding = build_embedding('glove.6B.100d.txt', sentence, voc_dict, length)
    print(input_embedding.shape)
    scores = cosine_similarity(input_embedding[length:], input_embedding[:length])
    print(len(scores))
    indice = np.argmax(scores, axis=1)
    answers = list()
    context = list()
    for index, row in enumerate(df_answers['tokens_answers']):
        answers.append(row)
    for index, row in enumerate(df_context['tokens_context']):
        context.append(row)

    indice = list(indice.ravel())
    f_1_sum = 0
    for i in range(len(indice)):
        f_1 = evaluation(answers, context[indice[i]], i)
        f_1_sum += f_1
    print(f_1_sum)
    print(f_1_sum / len(answers))

