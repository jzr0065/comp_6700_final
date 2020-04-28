import numpy as np
import json
from nltk.tokenize import sent_tokenize, word_tokenize
import pandas as pd
from nltk.corpus import stopwords


def read_data(filename):


    f = open(filename)
    data = json.load(f)

    question_list = list()
    answers_list = list()
    context_list = list()

    for i in range(100):
        for j in range(len(data['data'][i]['paragraphs'])):
            context = sent_tokenize(data['data'][i]['paragraphs'][j]['context'])[0]
            context_list.append(sent_tokenize(context))
            for k in range(len(data['data'][i]['paragraphs'][j]['qas'])):
                if len(data['data'][i]['paragraphs'][j]['qas'][k]['answers']) == 0:
                    continue
                q = data['data'][i]['paragraphs'][j]['qas'][k]['question']
                a = data['data'][i]['paragraphs'][j]['qas'][k]['answers'][0]['text']
                answers_list.append(a)
                question_list.append(q)

    print(len(answers_list))
    print(len(question_list))
    print(len(context_list))

    print(answers_list[0])
    print(question_list[0])
    print(context_list[0])

    df_context = pd.DataFrame(context_list, columns=['sent_context'])
    df_questions = pd.DataFrame(question_list, columns=['sent_questions'])
    df_answers = pd.DataFrame(answers_list, columns=['sent_answers'])
    df_context['clean_sent_context'] = df_context['sent_context'].apply(word_tokenize)
    df_questions['clean_sent_questions'] = df_questions['sent_questions'].apply(word_tokenize)
    df_answers['clean_sent_answers'] = df_answers['sent_answers'].apply(word_tokenize)

    stopWords = set(stopwords.words('english'))
    punctuation = [',', '.', '?', '(', ')']

    token_list = list()
    # remove the stop words and punctuations
    for index, row in enumerate(df_context['clean_sent_context']):
        word_filter = list()
        for item in row:
            # if item not in stopWords and item not in punctuation:
            if item not in punctuation:
                item = item.lower()
                word_filter.append(item)
        df_context['clean_sent_context'][index] = " ".join(word_filter)
        token_list.append(word_filter)
    df_context['tokens_context'] = token_list

    token_list = list()
    for index, row in enumerate(df_questions['clean_sent_questions']):
        word_filter = list()
        for item in row:
            # if item not in stopWords and item not in punctuation:
            if item not in punctuation:
                item = item.lower()
                word_filter.append(item)
        df_questions['clean_sent_questions'][index] = " ".join(word_filter)
        token_list.append(word_filter)
    df_questions['tokens_questions'] = token_list

    token_list = list()
    for index, row in enumerate(df_answers['clean_sent_answers']):
        word_filter = list()
        for item in row:
            # if item not in stopWords and item not in punctuation:
            if item not in punctuation:
                item = item.lower()
                word_filter.append(item)
        df_answers['clean_sent_answers'][index] = " ".join(word_filter)
        token_list.append(word_filter)
    df_answers['tokens_answers'] = token_list


    return df_context, df_questions, df_answers

    # print(len(data['data']))
    # print(data['data'][1]['paragraphs'][1]['context'])
    # print(data['data'][0]['paragraphs'][0]['qas'])



df_context, df_questions, df_answers = read_data('./train-v2.0.json')
print(df_answers['tokens_answers'])
print(df_context['tokens_context'])
print(df_questions['tokens_questions'])
df_context.to_pickle('squad_context.pkl')
df_questions.to_pickle('squad_question.pkl')
df_answers.to_pickle('squad_answer.pkl')


