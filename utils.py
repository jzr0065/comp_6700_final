import gensim
import os
import shutil
import hashlib
from sys import platform
import json


# 计算行数，就是单词数
def getFileLineNums(filename):
    f = open(filename, 'r', encoding='utf8')
    count = 0
    for line in f:
        count += 1
    return count


# Linux或者Windows下打开词向量文件，在开始增加一行
def prepend_line(infile, outfile, line):
    with open(infile, 'r') as old:
        with open(outfile, 'w') as new:
            new.write(str(line) + "\n")
            shutil.copyfileobj(old, new)


def prepend_slow(infile, outfile, line):
    with open(infile, 'r', encoding='utf8') as fin:
        with open(outfile, 'w', encoding='utf8') as fout:
            fout.write(line + "\n")
            for line in fin:
                fout.write(line)


def load_model(filename):
    num_lines = getFileLineNums(filename)
    gensim_file = 'glove_model.txt'
    gensim_first_line = "{} {}".format(num_lines, 100)
    # Prepends the line.
    if platform == "linux" or platform == "linux2":
        prepend_line(filename, gensim_file, gensim_first_line)
    else:
        prepend_slow(filename, gensim_file, gensim_first_line)

    model = gensim.models.KeyedVectors.load_word2vec_format(gensim_file)
    print(model['computer'])
    return model

def load_dataset(filename):
    f = open(filename)

    # returns JSON object as
    # a dictionary
    data = json.load(f)

    # Iterating through the json
    # list
    f.close()
    return data

def evaluation(answer, predict_sentences,k):



    common = 0
    for word in answer[k]:
        if word in predict_sentences:
            common += 1

    precision = float(common) / len(answer)
    recall = float(common) / len(predict_sentences)
    try:
        f_1 = (2 * precision * recall) / (precision + recall)
    except:
        f_1 = 0


    return f_1

if __name__ == '__main__':
    model = load_model('glove.6B.100d.txt')
    # data = load_dataset('train-v2.0.json')
    # print(data['data'][1])