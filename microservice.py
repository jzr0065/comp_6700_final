import os
from flask import Flask, request
from chat_box_v2 import chatBox
from embedding_training import read
from SIF_sentence_embedding import TF_build
app = Flask(__name__)


# -----------------------------------
#  The following code is invoked when the path portion of the URL matches
#         /othello
#
#  Parameters are passed as a URL query:
#        /othello?parm1=value1&parm2=value2
#
@app.route('/chat')
def server():
    try:
        parms = {}
        for key in request.args:
            parms[key] = str(request.args[key])
        df = read()
        sentence = list()
        for index, row in enumerate(df['tokens']):
            sentence.append(row)
        voc_dict = TF_build(sentence)
        result = chatBox(parms['Q'], sentence, voc_dict, df)
        print(result)
        return result
    except Exception as e:
        return str(e)


# -----------------------------------
port = os.getenv('PORT', '5000')
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(port))