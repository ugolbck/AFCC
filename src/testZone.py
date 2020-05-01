from pycorenlp import StanfordCoreNLP
import pandas as pd
import os, re
import json

from words import contraction_mapping, discourse_markers, modals, softeners

nlp_wrapper = StanfordCoreNLP('http://localhost:9000')


data = {
    'text': [['hi', 'though', 'thought', 'tho', 'might', 'Ugo'], ['as', 'a', 'result', 'is', 'quite', 'high'], ['as', 'if', 'rarely', 'bla']],
    'tag': [1, 10, 20]
}

df = pd.DataFrame(data)


def discourse_features(data, text_column):
    discourse, modal, soft = [], [], []
    for sent in data[text_column]:
        cnt_discourse, cnt_modal, cnt_soft = 0, 0, 0
        if isinstance(sent, str):
            sent = sent.split(' ')

        for i in range(len(sent)):
            # Match individual word in lists
            if sent[i] in discourse_markers:
                cnt_discourse += 1
            if sent[i] in modals:
                cnt_modal += 1
            if sent[i] in softeners:
                cnt_soft += 1

            # Match several word in lists
            if i > 0:
                if sent[i-1] + ' ' + sent[i] in discourse_markers:
                    cnt_discourse += 1
            if i > 1:
                if sent[i-2] + ' ' + sent[i-1] + ' ' + sent[i] in discourse_markers:
                    cnt_discourse += 1

        discourse.append(cnt_discourse)
        modal.append(cnt_modal)
        soft.append(cnt_soft)
    data['num_discourse'] = discourse
    data['num_modals'] = modal
    data['num_softeneers'] = soft


discourse_features(df, 'text')

print(df.head())