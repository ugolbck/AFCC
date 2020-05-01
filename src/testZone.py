from pycorenlp import StanfordCoreNLP
import pandas as pd
import os
import json
from src.utils import *

nlp_wrapper = StanfordCoreNLP('http://localhost:9000')


data = {
    'text': [['hi', 'my', 'name', 'is', 'Ugo'], ['The', 'Eiffel', 'Tower', 'is', 'very', 'high']],
    'tag': [1, 10]
}

df = pd.DataFrame(data)


discourse_features(df, 'text')
