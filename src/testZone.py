from pycorenlp import StanfordCoreNLP
import pandas as pd
import os, re
import json
import textstat

from words import contraction_mapping, discourse_markers, modals, softeners

nlp_wrapper = StanfordCoreNLP('http://localhost:9000')


data = {
    'text': ['his name is Ugo', 'the Eiffel Tower is tall', 'hello Marianna'],
    'tag': [1, 10, 20]
}

df = pd.DataFrame(data)


text = "this review is goihousand and nine edid have made improvementsthis review is goihousand and nine edid have made improvementsthis review is goihousand and nine edid have made improvementsthis review is goihousand and nine edid have made improvementsthis review is goihousand and nine edid have made improvementsthis review is goihousand and nine edid have made improvementsthis review is goihousand and nine edid have made improvements this review is goihousand and nine edid have made improvements this review is goihousand and nine edid have made improvements this review is goihousand and nine edid have made improvements this review is goihousand and nine edid have made improvements this review is goihousand and nine edid have made improvements this review is goihousand and nine edid have made improvements this review is goihousand and nine edid have made improvements this review is goihousand and nine edid have made improvementsthis review is goihousand and nine edid have made improvements this review is goihousand and nine edid have made improvements this review is goihousand and nine edid have made improvements this review is goihousand and nine edid have made improvements this review is goihousand and nine edid have made improvements"

print(textstat.flesch_reading_ease(text))

