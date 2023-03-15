import spacy
from spacytextblob.spacytextblob import SpacyTextBlob

nlp = spacy.load('en_core_web_sm')
nlp.add_pipe("spacytextblob")
def sentiment_analysis(input_str):
    input_str = nlp(input_str)
    return input_str._.blob.polarity