import string
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


class paragraph_cleaner(object):
    def __init__(self, paragraph):
        self.paragraph = paragraph

    def remove_punctuation(self, text):
        for punctuation in string.punctuation:
            text = text.replace(punctuation, ' ')
        return text

    def lowercase(self, text):
        text = text.lower()
        return text

    def remove_numbers(self, text):
        text = ''.join([i for i in text if not i.isdigit()])
        return text

    def remove_stopwords(self, text, language='english'):
        stop_words = set(stopwords.words(language))
        tokenized = word_tokenize(text)
        text = [
            word for word in tokenized if not word in stop_words]
        return text

    def lemmatize(self, text):
        lemmatizer = WordNetLemmatizer()  # Instantiate lemmatizer
        text = [lemmatizer.lemmatize(word) for word in text]  # Lemmatize
        text = " ".join(text)
        return text
