import numpy as np
import pickle as pk
import os
import nltk
import string
from nltk import word_tokenize
from sentiment_analysis.utils.logger import Logger

logger = Logger.__call__().get_logger()


class PredictionPipeline:
    def __init__(self):
        pass

    def text_preprocess(self, text):
        lower_casing = text.lower()
        tokens = word_tokenize(lower_casing)
        tokens = [self.stemmer.stem(token) for token in tokens if token not in self.stopwords and token not in string.punctuation]
        return " ".join(tokens)

    def predict(self, text):
        ## load model
        
        # model = load_model(os.path.join("artifacts","training", "model.h5"))
        self.stopwords = pk.load(open(os.path.join('model', 'english_stop_words.pkl'), 'rb'))
        self.stemmer = pk.load(open(os.path.join('model', 'stemmer.pkl'), 'rb'))
        self.model = pk.load(open(os.path.join('model', 'logistic_regression_model.pkl'), 'rb'))
        self.vectorizer = pk.load(open(os.path.join('model', 'vectorizer.pkl'), 'rb'))
        cleaned_text = self.text_preprocess(text)
        vector = self.vectorizer.transform([cleaned_text])
        y_pred = self.model.predict(vector)


        if y_pred == 1:
            prediction = 'Positive'
            logger.info(f"inference {prediction}")
            return prediction
        else:
            prediction = 'Negative'
            logger.info(f"inference {prediction}")
            return prediction
