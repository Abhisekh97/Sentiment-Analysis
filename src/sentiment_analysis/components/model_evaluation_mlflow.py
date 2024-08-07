from sentiment_analysis.utils.common import save_json
from sentiment_analysis.utils.logger import Logger
import tensorflow as tf
from pathlib import Path
import mlflow
import mlflow.keras
import mlflow.sklearn
from urllib.parse import urlparse
import nltk
from nltk import word_tokenize
import pandas as pd
import string
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
import pickle as pk
import os

from sentiment_analysis.config.configuration import EvaluationConfig

logger = Logger.__call__().get_logger()

class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config

    
    # def _valid_generator(self):

    #     datagenerator_kwargs = dict(
    #         rescale = 1./255,
    #         validation_split=0.30
    #     )

    #     dataflow_kwargs = dict(
    #         target_size=self.config.params_image_size[:-1],
    #         batch_size=self.config.params_batch_size,
    #         interpolation="bilinear"
    #     )

    #     valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
    #         **datagenerator_kwargs
    #     )

    #     self.valid_generator = valid_datagenerator.flow_from_directory(
    #         directory=self.config.training_data,
    #         subset="validation",
    #         shuffle=False,
    #         **dataflow_kwargs
    #     )


    # @staticmethod
    # def load_model(path: Path) -> tf.keras.Model:
    #     return tf.keras.models.load_model(path)
    def load_model(self):
        self.model = pk.load(open(self.config.path_of_model, 'rb'))
        self.stopwords = pk.load(open(self.config.path_of_stopwords, 'rb'))
        self.stemmer = pk.load(open(self.config.path_of_stemmer, 'rb'))
        self.vectorizer = pk.load(open(self.config.path_of_vectorizer, 'rb'))

    def data_preprocess(self):
        df = pd.read_csv(self.config.training_data)
        df = df[['Text', 'Score']]
        df = df.sample(n=self.config.params_data_size)
        df = df.loc[df['Score']!=3]
        df = df.loc[df['Score']!=4]
        def category(score):
            return 0 if score==1 or score==2 else 1
        df['Sentiment']= df['Score'].apply(category)
        def text_preprocessing(text):
            lower_casing = text.lower()
            tokens = word_tokenize(lower_casing)
            tokens = [self.stemmer.stem(token) for token in tokens if token not in self.stopwords and token not in string.punctuation]
            return " ".join(tokens)
        
        df['Text'] = df['Text'].apply(text_preprocessing)
        self.y_test = np.array(df['Sentiment'])
        self.df = self.vectorizer.transform(df['Text'])
    
    def evaluation(self):
        # self.model = self.load_model(self.config.path_of_model)
        self.load_model()
        # self._valid_generator()
        self.data_preprocess()
        self.y_pred = self.model.predict(self.df)
        self.score = accuracy_score(self.y_test, self.y_pred)
        self.save_score()

    def save_score(self):
        scores = { "accuracy": self.score}
        save_json(path=Path("scores.json"), data=scores)

    
    def log_into_mlflow(self):
        # mlflow.set_registry_uri(self.config.mlflow_uri)
        mlflow.set_tracking_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        
        with mlflow.start_run():
            mlflow.log_params(self.config.all_params)
            mlflow.log_metrics(
                { "accuracy": self.score}
            )
            # Model registry does not work with file store
            if tracking_url_type_store != "file":

            # Register the model
            # There are other ways to use the Model Registry, which depends on the use case,
            # please refer to the doc for more information:
            # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                mlflow.sklearn.log_model(self.model, "model", registered_model_name="logisticRegression")
                logger.info("MLFLOW Remote logging completed ")
            else:
                mlflow.sklearn.log_model(self.model, "model")
                logger.info("MLFLOW local logging completed")
