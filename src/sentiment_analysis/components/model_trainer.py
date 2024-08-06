import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
from pathlib import Path
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from sentiment_analysis.entity.configuration_entity import TrainingConfig
nltk.download('punkt')
import string
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
plt.style.use('ggplot')
nltk.download('stopwords')
print(stopwords.words('english'))
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from sentiment_analysis.utils.logger import Logger
logger = Logger.__call__().get_logger()

class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config

    
    def get_base_model(self):
        # self.model = tf.keras.models.load_model(
        #     self.config.updated_base_model_path
        # )
        self.model = LogisticRegression()

    def train_valid_generator(self):

        # datagenerator_kwargs = dict(
        #     rescale = 1./255,
        #     validation_split=0.20
        # )

        # dataflow_kwargs = dict(
        #     target_size=self.config.params_image_size[:-1],
        #     batch_size=self.config.params_batch_size,
        #     interpolation="bilinear"
        # )

        # valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
        #     **datagenerator_kwargs
        # )

        # self.valid_generator = valid_datagenerator.flow_from_directory(
        #     directory=self.config.training_data,
        #     subset="validation",
        #     shuffle=False,
        #     **dataflow_kwargs
        # )

        # if self.config.params_is_augmentation:
        #     train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
        #         rotation_range=40,
        #         horizontal_flip=True,
        #         width_shift_range=0.2,
        #         height_shift_range=0.2,
        #         shear_range=0.2,
        #         zoom_range=0.2,
        #         **datagenerator_kwargs
        #     )
        # else:
        #     train_datagenerator = valid_datagenerator

        # self.train_generator = train_datagenerator.flow_from_directory(
        #     directory=self.config.training_data,
        #     subset="training",
        #     shuffle=True,
        #     **dataflow_kwargs
        # )
        pass

    def training_data_preparation(self):
        df = pd.read_csv(self.config.training_data)
        df = df[['Text', 'Score']]
        df = df.sample(n=self.config.params_data_size)
        df = df.loc[df['Score']!=3]
        df = df.loc[df['Score']!=4]
        def category(score):
            return 0 if score==1 or score==2 else 1
        df['Sentiment']= df['Score'].apply(category)
        self.stop_words = stopwords.words('english')
        self.stemmer = PorterStemmer()
        def text_preprocessing(text):
            lower_casing = text.lower()
            tokens = word_tokenize(lower_casing)
            tokens = [self.stemmer.stem(token) for token in tokens if token not in self.stop_words and token not in string.punctuation]
            return " ".join(tokens)
        
        df['Text'] = df['Text'].apply(text_preprocessing)
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(df['Text'], df['Sentiment'], test_size=0.2)

        self.vectorizer = TfidfVectorizer()

        self.X_train_vect = self.vectorizer.fit_transform(self.X_train)
        
        self.X_test_vect = self.vectorizer.transform(self.X_test)
    
    # @staticmethod
    # def save_model(path: Path, model: tf.keras.Model):
    #     # model.save(path)
    #     pass
    
    @staticmethod
    def save_artifacts(obj, path: Path):
        with open(path,'wb') as f:
            pickle.dump(obj,f)


    
    def train(self):

        # self.steps_per_epoch = self.train_generator.samples // self.train_generator.batch_size
        # self.validation_steps = self.valid_generator.samples // self.valid_generator.batch_size

        # self.model.fit(
        #     self.train_generator,
        #     epochs=self.config.params_epochs,
        #     steps_per_epoch=self.steps_per_epoch,
        #     validation_steps=self.validation_steps,
        #     validation_data=self.valid_generator
        # )

        # self.save_model(
        #     path=self.config.trained_model_path,
        #     model=self.model
        # )

        self.model.fit(self.X_train_vect, self.y_train)
        y_pred = self.model.predict(self.X_test_vect)
        accuracy = accuracy_score(self.y_test, y_pred)
        confusion_metrix = confusion_matrix(self.y_test, y_pred)
        logger.info(f"model accuracy {accuracy}")
        logger.info(f"model confusion matrix{confusion_metrix}")
        self.save_artifacts(self.model, self.config.trained_model_path)
        self.save_artifacts(self.stop_words, self.config.trained_stopwords_path)
        self.save_artifacts(self.stemmer, self.config.trained_stemmer_path)
        self.save_artifacts(self.vectorizer, self.config.trained_vectorizer_path)


