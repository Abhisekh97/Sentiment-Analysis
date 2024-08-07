import os
from sentiment_analysis.constants import *
from sentiment_analysis.utils.common import read_yaml, create_directories, save_json
from sentiment_analysis.entity.configuration_entity import DataIngestionConfig, PrepareBaseModelConfig, TrainingConfig, EvaluationConfig

class ConfigurationManager:
    def __init__(self,
                 config_filepath=CONFIG_FILE_PATH,
                 params_filepath=PARAMS_FILE_PATH) -> None:
        self.config = read_yaml(config_filepath)
        # print(self.config)
        self.params = read_yaml(params_filepath)
        create_directories([self.config.artifacts_root])
    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        create_directories([self.config.data_ingestion.root_dir])
        data_ingestion_config = DataIngestionConfig(
            root_dir= self.config.data_ingestion.root_dir,
            source_URL=self.config.data_ingestion.source_URL,
            local_data_file=self.config.data_ingestion.local_data_file,
            unzip_dir=self.config.data_ingestion.unzip_dir
        )
        return data_ingestion_config
    
    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        config = self.config.prepare_base_model
        
        create_directories([config.root_dir])

        prepare_base_model_config = PrepareBaseModelConfig(
            root_dir=Path(config.root_dir),
            base_model_path=Path(config.base_model_path),
            updated_base_model_path=Path(config.updated_base_model_path),
            params_image_size=self.params.IMAGE_SIZE,
            params_learning_rate=self.params.LEARNING_RATE,
            params_include_top=self.params.INCLUDE_TOP,
            params_weights=self.params.WEIGHTS,
            params_classes=self.params.CLASSES
        )

        return prepare_base_model_config
    
    def get_training_config(self) -> TrainingConfig:
        training = self.config.training
        prepare_base_model = self.config.prepare_base_model
        params = self.params
        # training_data = os.path.join(self.config.data_ingestion.unzip_dir, "Chest--Scan-data")
        training_data = os.path.join(self.config.data_ingestion.unzip_dir, "Reviews.csv")
        create_directories([
            Path(training.root_dir)
        ])

        training_config = TrainingConfig(
            root_dir=Path(training.root_dir),
            trained_model_path=Path(training.trained_model_path),
            trained_stopwords_path=Path(training.trained_stop_words_path),
            trained_vectorizer_path = Path(training.trained_vectorizer_path),
            trained_stemmer_path=Path(training.trained_stemmer_path),
            updated_base_model_path=Path(prepare_base_model.updated_base_model_path),
            training_data=Path(training_data),
            params_epochs=params.EPOCHS,
            params_batch_size=params.BATCH_SIZE,
            params_is_augmentation=params.AUGMENTATION,
            params_image_size=params.IMAGE_SIZE,
            params_data_size=params.DATA_SIZE

        )

        return training_config
    
    def get_evaluation_config(self) -> EvaluationConfig:
        eval_config = EvaluationConfig(
            path_of_model=Path(self.config.training.trained_model_path),
            path_of_stemmer=Path(self.config.training.trained_stemmer_path),
            path_of_stopwords=Path(self.config.training.trained_stop_words_path),
            path_of_vectorizer=Path(self.config.training.trained_vectorizer_path),
            training_data=Path(self.config.evaluation.dataset_path),
            mlflow_uri=self.config.evaluation.ml_flow_tracking_url,
            all_params=self.params,
            params_image_size=self.params.IMAGE_SIZE,
            params_batch_size=self.params.BATCH_SIZE,
            params_data_size=self.params.DATA_SIZE
        )
        return eval_config
