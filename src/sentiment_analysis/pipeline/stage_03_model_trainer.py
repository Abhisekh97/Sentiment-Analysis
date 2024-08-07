

from sentiment_analysis.components.model_trainer import Training
from sentiment_analysis.config.configuration import ConfigurationManager
from sentiment_analysis.utils.logger import Logger
logger = Logger.__call__().get_logger()


class ModelTrainerPipeline:
    def __init__(self) -> None:
        
        pass

    def main(self):
        config = ConfigurationManager()
        training_config = config.get_training_config()
        training = Training(config=training_config)
        training.get_base_model()
        # training.train_valid_generator()
        training.training_data_preparation()
        training.train()
        


STAGE_NAME= "Training Pipeline"

if __name__ == '__main__':
    try:
        logger.info(f"*******************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelTrainerPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
        



