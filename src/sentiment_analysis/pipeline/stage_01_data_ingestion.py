

from sentiment_analysis.config.configuration import ConfigurationManager
from sentiment_analysis.components.data_ingestion import DataIngestion
from sentiment_analysis.utils.logger import Logger

logger = Logger.__call__().get_logger()
STAGE_NAME = "Data ingestion Training Pipline"

class DataIngesstionTrainingPipeline:
    def __init__(self) -> None:
        pass
    def main(self):
        config = ConfigurationManager()
        data = config.get_data_ingestion_config()
        obj = DataIngestion(config=data)
        obj.download_file()



if __name__=='__main__':
    try:
        obj = DataIngesstionTrainingPipeline()
        obj.main()
        logger.info("DATA ingestion pipeline finished")
    except Exception as e:
        logger.error(str(e))