from sentiment_analysis.pipeline.stage_01_data_ingestion import DataIngesstionTrainingPipeline
from sentiment_analysis.utils.logger import Logger

logger = Logger.__call__().get_logger()

try:
    obj = DataIngesstionTrainingPipeline()
    obj.main()
    logger.info("DATA ingestion pipeline finished")
except Exception as e:
    logger.error(str(e))