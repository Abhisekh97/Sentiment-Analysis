from sentiment_analysis.pipeline.stage_01_data_ingestion import DataIngesstionTrainingPipeline
from sentiment_analysis.pipeline.stage_02_prepare_base_model import PrepareBaseModelTrainingPipeline
from sentiment_analysis.utils.logger import Logger

logger = Logger.__call__().get_logger()
STAGE_NAME = "Data ingestion Training Pipline"
try:
    obj = DataIngesstionTrainingPipeline()
    obj.main()
    logger.info("DATA ingestion pipeline finished")
except Exception as e:
    logger.error(str(e))

STAGE_NAME = "Data ingestion Training Pipline"
try:
    logger.info(f"*******************")
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    obj = PrepareBaseModelTrainingPipeline()
    obj.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e



