import os
import kaggle
#make sure to copy kaggle.json file ~/.kaggle/ folder
kaggle.api.authenticate()
from pathlib import Path
from sentiment_analysis.utils.logger import Logger
from sentiment_analysis.entity.configuration_entity import DataIngestionConfig

logger = Logger.__call__().get_logger()
class DataIngestion:
    def __init__(self, config: DataIngestionConfig) -> None:
        self.config =config
        print(self.config)
    
    def download_file(self) -> str:
        try:
            dataset_url = self.config.source_URL
            os.makedirs(Path(self.config.root_dir), exist_ok=True)
            logger.info(f"Downloading dataset {dataset_url}  into {self.config.root_dir}")
            kaggle.api.dataset_download_files(dataset_url, path=Path(self.config.root_dir), unzip=True)
            logger.info("Dataset downloaded successfully!")

        except Exception as e:
            logger.info(self.config)
            logger.error("Exception occurred while download dataset" + str(e))