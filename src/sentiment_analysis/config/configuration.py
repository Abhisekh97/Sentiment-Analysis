from sentiment_analysis.constants import *
from sentiment_analysis.utils.common import read_yaml, create_directories
from sentiment_analysis.entity.configuration_entity import DataIngestionConfig
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
