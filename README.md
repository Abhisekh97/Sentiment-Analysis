# Sentiment-Analysis 
This is an end to end Sentiment Analysis Project . Dataset used in this project is taken from kaggle. Link of the dataset is provided below. Here the dataset size is huge so for demonstration purpose i have only worked with 50000 rows of data randomely sampled from the original dataset. You can Use the entire dataset if you wish. 

# Note:
    Normally Datasets are present  in object storages if you are working in production grade system. But here i am using kaggle data source. Since I don't want to spend money for this project. But if you have data you might want to upload the data into such storage Spaces. Example of such bucket is "Amazone S3" . So Amazone S3 buckets are object storage facility provided by AWS cloud platform. It is popular. But remember here I am not using S3 i am directly downloading the dataset and Working on it. But if you want you can also Use 'Google Drive' and upload the dataset there and work on this project. Below I have provided both the methods. You can follow-

### download from kaggle:
Step 1: Install the required libraries
Here we use a Python library called opendatasets

Let’s install opendatasets. If you wish, you can install any other libraries which you might need like pandas and others

!pip install opendatasets --upgrade --quiet
Step 2: Import the library
Here we import the required libraries; we just need a few to download and view the data sets along with opendatasets
`
import pandas as pd
import os
import opendatasets as od
`
Step 3: Get the data set URL from Kaggle.
Next, we get the Kaggle URL for the specific data set we need to download. We chose the DL Course data set for this post, but you can choose any one of your choices. The total size of the data set that we are downloading is ~586 MB

Step 3: Get Kaggle API token
Before we start downloading the data set, we need the Kaggle API token. To get that

Login into your Kaggle account
Get into your account settings page
Click on Create a new API token
This will prompt you to download the .json file into your system. Save the file, and we will use it in the next step.
Step 4: Download the data set files
Now that we have all the required information let’s start downloading the data sets from Kaggle.
`
# Assign the Kaggle data set URL into variable
dataset = 'https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews/data'
# Using opendatasets let's download the data sets
od.download(dataset)
`
After running the above statements, you will be prompted for the Kaggle username and then the key. This you can get from the .json file which you downloaded earlier, and the file content looks something like this
`
{“username”:”<userID>",”key”:”<userKey>"}
`
After providing the above credentials, the data set files will be downloaded into your working environment (either local or any other platform). If there is no issue in downloading, then the message looks something like this

Please provide your Kaggle credentials to download this dataset. Learn more: http://bit.ly/kaggle-creds Your Kaggle username: <userID> Your Kaggle Key: ········ Downloading dl-course-data.zip to ./dl-course-data

100%|██████████| 231M/231M [00:03<00:00, 64.4MB/s]

For the data set which we are working on, related files will be downloaded into the dir named /dl-course-data(You can get the dir name from the above message). Now that we have all the files in our working environment, let’s list them.

1. Dataset [kaggle link](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews/data)

## Project Workflows

1. Update config.yaml
2. Update secrets.yaml [Optional]
3. Update params.yaml
4. Update the entity
5. Update the configuration manager in src config
6. Update the components
7. Update the pipeline 
8. Update the main.py
9. Update the dvc.yaml




