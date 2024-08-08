from flask import Flask, request, jsonify, render_template
import os
from flask_cors import CORS, cross_origin
from sentiment_analysis.utils.common import decodeImage
from sentiment_analysis.pipeline.inference_pipeline import PredictionPipeline
from sentiment_analysis.utils.logger import Logger
logger = Logger.__call__().get_logger()

os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__)
CORS(app)


class ClientApp:
    def __init__(self):
        self.classifier = PredictionPipeline()





@app.route("/", methods=['GET'])
@cross_origin()
def home():
    return render_template('index.html')



@app.route("/train", methods=['GET','POST'])
@cross_origin()
def trainRoute():
    os.system("python main.py")
    # os.system("dvc repro")
    return "Training done successfully!"




@app.route("/inference", methods=['POST'])
@cross_origin()
def predictRoute():
    print(request.json)
    text = request.json['text']
    logger.info(f"input_text for inference{text}")
    result = clApp.classifier.predict(text)
    logger.info(f"the model result { result}")
    return jsonify({'result': result})



if __name__ == "__main__":
    clApp = ClientApp()
    app.run(host='0.0.0.0', port=8080, debug=True) #for AWS

