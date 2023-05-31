from pureml import BasePredictor,Input,Output
import pureml

class Predictor(BasePredictor):
    label = 'nlpexample:model:v3'
    input = Input(type = 'numpy ndarray')
    output = Output(type = 'numpy ndarray')

    def load_models(self):
        self.model = pureml.model.fetch(self.label)

    def predict(self, data):
        prediction = self.model.predict(data)

        return prediction