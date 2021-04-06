
# # after geting a model , need to check the predection of it , in other word
# # what is the Success rate of the model
# # load functions from classification module(or other model which depend on the modulr that have choosing)
from pycaret.classification import *

# # after that loading the Model to generate predictions
model = load_model('/home/nadeem/PycharmProjects/FinalProject/MachineLearning')

# # and adding the dataset that we want to run the code over it
import pandas as pd
dataset =pd.read_csv('../MachineLearning/DataSet.csv')

# # last step is predict charges
predictions = predict_model(model, data=dataset)
# print(predictions)
