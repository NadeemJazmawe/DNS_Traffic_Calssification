# first importing the dataset
import pandas as pd
dataset = pd.read_csv('../MachineLearning/DataSet.csv')

# after that we need to choose the model that we want to use:
# import calssification module
from pycaret.classification import *
# init setup
clasific = setup(data=dataset, target = 'type Pcap', silent = True, html = False)
# # compare models
best_model = compare_models()
# # finalize best model
best = finalize_model(best_model)
# # model visualization and interpretation , u can choose one of these:-
plot_model(best)
plot_model(best, 'confusion_matrix')
interpret_model(best)
# # save best model
save_model(best, '/home/nadeem/PycharmProjects/FinalProject/MachineLearning')
# # return the performance metrics df
dataset = pull()
#
# # after we choose the model , now we Deploy Model
# # to generate predictions
# # load functions from classification module
from pycaret.classification import *
# # load model with a variable
model = load_model('/home/nadeem/PycharmProjects/FinalProject/MachineLearning')
# # predict charges
predictions = predict_model(model, data=dataset)
