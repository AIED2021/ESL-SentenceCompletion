import sys
sys.path.append('..')

from src.models import XLNet as MODEL
from src.configs import en_xlnet_large_cased_config as config

import os
base_path = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(base_path,'../data/','/CoLA')
train_path = os.path.join(data_dir, 'train.csv')
dev_path = os.path.join(data_dir, 'dev.csv')
test_path = os.path.join(data_dir, 'test.csv')

config['save_dir'] = os.path.join(base_path,'../tmp/glue/CoLA')
model = MODEL(config)
model_report = model.train(train_path, dev_path, test_path)
print("model result is:{}".format(model_report))
model.load_model(model.model_path)

p = model.predict("hello") 
print(p)

model.release()
