import sys
sys.path.append('..')

from src.models import ROBERTA
from src.configs import hfl_chinese_roberta_wwm_ext_large_config as config

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

p = model.predict("Hello")
print(p)

model.release()