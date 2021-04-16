from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
config = tf.ConfigProto()
#config = tf.compat.v1.ConfigProto
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))
import os
import numpy as np
import pandas as pd
from bert4keras.backend import keras, set_gelu
from bert4keras.tokenizers import Tokenizer
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import Adam, extend_with_piecewise_linear_lr
from bert4keras.snippets import sequence_padding, DataGenerator
from keras.layers import Lambda, Dense
from src.utils.data_utils import init_dir
from src.models.base_model import BaseModel
from src.utils.metrics_utils import get_model_metrics
from src.utils.data_utils import load_df
from keras.callbacks import EarlyStopping,ModelCheckpoint
set_gelu('tanh')  # change the version of gelu

AdamLR = extend_with_piecewise_linear_lr(Adam, name='AdamLR')


class SinusoidalInitializer:
    def __call__(self, shape, dtype=None):
        return sinusoidal(shape, dtype=dtype)

def sinusoidal(shape, dtype=None):
    """NEZHA use sin-cos pos 
    """
    vocab_size, depth = shape
    embeddings = np.zeros(shape)
    for pos in range(vocab_size):
        for i in range(depth // 2):
            theta = pos / np.power(10000, 2. * i / depth)
            embeddings[pos, 2 * i] = np.sin(theta)
            embeddings[pos, 2 * i + 1] = np.cos(theta)
    return embeddings

custom_objects = {"sinusoidal":SinusoidalInitializer}

class data_generator(DataGenerator):

    def __init__(self, data, tokenizer, max_len, batch_size=32, buffer_size=None):
        self.data = data
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.max_len = max_len
        if hasattr(self.data, '__len__'):
            self.steps = len(self.data) // self.batch_size
            if len(self.data) % self.batch_size != 0:
                self.steps += 1
        else:
            self.steps = None
        self.buffer_size = buffer_size or batch_size * 1000

    def __iter__(self, random=False):
        idxs = list(range(len(self.data)))
        if random:
            np.random.shuffle(idxs)
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for i in idxs:
            text, label = self.data[i]
            token_ids, segment_ids = self.tokenizer.encode(text, max_length=self.max_len)
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append([label])
            if len(batch_token_ids) == self.batch_size or i == idxs[-1]:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []



class Bert4KearsBase(BaseModel):
    def __init__(self, config):
        # config_path,checkpoint_path,dict_path
        '''
        config = {"config_path":,"checkpoint_path":,"save_dir":,"dict_path":}
        '''
        super().__init__(config)
        init_dir(self.save_dir)
        self.tokenizer = Tokenizer(
            self.config['dict_path'], do_lower_case=True)
        self.graph = tf.get_default_graph()
        self.model_name = None
        self.best_weights_path = None
        self.model_path = None

    def optimizer(self):
        AdamLR = extend_with_piecewise_linear_lr(Adam, name='AdamLR')
        _optimizer = AdamLR(lr=1e-5, lr_schedule={
            1000: 1,
            2000: 0.1
        })
        return _optimizer

    def _init_model(self):
        bert = build_transformer_model(
            config_path=self.config['config_path'],
            checkpoint_path=self.config['checkpoint_path'],
            model=self.model_name,
            return_keras_model=False,
        )
        output = Lambda(lambda x: x[:, 0], name='CLS-token')(bert.model.output)
        output = Dense(
            units=self.num_labels,
            activation='softmax',
            kernel_initializer=bert.initializer
        )(output)
        model = keras.models.Model(bert.model.input, output)
        return model

    def _load_data(self,path):
        df = load_df(path)
        D = []
        for text, label in zip(df['text'], df['label']):
            D.append((str(text), int(label)))
        return D

    def process_data(self, train_path, dev_path, test_path):
        train_data = self._load_data(train_path)
        dev_data = self._load_data(dev_path)
        test_data = self._load_data(test_path)

        train_generator = data_generator(
            train_data, self.tokenizer, self.max_len, self.batch_size)
        dev_generator = data_generator(
            dev_data,  self.tokenizer, self.max_len, self.batch_size)
        test_generator = data_generator(
            test_data,  self.tokenizer, self.max_len, self.batch_size)

        return train_generator, dev_generator, test_generator

    def train(self, train_path, dev_path, test_path):
        self.set_seed(self.seed) 
        train_generator, dev_generator, test_generator = self.process_data(
            train_path, dev_path, test_path)
        # load model
        with self.graph.as_default():
            self.model = self._init_model()
            _optimizer = self.optimizer()
            self.model.compile(
                loss='sparse_categorical_crossentropy',
                optimizer=_optimizer,
                metrics=['accuracy'],
            )
            # start train
            early_stopping_monitor = EarlyStopping(patience=self.patience, verbose=1)
            checkpoint = ModelCheckpoint(self.best_weights_path, monitor='val_loss', verbose=1, save_best_only=True, mode='auto', period=1)
            callbacks = [early_stopping_monitor,checkpoint]

            self.model.fit_generator(train_generator.forfit(),
                                steps_per_epoch=len(train_generator),
                                validation_data = dev_generator.forfit(),
                                validation_steps = len(dev_generator),
                                epochs=self.epochs,
                                callbacks=callbacks)

            self.model.load_weights(self.best_weights_path)
            self.model.save(self.model_path)
        model_report = self.evaluate(test_path)
        return model_report

    def load_model(self,model_path):
        self.model = keras.models.load_model(model_path,custom_objects=custom_objects)
        
    def demo(self,text):
        text_list = [text]
        pred_list = self.demo_text_list(text_list)
        pred = pred_list[0]
        return pred

    def demo_text_list(self,text_list):
        batch_token_ids, batch_segment_ids = [], []
        for text in text_list:
            token_ids, segment_ids = self.tokenizer.encode(text, max_length=self.max_len)
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)  
        batch_token_ids = sequence_padding(batch_token_ids)
        batch_segment_ids = sequence_padding(batch_segment_ids)
        with self.graph.as_default():
            preds = self.model.predict([batch_token_ids, batch_segment_ids])
        if self.num_labels==2:
            pred_list = preds[:,1]
        else:
            pred_list = np.argmax(preds, axis=1).flatten()
        return pred_list

    def release(self):
        # K.clear_session()
        del self.model
        del self.graph
        del self.tokenizer
