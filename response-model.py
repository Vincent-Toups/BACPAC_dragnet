import math
import pandas as pd
import numpy as np
import polars as pl
from polars import col as c
from polars import lit as l
import keras
from keras import layers, backend
from numpy.random import seed
from tensorflow.random import set_seed as tf_set_seed
from plotnine import *

seed(1000);
tf_set_seed(1000);

data = pl.read_csv("derived_data/ex-wide-gbm-encoded.csv");

input_columns = ['Mindfulness or meditation or relaxation', 'Exercise', 'NSAIDs', 'Opioids', 'Diet or weight loss program', 'Non-spinal fusion', 'Therapy or counseling', 'SSRI_SNRI', 'Acupuncture', 'Spinal fusion', 'Gabapentin or pregabalin', 'Tricyclic antidepressants'];

def build_network(input_shape=12,
                  inner_layer_count=3,
                  inner_layer_size=64,
                  output_shape=1):
    input = keras.Input(shape=(input_shape,));
    e = layers.Dropout(0.1, input_shape=(input_shape,))(input);
    for i in range(inner_layer_count):
        e = layers.Dense(inner_layer_size, activation="relu")(e);
    e = layers.Dense(output_shape,activation='linear')(e);
    model = keras.Model(input, e);
    model.compile(optimizer='adam', loss='mean_absolute_error');
    return model;

data_tt = data.with_columns(pl.Series(np.random.uniform(0,1,data.shape[0])<0.5).alias('train'))

train = data_tt.filter(c('train')==True).drop('train');
test = data_tt.filter(c('train')==False).drop('train');

m = build_network();

m.fit(train.select(input_columns).to_numpy(),
      train.select('Change').to_numpy(),
      batch_size=100, epochs=50000);
predicted = m.predict(test.select(input_columns).to_numpy());

test = test.with_columns(pl.Series(predicted.reshape(predicted.shape[0])).alias('predicted Change'))
(ggplot(test.to_pandas(), aes('Change','predicted Change'))
 + geom_point()).save("figures/nn_response_mode.png");
