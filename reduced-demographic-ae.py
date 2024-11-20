import keras
from keras import layers, backend
import pandas as pd
import numpy as np
from dfply import *
from plotnine import *
from sklearn.cluster import SpectralClustering
from numpy.random import seed
from tensorflow.random import set_seed as tf_set_seed
import util as u
import pdb;
from pandasql import sqldf
import json

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


sql = lambda q: sqldf(q, globals())

def to_json(obj, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=4, cls=NpEncoder)
 
seed(1000);
tf_set_seed(1000);

def boolean_perturber(p_flip, p_remove, censored=-1):
    def do_it(column):
        columnc = np.array(column);
        ii = np.where(np.random.uniform(0,1,columnc.size) < p_flip);
        print(columnc.size)
        v = np.logical_not(columnc[ii])*1;
        columnc[ii] = v;
        ii = np.where(np.random.uniform(0,1,columnc.size) < p_remove);
        columnc[ii] = censored;
        return columnc;
    return do_it;

def float_perturber(s, p_remove, censored=-1):
    def do_it(column):
        cc = column.copy() + np.random.standard_normal(size=column.size)*s;
        cc[np.random.uniform(0,1,size=column.size)<p_remove] = -1;
        return cc;
    return do_it;

def perturb_columns(df, transformers):
    df = df.copy();
    for k in list(transformers.keys()):
        df[k] = transformers[k](df[k]);
    return df;

def norm_columns(df, columns):
    df = df.copy();
    out = {};
    for column in columns:
        c = df[column];
        mn = c.min();
        mx = c.max();
        d = mx-mn;
        out[column] = {"min":mn,"max":mx};
        df[column] = (c-mn)/d;
    return (df,out);

data = pd.read_csv("source_data/demographics.csv");
reduced_data = sql('''
select 
 id,
 case `group` 
  when 1 then 'prt' 
  when 2 then 'saline' 
  when 3 then 'soc' end
  as treatment,
 case when hispanic = 1 then 'hispanic' 
      when ethnicity = 4 then 'white' 
      when ethnicity = 5 then 'black'  
      else 'other' end as ethnicity,
 married_or_living_as_marri as married,
 age,
 weight,
 case when gender = 1 then 'male' when gender = 2 then 'female' else 'other' end as gender,
 backpain_length 
from data''');
one_hot_data = u.one_hot_encode(reduced_data, "treatment ethnicity gender".split(" ")).drop('backpain_length',axis=1);
(one_hot_data,norm_info) = norm_columns(one_hot_data,["age","weight"]);

target = one_hot_data.sample(n=10000, replace=True).reset_index(drop=True);
inputs = perturb_columns(target, {
    "married":boolean_perturber(0.1, 0.1),
    "age":float_perturber(0.1, 0.1),
    "weight":float_perturber(0.1,0.1),
    "treatment_saline":boolean_perturber(0.1, 0.1),
    "treatment_soc":boolean_perturber(0.1, 0.1),
    "treatment_prt":boolean_perturber(0.1, 0.1),
    "ethnicity_black":boolean_perturber(0.1,0.1),
    "ethnicity_white":boolean_perturber(0.1,0.1),
    "ethnicity_other":boolean_perturber(0.1,0.1),
    "ethnicity_hispanic":boolean_perturber(0.1,0.1),
    "gender_female":boolean_perturber(0.1,0.1),
    "gender_male":boolean_perturber(0.1,0.1)
});

ae_cols = ['married', 'age', 'weight', 'ethnicity_white', 'ethnicity_other',
           'ethnicity_black', 'ethnicity_hispanic', 'gender_female',
           'gender_male'];
ae_ext_layer_size = len(ae_cols);

def build_vae(n_input=ae_ext_layer_size,
              n_intermediate=2,
              encoded_dimension=2,
              intermediate_size=6):

    input = keras.Input(shape=(n_input,));
    #e = layers.Dropout(0.1, input_shape=(n_input,))(input);
    #e = layers.GaussianNoise(0.05)(e);
    e = layers.Dense(intermediate_size, activation='relu')(input);
    for i in range(n_intermediate-1):
        e = layers.Dense(intermediate_size, activation='relu')(e);

    mu_layer = layers.Dense(encoded_dimension, name="encoder_mu")(e);
    log_var_layer = layers.Dense(encoded_dimension, name="encoder_log_var")(e);

    def sampler(mu_log_var):
        mu, log_var = mu_log_var;
        eps = keras.backend.random_normal(keras.backend.shape(mu), mean=0.0, stddev=1.0)
        sample = mu + backend.exp(log_var/2) * eps
        return sample

    encoder_output = layers.Lambda(sampler, name="encoder_output")([mu_layer, log_var_layer])

    d = layers.Dense(intermediate_size, activation='relu')(encoder_output);
    for i in range(n_intermediate-1):
        d = layers.Dense(intermediate_size, activation='relu')(d);

    d = layers.Dense(n_input, activation='linear')(d);

    ae = keras.Model(input, d);
    encoder = keras.Model(input, encoder_output);
    ae.compile(optimizer='adam', loss='mean_squared_error');

    return (ae,encoder)

(ae, enc) = build_vae();
ae.fit(inputs[ae_cols], target[ae_cols], epochs=100, batch_size=25, shuffle=True, verbose=2);

one_hot_data.to_csv("derived_data/reduced-demographics-one-hot.csv", index=False);
to_json(norm_info, "derived_data/reduced-demographics-one-hot-norm-info.json");
to_json(ae_cols,"derived_data/reduced-demographics-columns.json");

ae.save("models/reduced-demographics-ae")
enc.save("models/reduced-demographics-enc")



