import math
import pandas as pd
import numpy as np
import polars as pl
from polars import col as c
from polars import lit as l
from numpy.random import seed
from tensorflow.random import set_seed as tf_set_seed
from plotnine import *

import tensorflow as tf
from tensorflow.keras import layers, Model, backend as K
from tensorflow.keras import models

seed(1000);
tf_set_seed(1000);

def ohe_columns(df, columns):
    for column in columns:
        unique_values = list(set(df[column]));
        for item in unique_values:
            if item == None:
                item = "None";
            df = df.with_columns(((pl.col(column)==item)*1.0).alias(column+'_'+item));
    return df.drop(columns);

def keyfun_group_count(s):
    if s.split(" ")[0] == "Other":
        return -1;
    return int(s.split("(")[1].split(")")[0])

def order_category_column(pdf, colname, keyfun=lambda x,y: x < y):
    
    labels = sorted(list(set(pdf[colname])),key=keyfun,reverse=True);
    cc = pd.Categorical(values=pdf[colname],
                        categories=labels,
                        ordered=True)
    pdf[colname] = cc;
    
    return pdf;

def count_and_auto_other(df, colname, new_column_name, threshold):
    counts = (df
              .group_by(colname)
              .count()
              .with_columns(pl
                            .when(c('count')<threshold)
                            .then(pl.lit('Other'))
                            .otherwise(c(colname))
                            .alias(new_column_name)))
    return (counts
            .group_by(new_column_name)
            .agg(pl.col("count")
                 .sum())
            .join(counts.drop("count"), on=new_column_name, how="inner")
            .with_columns((pl.col(new_column_name)+
                           pl.lit(" (")+
                           pl.col("count").cast(str)+
                           pl.lit(")")).alias(new_column_name)));


add_label_order = lambda df: order_category_column(df,
                                                   'Gender, Race, Ethnicity (Count)',
                                                   keyfun=keyfun_group_count);


meta_data = (pl.read_csv("derived_data/meta-data.csv")
                            .filter(pl.col("domain")=="DM")
                            .filter(pl.col("archive")=="false")
                            .filter(pl.col("duplicate")=="false"));



demographics = pl.concat([pl.read_csv(file).select(['AGE',
                                                    'BRTHDTC',
                                                    'DOMAIN',
                                                    'ETHNIC',
                                                    'RACE',
                                                    'RACEMULT',
                                                    'RFPENDTC',
                                                    'RFSTDTC',
                                                    'SEX',
                                                    'STUDYID',
                                                    'USUBJID']) for file in meta_data["file"]]);

string_columns = [col for col, dtype in demographics.schema.items() if dtype == pl.Utf8]

demographics = demographics.with_columns([
        pl.col(col).fill_null("Not Reported") for col in string_columns
    ]).filter(pl.col('AGE').is_not_nan())


to_encode = demographics.select(['SEX','ETHNIC','RACE']).unique();



demo_ohe = ohe_columns(demographics, ['SEX','ETHNIC','RACE']);
for_tags = demographics.with_columns((pl.col('SEX')+", "+pl.col('RACE')+", "+pl.col('ETHNIC')).alias('tag')).select(['USUBJID','tag']);
tag_counts = for_tags.group_by("tag").agg([pl.count()]).to_pandas().sort_values('count',ascending=False);
for_tags = (for_tags
            .join(pl.from_pandas(tag_counts),on="tag",how="inner")
            .with_columns(pl.when(pl.col('count')>=15).then(pl.col('tag')).otherwise(pl.lit('Other')).alias('Gender, Race, Ethnicity')));

other_count = for_tags.filter(pl.col('Gender, Race, Ethnicity')=="Other").shape[0];
for_tags = (for_tags
            .with_columns((pl.col('Gender, Race, Ethnicity') + pl.lit(' (') +
                          pl.when(pl.col('Gender, Race, Ethnicity')=="Other").then(pl.lit(other_count)).otherwise(pl.col('count').cast(str)) +
                          pl.lit(')')).alias('Gender, Race, Ethnicity (Count)')));
 
tag_cat = pd.Categorical(values=tag_counts['tag'], categories=tag_counts['tag'], ordered=True);

demo_prepped = demo_ohe.with_columns((pl.col('AGE')*1.0).alias('AGE')).select(['USUBJID',
                                                                               'STUDYID',
                                                                               'AGE',
                                                                             'SEX_Intersex',
                                                                             'SEX_Female',
                                                                             'SEX_Male',
                                                                             'ETHNIC_Hispanic or Latino',
                                                                             'ETHNIC_Not reported',
                                                                             'ETHNIC_Not Hispanic or Latino',
                                                                             'RACE_White',
                                                                             'RACE_Unknown',
                                                                             'RACE_Asian',
                                                                             'RACE_Native Hawaiian or Pacific Islander',
                                                                             'RACE_Black or African American',
                                                                             'RACE_Multiple',
                                                                             'RACE_American Indian or Alaska Native',
                                                                             'RACE_Not reported']);
min_age = demo_prepped["AGE"].min();
max_age = demo_prepped["AGE"].max();
demo_prepped = (demo_prepped                
                .with_columns(((pl.col('AGE')-min_age)/(max_age-min_age)).alias('AGE')))
classes = demo_prepped.drop("AGE").unique().with_row_count(name="class");

import keras
from keras import layers
from keras import ops
import tensorflow as tf


def build_variational_autoencoder(input_dim, intermediate_layers, intermediate_size, latent_size):
    """
    Constructs a Variational Autoencoder (VAE) for one-hot encoded inputs.

    Parameters:
        input_dim (int): Size of the input data (number of features).
        intermediate_layers (int): Number of intermediate layers.
        intermediate_size (int): Size of each intermediate layer.
        latent_size (int): Size of the latent space.

    Returns:
        vae: The constructed Variational AutoEncoder model.
        encoder: The encoder part of the VAE.
        decoder: The decoder part of the VAE.
    """
    class Sampling(layers.Layer):
        """Sampling layer using the reparameterization trick."""

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.seed_generator = keras.random.SeedGenerator(1337)

        def call(self, inputs):
            z_mean, z_log_var = inputs
            batch = ops.shape(z_mean)[0]
            dim = ops.shape(z_mean)[1]
            epsilon = keras.random.normal(shape=(batch, dim), seed=self.seed_generator)
            return z_mean + ops.exp(0.5 * z_log_var + 1e-7) * epsilon

    # Build encoder
    encoder_inputs = keras.Input(shape=(input_dim,), name="encoder_input")
    x = encoder_inputs
    for _ in range(intermediate_layers):
        x = layers.Dense(intermediate_size, activation="relu")(x)
    z_mean = layers.Dense(latent_size, name="z_mean")(x)
    z_log_var = layers.Dense(latent_size, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

    # Build decoder
    latent_inputs = keras.Input(shape=(latent_size,), name="latent_input")
    x = latent_inputs
    for _ in range(intermediate_layers):
        x = layers.Dense(intermediate_size, activation="relu")(x)
    decoder_outputs = layers.Dense(input_dim, activation="softmax")(x)
    decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")

    # Build VAE
    class VAE(keras.Model):
        def __init__(self, encoder, decoder, **kwargs):
            super().__init__(**kwargs)
            self.encoder = encoder
            self.decoder = decoder
            self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
            self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
            self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

        @property
        def metrics(self):
            return [
                self.total_loss_tracker,
                self.reconstruction_loss_tracker,
                self.kl_loss_tracker,
            ]

        def train_step(self, data):
            with tf.GradientTape() as tape:
                z_mean, z_log_var, z = self.encoder(data)
                reconstruction = self.decoder(z)
                reconstruction_loss = ops.mean(ops.square(data - reconstruction));
                kl_loss = -0.5 * ops.sum(1 + z_log_var - ops.square(z_mean) - ops.exp(z_log_var + 1e-7), axis=-1)
                total_loss = ops.mean(reconstruction_loss + 0.0001* kl_loss)
            grads = tape.gradient(total_loss, self.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
            self.total_loss_tracker.update_state(total_loss)
            self.reconstruction_loss_tracker.update_state(ops.mean(reconstruction_loss))
            self.kl_loss_tracker.update_state(ops.mean(kl_loss))
            return {
                "loss": self.total_loss_tracker.result(),
                "reconstruction_loss": self.reconstruction_loss_tracker.result(),
                "kl_loss": self.kl_loss_tracker.result(),
            }

    vae = VAE(encoder, decoder)
    vae.compile(optimizer=keras.optimizers.Adam(clipvalue=1.0))
    return vae, encoder, decoder 


ae, enc, dec = build_variational_autoencoder(demo_prepped.width - 2 , 2, 11, 2);

demo_prepped.write_csv("derived_data/demo-encoded.csv");

ae.fit(demo_prepped.drop(['USUBJID','STUDYID']).drop_nulls().to_numpy(),
       batch_size=10, epochs=100);


age_group = (demographics
             .select(['USUBJID','AGE'])
             .with_columns(pl.when(pl.col('AGE').is_between(18,29))
                           .then(pl.lit('18-29'))
                           .when(pl.col('AGE').is_between(30,49))
                           .then(pl.lit('30-49'))
                           .when(pl.col('AGE').is_between(50,69))
                           .then(pl.lit('50-69'))
                           .when(pl.col('AGE')>=70)
                           .then(pl.lit('70 or above')).otherwise(pl.lit("Missing"))
                           .alias('Age Group')))


projection = (pd.DataFrame(enc.predict(demo_prepped.drop(['USUBJID','STUDYID']).to_numpy())[0], columns=["E1","E2"])
              .eval("USUBJID=@demo_prepped['USUBJID']")
              .eval("STUDYID=@demo_prepped['STUDYID']"));
studyid_labels = count_and_auto_other(demographics, 'STUDYID', 'STUDYID (Count)',-1)
projection = (pl
              .from_pandas(projection)
              .join(age_group, on="USUBJID", how="inner")
              .join(for_tags, on="USUBJID", how="inner")
              .join(demographics.select("USUBJID","SEX","RACE","ETHNIC"), on="USUBJID", how="inner")
              .join(studyid_labels,on="STUDYID", how="inner"));

projection = projection.with_columns([c('E1')+np.random.normal(0,0.35,projection.shape[0]),c('E2')+np.random.normal(0,0.35,projection.shape[0])]).to_pandas()

projection = order_category_column(projection,
                                   'Gender, Race, Ethnicity (Count)',
                                   keyfun_group_count);
projection = order_category_column(projection,
                                   'STUDYID (Count)',
                                   keyfun_group_count);



projection = projection.rename(columns={"SEX":"GENDER"});
projection['RACE, ETHNICITY'] = projection["RACE"] + ', ' + projection['ETHNIC']

# Factorize the 'RACE, ETHNICITY' column by count
ordered_levels = projection['RACE, ETHNICITY'].value_counts().index.tolist()

projection['RACE, ETHNICITY'] = pd.Categorical(projection['RACE, ETHNICITY'], categories=ordered_levels, ordered=True)

projection.to_csv("derived_data/demo_vae_projection.csv");

