import pandas as pd
from bokeh.plotting import figure, output_file, save
from bokeh.models import ColumnDataSource, HoverTool, FactorRange
from bokeh.layouts import gridplot

# Load Data
projection = pd.read_csv("projection.csv")
subject_changes = pd.read_csv("derived_data/subject-changes.csv")

# Merge and Process Data
projection_ex = projection.merge(subject_changes, on="SUBJID", how="inner")
projection_ex.to_csv("derived_data/demographics-with-projection.csv", index=False)

# Plot 2D Projection by Race and Ethnicity
source = ColumnDataSource(projection)

p1 = figure(title="Demographic 2D Projection",
            x_axis_label="E1", y_axis_label="E2",
            tools="pan,zoom_in,zoom_out,reset,save", width=700, height=500)
p1.circle(x="E1", y="E2", source=source, size=10, alpha=0.6,
          color="blue", legend_field="RACE, ETHNICITY")

p1.legend.title = "Race & Ethnicity"
p1.add_tools(HoverTool(tooltips=[("E1", "@E1"), ("E2", "@E2"), ("Race, Ethnicity", "@{RACE, ETHNICITY}")]))

# Plot by Study ID
p2 = figure(title="2D Projection by Study ID",
            x_axis_label="E1", y_axis_label="E2",
            tools="pan,zoom_in,zoom_out,reset,save", width=700, height=500)
p2.circle(x="E1", y="E2", source=source, size=10, alpha=0.6,
          color="green", legend_field="STUDYID (Count)")

p2.legend.title = "Study ID"
p2.add_tools(HoverTool(tooltips=[("E1", "@E1"), ("E2", "@E2"), ("Study ID", "@{STUDYID (Count)}")]))

# Faceted Plot by Study ID
unique_study_ids = projection["STUDYID (Count)"].unique()
plots = []
for study_id in unique_study_ids:
    sub_df = projection[projection["STUDYID (Count)"] == study_id]
    source_sub = ColumnDataSource(sub_df)
    
    p = figure(title=f"Study ID: {study_id}",
               x_axis_label="E1", y_axis_label="E2",
               tools="pan,zoom_in,zoom_out,reset,save", width=350, height=350)
    p.circle(x="E1", y="E2", source=source_sub, size=10, alpha=0.6, color="purple")
    p.add_tools(HoverTool(tooltips=[("E1", "@E1"), ("E2", "@E2"), ("Count", f"{study_id}")]))
    plots.append(p)

# Create a grid layout for faceted plots
grid = gridplot([plots[i:i+2] for i in range(0, len(plots), 2)], sizing_mode="scale_width")

# Save Outputs
output_file("figures/bokeh_projection_plots.html")









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
            return z_mean + ops.exp(0.5 * z_log_var) * epsilon

    # Build encoder
    encoder_inputs = keras.Input(shape=(input_dim,), name="encoder_input")
    x = encoder_inputs
    for _ in range(intermediate_layers):
        x = layers.Dense(intermediate_size, activation="relu")(x)
    z_mean = layers.Dense(latent_size, name="z_mean")(x)    z_log_var = layers.Dense(latent_size, name="z_log_var")(x)
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
                reconstruction_loss = keras.losses.categorical_crossentropy(data, reconstruction)
                kl_loss = -0.5 * ops.sum(1 + z_log_var - ops.square(z_mean) - ops.exp(z_log_var), axis=-1)
                total_loss = ops.mean(reconstruction_loss + kl_loss)
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
    return vae, encoder, decoder 