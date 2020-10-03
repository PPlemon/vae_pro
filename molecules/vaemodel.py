from keras import backend as K
from keras import objectives
from keras.models import Model
from keras.layers import Input, Lambda, Dense, Flatten, RepeatVector, TimeDistributed, GRU, Convolution1D, LSTM

class VAE():
    auto_encoder = None
    latent = 196
    length = 192
    def create(self, length=length, latent_size=latent, weights_file=None):
        charset_length = 33
        epsilon_std = 0.01

        def sampling(args):
            z_mean_, z_log_var_ = args
            batch_size = K.shape(z_mean_)[0]
            epsilon = K.random_normal(shape=(batch_size, latent_size), mean=0., stddev = epsilon_std)
            return z_mean_ + K.exp(z_log_var_ / 2) * epsilon

        x = Input(shape=(length, charset_length))
        z_mean, z_log_var = self.build_Encoder(x, latent_size)
        z = Lambda(sampling, output_shape=(latent_size,), name='lambda')([z_mean, z_log_var])

        self.encoder = Model(x, z)

        decoder_input = Input(shape=(latent_size,))

        self.decoder = Model(
            decoder_input,
            self.build_Decoder(
                decoder_input,
                latent_size,
                length,
                charset_length
            )
        )
        x1 = Input(shape=(length, charset_length))
        z_mean, z_log_var = self.build_Encoder(x1, latent_size)

        z1 = Lambda(sampling, output_shape=(latent_size,), name='lambda')([z_mean, z_log_var])

        self.auto_encoder = Model(
            x1,
            self.build_Decoder(
                z1,
                latent_size,
                length,
                charset_length
            )
        )

        def vae_loss(x, x_decoded_mean):
            x = K.flatten(x)
            x_decoded_mean = K.flatten(x_decoded_mean)
            xent_loss = length * objectives.binary_crossentropy(x, x_decoded_mean)
            kl_loss = - 0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
            return xent_loss + kl_loss

        if weights_file:
            self.auto_encoder.load_weights(weights_file)
            self.encoder.load_weights(weights_file, by_name=True)
            self.decoder.load_weights(weights_file, by_name=True)

        self.auto_encoder.compile(optimizer='Adam', loss=vae_loss, metrics=['accuracy'])

    def build_Encoder(self, x, latent_size):
        h = Convolution1D(9, 9, activation = 'relu', name='conv_1')(x)
        h = Convolution1D(9, 9, activation = 'relu', name='conv_2')(h)
        h = Convolution1D(10, 11, activation='relu', name='conv_3')(h)
        h = Flatten(name='flatten_1')(h)
        h = Dense(435, activation='relu', name='dense_2')(h)
        z_mean = Dense(latent_size, name='z_mean', activation='relu')(h)
        z_log_var = Dense(latent_size, name='z_log_var', activation='relu')(h)
        return (z_mean, z_log_var)

    def build_Decoder(self, z, latent_size, length, charset_length):
        h = Dense(latent_size, name='latent_input', activation='relu')(z)
        h = RepeatVector(length, name='repeat_vector')(h)
        # h = LSTM(500, return_sequences=True, name='LSTM_1')(h)
        # h = LSTM(500, return_sequences=True, name='LSTM_2')(h)
        # h = LSTM(500, return_sequences=True, name='LSTM_3')(h)
        h = GRU(488, return_sequences=True, name='gru_1')(h)
        h = GRU(488, return_sequences=True, name='gru_2')(h)
        h = GRU(488, return_sequences=True, name='gru_3')(h)
        return TimeDistributed(Dense(charset_length, activation='softmax'), name='decoded_mean')(h)

    def save(self, filename):
        self.auto_encoder.save_weights(filename)
    
    def load(self, weights_file, latent_size=latent):
        self.create(weights_file=weights_file, latent_size=latent_size)
