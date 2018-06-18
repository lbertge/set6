from keras.layers import Input, Dense, Lambda
from keras.models import Model, Sequential
from keras import backend as K
from keras import objectives
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt

batch_size = 100
image_len = 784
latent_dim = 10
epochs = 20
NUM_LAYERS = 256
THRESHOLD = 127.
image_res = 28

def batch_sampling(args):
	z_m, z_v = args
	eps = K.random_normal(shape=(batch_size, latent_dim), mean = 0., std = 1.0)
	return z_m + K.exp(z_v / 2.) * eps

def sampling(args):
	z_m, z_v = args
	eps = K.random_normal(shape=(1, latent_dim), mean = 0., std = 1.0)
	return z_m + K.exp(z_v / 2.) * eps

def loss(x, out):
	entropy = image_len * objectives.binary_crossentropy(x, out)
	KL_div = -0.5 * K.sum(1. + log_var - K.square(mean) - K.exp(log_var), axis = 1)
	return entropy + KL_div

(xtr, ytr), (xts, yts) = mnist.load_data()

xtr = xtr > THRESHOLD
xts = xts > THRESHOLD

xtr = xtr.reshape((len(xtr), np.prod(xtr.shape[1:])))
xts = xts.reshape((len(xts), np.prod(xts.shape[1:])))

x = Input(batch_shape=(batch_size, image_len))
encoder = Dense(NUM_LAYERS, activation = 'softmax')
encoder_mean = Dense(latent_dim)
encoder_var = Dense(latent_dim)

encoded_x = encoder(x)
mean = encoder_mean(encoded_x)
log_var = encoder_var(encoded_x)

latent = Lambda(batch_sampling, output_shape=(latent_dim, ))([mean, log_var])

decoder_latent = Dense(NUM_LAYERS, activation = 'softmax')
decoder_latent2 = Dense(image_len, activation = 'sigmoid')

latent_decoded = decoder_latent(latent)
output = decoder_latent2(latent_decoded)

vae = Model(x, output)
vae.compile(optimizer = 'adam', loss = loss)


history = vae.fit(xtr, xtr, nb_epoch = epochs, batch_size = batch_size, validation_data = (xts, xts), verbose = 2)
print history.history

plt.figure()
plt.subplot(211)
up, = plt.plot(range(epochs), history.history['loss'], c = 'r', label = 'Training loss')
down, = plt.plot(range(epochs), history.history['val_loss'], c = 'b', label = 'Testing loss')
plt.legend(handles=[up, down])


reconstructed = vae.predict(xtr, batch_size = batch_size)

test = [0] * latent_dim
count = 0
for i in range(len(xtr)):
	if ytr[i] == count:
		test[count] = i
		count += 1
	if count == 10:
		break

plt.figure(figsize = (20, 10))
for i in range(10):
	plt.subplot(2, 10, i + 1)
	digit = xtr[test[i]].reshape(28, 28)
	plt.imshow(digit, cmap = 'Greys_r')
	
	plt.subplot(2, 10, i + 11	)
	digit = reconstructed[test[i]].reshape(28, 28)
	plt.imshow(digit, cmap = 'Greys_r')

plt.show()




# print np.array([np.random.normal(0., 1., 10)]).shape

# sample = generate.predict(np.array([np.random.normal(0., 1., 10)]))
# digit = sample[0].reshape(image_res, image_res)

# plt.subplot(223)
# plt.imshow(digit, cmap='Greys_r')
# plt.show()





