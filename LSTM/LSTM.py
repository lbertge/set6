import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file


SEQ_LEN = 50
step = 5
NUM_GENERATE = 200

f = open('harrypotter.txt', 'r').read().lower()

seed = 'they each seized a broomstick and kicked off into '
start = f.find('they each seized a broomstick and kicked off into ')
# print f[start:start + len(seed)]

char = list(set(f))

char_idx = {}
idx_char = {}

for i in range(len(char)):
	char_idx[char[i]] = i
	idx_char[i] = char[i]

seq = []
nxt = []
for i in range(0, len(f) - SEQ_LEN, step):
	seq.append(f[i:i + SEQ_LEN])
	nxt.append(f[i + SEQ_LEN])

inp = np.zeros(((len(seq), SEQ_LEN, len(char))))
out = np.zeros(((len(seq), len(char))))

for i in range(len(seq)):
	for j in range(len(seq[i])):
		#one-hot vector
		inp[i][j][char_idx[seq[i][j]]] = 1
	out[i][char_idx[nxt[i]]] = 1


model = Sequential()
model.add(LSTM(128, input_shape= (SEQ_LEN, len(char))))
model.add(Dense(len(char)))
model.add(Activation('softmax'))

optimizer = RMSprop(lr=0.001)
model.compile(loss ='categorical_crossentropy', optimizer = optimizer)

tao = [0.25, 0.75, 1.5]

for epoch in range(20):

	model.fit(inp, out, batch_size = 128, nb_epoch = 1, verbose = 2)



	for temp in tao:
		generated = seed
		newSeed = seed
		for i in range(NUM_GENERATE):
			genInput = np.zeros((1, SEQ_LEN, len(char)))
			for j in range(len(newSeed)):
				#one-hot vector
				genInput[0][j][char_idx[newSeed[j]]] = 1
			dist = model.predict(genInput)[0]
			dist = np.asarray(dist).astype('float64')
			logs = np.log(dist) / temp
			softmax = np.exp(logs) / np.sum(np.exp(logs))
			multi = np.random.multinomial(1, softmax)
			sample = np.argmax(multi)

			pred_char = idx_char[sample]

			generated += pred_char
			#shift window by 1
			newSeed = newSeed[1:] + pred_char
			

		if epoch == 0 or epoch == 19:
			print "Temperature: ", temp
			print generated











