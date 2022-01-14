import numpy as np
import keras
from keras.models import Sequential
import scipy.io
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

BASE_DATA = 'mat/6audios/'
learning_rate = 0.01
# visualize the training acc-loss
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch':[], 'epoch':[]}
        self.accuracy = {'batch':[], 'epoch':[]}
        self.val_loss = {'batch':[], 'epoch':[]}
        self.val_acc = {'batch':[], 'epoch':[]}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc="upper right")
        plt.show()

def conv_labels(ls):
    new_ls = np.zeros((int(ls.size), 2))
    new_ls[np.where(ls == 'n'), 0] = 0.
    new_ls[np.where(ls == 'y'), 1] = 1.
    return new_ls

def conv_labels1(ls):
    new_ls = np.zeros((int(ls.size), 1))
    new_ls[np.where(ls == 'n'), 0] = 0
    new_ls[np.where(ls == 'y'), 0] = 1
    return new_ls

# CNN
model = Sequential()
model.add(keras.layers.convolutional.ZeroPadding2D((1, 1), input_shape=(80, 15, 3)))
model.add(Conv2D(16, (3, 3), activation='relu'))  # , input_shape=(80, 15, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())

model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))

model.add(BatchNormalization())
model.add(Dense(2, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

history = LossHistory()

# 8-fold validation
k_fold = KFold(n_splits=8, shuffle=True, random_state=42)

AUDIO = 'audio_tolerance+throw_'
ONSET = 'onset_tolerance+throw_'

for i in range(24):
    ims = scipy.io.loadmat(BASE_DATA + AUDIO + str(i) + '.mat')['ims']
    if i == 0:
        y_train_all = conv_labels(scipy.io.loadmat(BASE_DATA + ONSET + str(i) + '.mat')['ls'])
        x_train_all = (ims - 127.5) / 127.5
        print(y_train_all.shape, x_train_all.shape)
    else:
        y_train_all = np.vstack((y_train_all, conv_labels(scipy.io.loadmat(BASE_DATA + ONSET + str(i) + '.mat')['ls'])))
        x_train_all = np.vstack((x_train_all, (ims - 127.5) / 127.5))
        print(y_train_all.shape, x_train_all.shape)

indices = np.arange(int(y_train_all.size / 2))
train_test_set = k_fold.split(indices)

for (train_set, test_set) in train_test_set:
    x_train = x_train_all[train_set - 1]
    y_train = y_train_all[train_set - 1]
    print(train_set.shape, test_set.shape)
    model.fit(x_train, y_train, epochs=5, batch_size=100, shuffle=True)

    x_test = x_train_all[test_set - 1]
    y_test = y_train_all[test_set - 1]

    y_pred = model.predict(x_test, batch_size=100)

    new_y_pred = np.argmax(y_pred, axis=1)

    print(classification_report(np.argmax(y_test, axis=1), new_y_pred))