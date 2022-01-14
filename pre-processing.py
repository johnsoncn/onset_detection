import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from skimage import transform
import matplotlib.pyplot as plt
from librosa.feature import melspectrogram
from librosa import load
from scipy import signal
import scipy.io

'''
The names of those violin audio files was changed to ---  Timesteap + str(i)
The annotation files keep only the first column (timestamps), and also was changed to the name -- Timesteap + str(i)

'''

def img_process(t):

    imgs = np.zeros((t.size, 80, 15 ,3))
    return imgs


def img_input_resize(img ,height, width):

    img_1 = img.reshape(int(height), int(width), 3)
    img_resize = transform.resize(img_1, (80, 15))
    return img_resize


def read_spec(name):

    DIR = 'dataset/audio/'
    frequency, samplerate = load(DIR + name + '.wav',
                                     offset=459.3255329, duration=57)

    return frequency, samplerate

def onsetfile_read(name):

    DIR = 'dataset/onsets/'
    times = np.array([])

    with open(DIR + name + '.csv') as file:
        while True:
            line = file.readline()
            if line:
                times = np.append(times, float(line))
            else:
                break

    return times

# visualize spectrograms
def pcolor_spec(f, t, Sxx, init, fin):

    fig = Figure()
    canvas = FigureCanvas(fig)
    plt.pcolormesh(t[init: fin + 1],   f ,  Sxx[:, init: fin + 1],
                  edgecolors='face', cmap='rainbow')
    canvas.draw()
    img = canvas.tostring_rgb()
    img = np.frombuffer(img, dtype='uint8')
    width, height = fig.get_size_inches() * fig.get_dpi()
    img = img_input_resize(img,height, width)

    return img

# Keep the positive and negative samples in balance
def tolerance50_cal(data):
    imgs   = data[0]
    onsets = data[1]
    num_size = onsets.size
    red_size = len(np.where(onsets == 'n')[0]) - len(np.where(onsets == 'y')[0])
    for i in range(red_size):
        rand_idx = np.random.randint(0, num_size)
        while onsets[rand_idx] == 'y':
            rand_idx = np.random.randint(0, num_size)
        imgs = np.delete(imgs, rand_idx, 0)
        onsets = np.delete(onsets, rand_idx, 0)
        num_size -= 1
        red_size -= 1
    return imgs, onsets

def preprocessing(name):

    frequency, samplerate = read_spec(name)
    f, t, Sxx = signal.spectrogram(frequency, samplerate)
    onset_times = onsetfile_read(name)
    onset_times = onset_times - 459.3255329
    onsets = np.array(['n'] * t.size)
    prev_onset = 'n'
    imgs = img_process(t)
    # mel filter bank
    mel_sepc = melspectrogram(frequency, sr=samplerate,
                              hop_length = 512,
                              n_fft=2048, window='hann', win_length = 2048,
                              n_mels=80)
    for i in range(t.size):
        time = t[i]
        onsets[i] = 'n'
        if np.abs((onset_times - time)[np.argmin(np.abs(onset_times - time))]) < 0.05:  # tolerance = 50ms
            if prev_onset != 'y':
                onsets[i] = 'y'
        prev_onset = onsets[i]
        win  = pcolor_spec(f, t, mel_sepc, i,   i+1)
        win  = np.asfarray(((win  / np.max(win))  * 255), dtype='float32')
        imgs[i] = np.array([win])

    for ann in onset_times:
        plt.axvline(x=ann * 1, color='w', linestyle=':', linewidth=2)

    plt.savefig('./spec_with_onsets11.jpg')
    plt.show()

    return imgs, onsets


if __name__ == "__main__":
    # save processed audio and onset files to matlab format
    num_files = 24
    for i in range(num_files):
        names = 'Timestamp'
        ims, ls = tolerance50_cal(preprocessing(names + str(i)))
        scipy.io.savemat('mat/6audios/audio_tolerance+throw_' + str(i) +'.mat', mdict={'ims': ims})
        scipy.io.savemat('mat/6audios/onset_tolerance+throw_' + str(i) +'.mat', mdict={'ls': ls})