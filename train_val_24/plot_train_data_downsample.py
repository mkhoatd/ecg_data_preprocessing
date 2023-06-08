# %%
import os
import os.path as osp
import cv2
import matplotlib.pyplot as plt
import numpy as np
import wfdb
from sklearn.preprocessing import scale
from wfdb import rdrecord
import neurokit2 as nk
from glob import glob
import argparse

# %%
parser = argparse.ArgumentParser()
parser.add_argument("--file", required=True)
parser.add_argument("--test", default=0, type=int)
args = parser.parse_args()
ecg = args.file
test = args.test
if test:
    exit(0)
# ecg = 'mit-bih/115'
name = osp.basename(ecg)
record = rdrecord(ecg)
ann = wfdb.rdann(ecg, extension='atr')
# record.sig_name[0] == 'MLII', record.sig_name[1]=='V5
# nen chi lay record.p_signal.T[1]
# signal = record.p_signal.T[1]
# sig_name = record.sig_name[1]
sig_idxs = []
for i in range(len(record.sig_name)):
    if record.sig_name[i] == 'MLII':
        sig_idxs.append(i)

    # plot = nk.events_plot([ann.sample], signal)
    # plt.show()
    # %%
try:
    FACTOR = 360 / 33
    MIN_PEAK_DIFF_AFTER_DOWNSAMPLE = 2


    def downsampling(signal, factor):
        factor = int(factor)
        output_length = int(len(signal) // factor)
        output_signal = np.zeros(output_length)

        for i in range(output_length):
            output_signal[i] = signal[i * factor]

        return np.array(output_signal)


    def downsampling_peak(peaks, factor):
        factor = int(factor)
        new_peaks = peaks // factor
        rpeak_ranges = []
        new_peaks = new_peaks[1:-1]
        return new_peaks


    signals = []
    for i in sig_idxs:
        signals.append(downsampling(record.p_signal.T[i], FACTOR))
    # rpeaks = [downsampling_peak(signals[i], ann.sample, FACTOR) for i in sig_idxs]
    rpeaks = downsampling_peak(ann.sample, FACTOR)

    mode = 12

    image_size = 128

    # dpi fix
    fig = plt.figure(frameon=False)
    dpi = fig.dpi

    # fig size / image size
    figsize = (image_size / dpi, image_size / dpi)
    image_size = (image_size, image_size)

    classes = ["N", "V", "PAB", "R", "L", "A", "!", "E"]


    def plot(signal, filename):
        plt.figure(figsize=figsize, frameon=False)
        plt.axis("off")
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        # plt.margins(0, 0) # use for generation images with no margin
        plt.plot(signal)
        plt.savefig(filename)

        plt.close()

        im_gray = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        im_gray = cv2.resize(im_gray, image_size, interpolation=cv2.INTER_LANCZOS4)
        cv2.imwrite(filename, im_gray)


    if __name__ == "__main__":
        DATADIR = 'downsample_24_train_data'
        if not os.path.exists(DATADIR):
            os.makedirs(DATADIR)
        for signal in signals:
            for i, (label, peak) in enumerate(zip(ann.symbol, rpeaks)):
                if label == "/":
                    label = "PAB"
                if label not in classes:
                    continue
                if isinstance(mode, int):
                    left, right = peak - mode // 2, peak + mode // 2
                else:
                    raise Exception("Wrong mode in script beginning")

                if np.all([left > 0, right < len(signal)]):
                    if osp.exists(osp.join(DATADIR, label)) is False:
                        os.makedirs(osp.join(DATADIR, label))
                    filename = osp.join(DATADIR, label, f"{name}-{peak}.png")

                    plot(signal[left:right], filename)
except Exception as e:
    with open('log.txt', 'a') as f:
        f.write(str({ecg: str(e)}))
