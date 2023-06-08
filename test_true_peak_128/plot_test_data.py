import wfdb
import neurokit2 as nk
import os
import os.path as osp
from wfdb import rdrecord
import cv2
import matplotlib.pyplot as plt
import tqdm


MIT_DIR = '../mit-bih'
sig_names = list(map(lambda x: x.split('.')[0], 
                     filter(lambda x: x.endswith('.dat'), os.listdir(MIT_DIR))))
sig_names.sort(key=lambda x: int(x))


sig_names_with_mlii = []
for sig_name in sig_names:
    record = rdrecord(osp.join(MIT_DIR, sig_name))
    if 'MLII' in record.sig_name:
        sig_names_with_mlii.append((sig_name, record.sig_name.index('MLII')))


classes = ["N", "V", "PAB", "R", "L", "A", "!", "E"]

test_file_names = ['100', '101', '103', '106', '107', '118', '109', '209']

bar = tqdm.tqdm(total=len(test_file_names))


OUTPUT_DIR = '128_test_data'
if not osp.exists(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)
for class_name in classes:
    if not osp.exists(osp.join(OUTPUT_DIR, class_name)):
        os.mkdir(osp.join(OUTPUT_DIR, class_name))


def plot(signal, figsize, image_size, filename):
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



image_size = 128
fig = plt.figure(frameon=False)
dpi = fig.dpi
figsize = (image_size / dpi, image_size / dpi)


def create_data(sig_names_with_mlii):
    for sig_name, sig_idx in sig_names_with_mlii:
        if sig_name not in test_file_names:
            continue
        record = rdrecord(osp.join(MIT_DIR, sig_name))
        signal = record.p_signal[:, sig_idx]
        _, rpeaks = nk.ecg_peaks(signal, sampling_rate=33, method='nabian2018')
        ann = wfdb.rdann(osp.join(MIT_DIR, sig_name), 'atr')

        for label, peak in zip(ann.symbol, ann.sample):
            if label == '/':
                label = 'PAB'
            if label not in classes:
                continue
            filename = osp.join(OUTPUT_DIR, label, f'{sig_name}_{peak}.png')
            plot(signal[peak - 64: peak + 64], figsize, (128, 128), filename)
        bar.update(1)


create_data(sig_names_with_mlii)
