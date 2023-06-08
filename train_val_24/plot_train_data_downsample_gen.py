import multiprocessing as mp
import os.path as osp
import subprocess
from glob import glob
import os

from tqdm import tqdm

input_dir = "../mit-bih/*.atr"
ecg_data = sorted([osp.splitext(i)[0] for i in glob(input_dir)])
pbar = tqdm(total=len(ecg_data))


def run(file):
    test_file_names = ['100', '101', '103', '106', '102', '118', '109', '209']
    params = ["python", "plot_train_data_downsample.py", "--file", file]
    if osp.basename(file) in test_file_names:
        params.append("--test")
        params.append("1")
    subprocess.check_call(params)
    pbar.update(1)


if __name__ == "__main__":
    DATADIR = 'downsample_24_train_data'
    classes = ["N", "V", "PAB", "R", "L", "A", "!", "E"]
    for cl in classes:
        if not osp.exists(osp.join(DATADIR, cl)):
            os.makedirs(osp.join(DATADIR, cl))

    p = mp.Pool(processes=mp.cpu_count())
    p.map(run, ecg_data)
    print(ecg_data[0])
