import json
import os
import os.path as osp
import pandas as pd
import numpy as np
DATA_PATH = 'downsample_24_train_data'
classes = ["N", "V", "PAB", "R", "L", "A"]

data_json = []
for cl in classes:
    for sig in os.listdir(osp.join(DATA_PATH, cl)):
        data_json.append({
            "path": osp.join(DATA_PATH, cl, sig),
            "label": cl,
            "filename": sig,
            "name": sig.split('.')[0]
        })
data_json = pd.DataFrame(data_json)

val_size = 0.2
val_ids = []
for cl in classes:
    val_ids.extend(
        data_json[data_json['label'] == cl].sample(frac=val_size, random_state=0)['name'].index
    )
val = data_json.loc[val_ids, :]
train = data_json[~data_json.index.isin(list(val.index))]
val.to_json(f'{DATA_PATH}/val.json', orient='records')
train.to_json(f'{DATA_PATH}/train.json', orient='records')
