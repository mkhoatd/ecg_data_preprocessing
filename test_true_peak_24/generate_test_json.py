import os
import os.path as osp

DATA_PATH = '24_test_data'
classes = ["N", "V", "PAB", "R", "L", "A"]

data_json = []
for cl in classes:
    data_json.extend(
        {
            "path": osp.join(DATA_PATH, cl, sig),
            "label": cl,
            "filename": sig,
            "name": sig.split('.')[0],
        }
        for sig in os.listdir(osp.join(DATA_PATH, cl))
    )
data_json.to_json(f'{DATA_PATH}/test.json', orient='records')
