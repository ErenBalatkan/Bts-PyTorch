import KittiDataLoader
import BTS
import os

experiment_name = "final_train7"
continue_training = True

batch_size = 1

dataloader = KittiDataLoader.KittiDataLoader(batch_size)

values = dataloader.dataset[255]

print(values.keys())
label = values["label"]

import numpy as np
print(label.shape)
print(np.max(label))
print(np.min(label))
print(np.mean(label))