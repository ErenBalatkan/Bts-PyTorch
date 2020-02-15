import KittiDataLoader
import BTS
import os
import torch
import configs

experiment_name = configs.EXPERIMENT_NAME # This determines folder names used for saving tensorboard logs and model files
dataset_path = configs.DATASET_PATH

total_epochs = configs.TOTAL_TRAIN_EPOCHS

# This repository uses gradient accumulation to support bigger batch sizes then what
# would fit into memory, effective_batch_size determines total batch size with accumulation.
# batch size represents running batch size
# In example of effective batch size 4 and batch size of 2, batches will be accumulated 2 times, resulting in effective
# batch size of 4
effective_batch_size = 4
batch_size = 2

dataloader = KittiDataLoader.KittiDataLoader(batch_size, dataset_path)
model = BTS.BtsController(experiment_name, backprop_frequency=effective_batch_size/batch_size)

saved_models_path = os.path.join("models", experiment_name)
if not os.path.exists(saved_models_path):
    os.mkdir(saved_models_path)

start_epoch = 0
if os.path.exists(os.path.join(saved_models_path, "bts_latest")):
    model.load_model(os.path.join(saved_models_path, "bts_latest"))
    start_epoch = model.current_epoch

    skip_step = model.current_step % (len(dataloader) / batch_size)
    print("Found model, step:", skip_step, "epoch:", start_epoch)

import time
old_time = time.time()

for i in range(start_epoch, 50):
    for i_batch, sample_batched in enumerate(dataloader):
        if i_batch > skip_step:
            img_inputs, img_labels, img_focals = sample_batched["image"], sample_batched["label"], sample_batched["focal_length"]
            model.run_train_step(img_inputs, img_labels, img_focals)

            if i_batch % 100 == 0:
                print("100 Iteration Pass Time: %.2f Epoch progress: %.2f" % ((time.time() - old_time), 100 * i_batch/len(dataloader)))
                old_time = time.time()

            if i_batch % 1000 == 999:
                print("Beginning Save")
                model.save_model(os.path.join(saved_models_path, "bts_latest"))
                print("Save Complete Step:", model.current_step % (len(dataloader) / batch_size), "Epoch:", model.current_epoch)
        else:
            if i_batch % 100 == 0:
                print("Skipping to:", skip_step, "currently at", i_batch)

    skip_step = 0
    print("Epoch :", i)
    model.learning_rate_scheduler.step()
    model.current_epoch = i
    model.save_model(os.path.join(saved_models_path, "bts_model_epoch_"+str(i)))
