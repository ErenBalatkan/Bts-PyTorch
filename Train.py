import KittiDataLoader
import BTS
import os
import torch

experiment_name = "Balatkan"  # This determines folder names used for saving tensorboard logs and model files
dataset_path = "e://Code/Tez/bts_eren/kitti"

continue_training = False
total_epochs = 50

# This repository uses gradient accumilation to support bigger batch sizes then what
# would fit into memory, effective_batch_size determines total batch size with accumilation.
# batch size represents running batch size
# In example of effective batch size 16 and batch size of 2, batches will be accumilated 8 times, resulting in effective
# batch size of 16
effective_batch_size = 16
batch_size = 2

dataloader = KittiDataLoader.KittiDataLoader(batch_size, dataset_path)
model = BTS.BtsController(experiment_name, backprop_frequency=effective_batch_size/batch_size)

saved_models_path = os.path.join("models", experiment_name)
if not os.path.exists(saved_models_path):
    os.mkdir(saved_models_path)

start_epoch = 0

if continue_training:
    old_models = {int(x.split("_")[-1]): x for x in os.listdir(saved_models_path)}
    best_model = old_models[max(old_models)]
    print(best_model)

    model.load_model(os.path.join(saved_models_path, best_model))
    start_epoch = model.current_epoch
    print("Found model:", best_model, "Continuing training from epoch:", start_epoch)

import time
old_time = time.time()

for i in range(start_epoch, 50):
    for i_batch, sample_batched in enumerate(dataloader):
        print("Pass Time:", time.time() - old_time)
        old_time = time.time()
        img_inputs, img_labels = sample_batched["image"], sample_batched["label"]
        model.run_train_step(img_inputs, img_labels)

    model.current_epoch = i
    model.save_model(os.path.join(saved_models_path, "bts_model_epoch_"+str(i)))

    if 2 < i < 40:
        model.learning_rate_scheduler.step()
