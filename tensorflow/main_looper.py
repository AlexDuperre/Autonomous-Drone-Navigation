"""

Main looper

"""

from main import main
import os
import shutil
import time


hyper_params = {
    "validationRatio" : 0.3,
    "validationTestRatio" : 0.5,
    "pretrained" : True,
    "batch_size" : 100,
    "learning_rate" : 0.001,
    "specific_lr" : 0.00001,
    "rep_lr" : 0.0001,
    "weight_decay" : 0.01,
    "lr_scheduler_step" : 20,
    "num_epochs" : 65,
    "input_dim" : 850,
    "hidden_dim" : 1000,
    "layer_dim" : 1,
    "output_dim" : 5,
    "frame_nb" : 100,
    "sub_segment_nb": 1,
    "segment_overlap": 0,
    "patience" : 20,
    "skip_frames" : 3,
    "train_augmentation": True,
    "valid_augmentation" : False,
    "prob_goal" : 0.4,
    "prob_flip" : 0.4
}

probs = [(0,0), (0.2,0), (0.4,0), (0.6,0), (0,0.2), (0,0.4), (0,0.6), (0.2,0.2), (0.4,0.2), (0.6,0.2), (0.2,0.4), (0.2,0.6), (0.4,0.4), (0.6,0.4), (0.4,0.6), (0.6,0.6)]

for i,pair in enumerate(probs):
    hyper_params["prob_goal"] = pair[0]
    hyper_params["prob_flip"] = pair[1]

    print(hyper_params["prob_goal"],hyper_params["prob_flip"] )

    root = "./Best_models/" + "Full loss + " + str(hyper_params["prob_goal"]) + " goal + " + str(hyper_params["prob_flip"]) + " flip"
    os.makedirs(root)

    print("Runing main experiment #{} out of {}".format(i+1, len(probs)))
    main(hyper_params)

    print("Saving...")
    shutil.copyfile("./checkpoint.pt", root + "/checkpoint.pt")
    shutil.copyfile("./confusion_matrix.png", root + "/confusion_matrix.png")
    shutil.copyfile("./hyper_params.txt", root + "/hyper_params.txt")
    shutil.copytree("./models", root + "/models")
    time.sleep(120)




