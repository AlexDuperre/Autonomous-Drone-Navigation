import torch
import numpy as np
import pyautogui
from util.tools import fix_angle
import cv2
from models.model import LSTMModel

class Navigate(object):

    def __init__(self):
        # LSTM Hyperparameters
        self.hyper_params = {
            "validationRatio": 0.3,
            "validationTestRatio": 0.5,
            "batch_size": 100,
            "learning_rate": 0.01,
            "specific_lr": 0.001,
            "lr_scheduler_step": 12,
            "num_epochs": 45,
            "input_dim": 650,
            "hidden_dim": 1000,
            "layer_dim": 1,
            "output_dim": 5,
            "frame_nb": 100,
            "sub_segment_nb": 1,
            "segment_overlap": 0,
            "patience": 10,
            "skip_frames": 3
        }

        self.keys_dict = {
            0: "w",
            1: "q",
            2: "e",
            3: ["w", "q"],
            4: ["w", "e"]
        }

        # Add Pytorch Navigator network

        self.model = LSTMModel(input_dim=self.hyper_params["input_dim"],
                          hidden_dim=self.hyper_params["hidden_dim"],
                          layer_dim=self.hyper_params["layer_dim"],
                          output_dim=self.hyper_params["output_dim"],
                          Pretrained=False)

        state_dict = torch.load("./Best_models/BEST/checkpoint.pt")
        self.model.load_state_dict(state_dict)

        # Initialize hidden state with zeros
        self.hn = torch.zeros(self.hyper_params["layer_dim"], 1, self.hyper_params["hidden_dim"]).requires_grad_()
        # Initialize cell state
        self.cn = torch.zeros(self.hyper_params["layer_dim"], 1, self.hyper_params["hidden_dim"]).requires_grad_()

        # model = model
        self.model.eval()

        self.calls = 0
    def forward(self, pred, rel_orientation):
        self.calss += 1
        depth = cv2.resize(pred[0,:,:,0], dsize=(160, 92), interpolation=cv2.INTER_CUBIC)
        lstm_inputA = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0)
        lstm_inputB = torch.from_numpy(np.asarray(fix_angle(rel_orientation))).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        out, (self.hn, self.cn) = self.model([lstm_inputA, lstm_inputB], self.hn, self.cn)
        _, predicted = torch.max(out.data, 2)
        predicted = predicted.numpy()[0][0]

        command = self.keys_dict[predicted]

        # makes sure we are in the drone commande window
        # subprocess.call(['./activate_window.sh'])


        if predicted == 0:
            print("w")
            pyautogui.keyDown(command)

            # if start:
            #     t0 = time.time()
            #     start = False

        elif any(predicted == [1, 2]):
            print(command)
            pyautogui.keyUp("w")
            pyautogui.keyDown(command, pause=1.5)
            pyautogui.keyUp(command)

            # dt = time.time() - t0

        elif predicted == 3:
            print("w+q")
            # if not start:
            pyautogui.keyDown("w")
            pyautogui.keyDown("q", pause=0.5)
            pyautogui.keyUp("q")

            # if not start:
            #     pyautogui.keyUp("w")

            # dt = time.time() - t0

        elif predicted == 4:
            print("w+e")
            # if not start:
            pyautogui.keyDown("w")
            pyautogui.keyDown("e", pause=0.5)
            pyautogui.keyUp("e")

            # if not start:
            #     pyautogui.keyUp("w")
            #
            # dt = time.time() - t0

        # Refresh the hidden state (to be deactivated for long sequences)
        if self.calls % 30 == 0:
            # Initialize hidden state with zeros
            self.hn = torch.zeros(self.hyper_params["layer_dim"], 1, self.hyper_params["hidden_dim"]).requires_grad_()
            # Initialize cell state
            self.cn = torch.zeros(self.hyper_params["layer_dim"], 1, self.hyper_params["hidden_dim"]).requires_grad_()

            print("Reresh hidden state")