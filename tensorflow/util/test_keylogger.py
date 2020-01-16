
"""
Copyright (c) 2015, Aman Deep
All rights reserved.
A simple keylogger witten in python for linux platform
All keystrokes are recorded in a log file.
The program terminates when grave key(`) is pressed
grave key is found below Esc key
"""

import util.pyxhook as pyxhook

# #change this to your log file's path
# log_file='./file.log'
#
# #this function is called everytime a key is pressed.
# def OnKeyPress(event):
#   # fob=open(log_file,'a')
#   # fob.write(event.Key)
#   # fob.write('\n')
#   print(event.Key)
#
#   if event.Ascii==96: #96 is the ascii value of the grave key (`)
#     #fob.close()
#     new_hook.cancel()
# #instantiate HookManager class
# new_hook=pyxhook.HookManager()
# #listen to all keystrokes
# new_hook.KeyDown=OnKeyPress
# #hook the keyboard
# new_hook.HookKeyboard()
# #start the session
# new_hook.start()

class Keystroke_Watcher:
  def __init__(self):
    self.hm = pyxhook.HookManager()
    self.hm.KeyDown = self.on_key_down
    self.hm.KeyUp = self.on_key_up
    self.hm.HookKeyboard()
    self.keys_held = set()  # set of all keys currently being pressed

  def get_key_combo_code(self):
    # find some way of encoding the presses.
    return '+'.join([key for key in self.keys_held])

  def on_key_down(self, event):
    try:
      self.keys_held.add(event.Key)
      # print(event.Key)
      # [print(key) for key in self.keys_held]
    finally:
      return True

  def on_key_up(self, event):
    keycombo = self.get_key_combo_code()
    print("keycombo: ", keycombo)
    try:
      print("success")# Do whatever you want with your keycombo here
    finally:
      self.keys_held.remove(event.Key)
      return True

  def shutdown(self):
    self.hm.PostQuitMessage(0)
    self.hm.UnhookKeyboard()


