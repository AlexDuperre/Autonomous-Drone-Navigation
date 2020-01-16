"""Functions to call for the keylogger"""


import util.pyxhook as pyxhook

def init():
    global key
    key = "None"


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
        # Get the press event and get combo
        self.keys_held.add(event.Key)
        keycombo = self.get_key_combo_code()

        # Cast the key global
        global key
        key = keycombo

    finally:
        return True

  def on_key_up(self, event):
    try:
       # Remove release event from held keys
        self.keys_held.remove(event.Key)
        # Getting up the combo
        keycombo = self.get_key_combo_code()
        if keycombo=="":
            keycombo= "None"

        # Cast the key global
        global key
        key = keycombo

    finally:
        return True

  def shutdown(self):
    self.hm.UnhookKeyboard()