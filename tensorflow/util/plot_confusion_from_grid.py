from util.confusion_matrix import plot_confusion_matrix
import numpy as np

cm = [[2216,   41,   80,   31,   39],
 [89,  232,    8,    3,    0],
 [60,   10,  394,    0,   10],
 [126,   19,    0,   10,    0],
 [96,    0,   13,    1,   20]]
cm = np.asarray(cm)
plot_confusion_matrix(cm.astype(np.int64), classes=[ "AVANT", "GAUCHE", "DROITE", "AVANT-GAUCHE", "AVANT-DROITE"], path=".")