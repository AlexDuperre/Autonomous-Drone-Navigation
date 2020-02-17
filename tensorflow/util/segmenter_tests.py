import shutil
import zipfile
from unittest import TestCase

import h5py
import numpy
import pandas as pd
import torch
from scipy.spatial.transform import Rotation

from util.MidAirSegmenter import MidAirDataSegmenter


data_segmenter = MidAirDataSegmenter("./")
table = data_segmenter.segment((20,50), 10)

print(table[45][0])