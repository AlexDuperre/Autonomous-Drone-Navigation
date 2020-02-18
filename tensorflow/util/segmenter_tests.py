import shutil
import zipfile
from unittest import TestCase

import h5py
import numpy
import pandas as pd
import torch
from scipy.spatial.transform import Rotation

from util.MidAirSegmenter import DNDSegmenter


data_segmenter = DNDSegmenter("./")
table = data_segmenter.segment((20,50), 10)

print(table[45][0])