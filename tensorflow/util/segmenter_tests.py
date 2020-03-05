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
table = data_segmenter.segment((60,),0,1)

print(table[45][0])