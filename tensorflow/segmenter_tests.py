import shutil
import zipfile
from unittest import TestCase

import h5py
import numpy
import pandas as pd
import torch
from scipy.spatial.transform import Rotation

from MidAir import MidAirDataSegmenter, MidAirImageSequenceDataset, MidAirImageSequenceDatasetDeepVO
from MidAirDataPreprocessor import MidAirDataPreprocessor


class TestMidAirDataSegmenter(TestCase):
    def setUp(self):
        with zipfile.ZipFile("./Ressources/MidAir.zip", "r") as zip_ref:
            zip_ref.extractall("./Ressources")
        self.processor = MidAirDataPreprocessor("./Ressources/MidAir")
        self.processor.clean()

    def tearDown(self):
        shutil.rmtree("./Ressources/MidAir")

    def test_segment_givenSingleSegmentLength_shouldProduceFixedLengthSegments(self):
        data_segmenter = MidAirDataSegmenter("./Ressources/MidAir/Kite_test")
        data_segmenter.segment((4,), 0)
        sensor_records = h5py.File("./Ressources/MidAir/Kite_test/cloudy/sensor_records.hdf5", "r+")

        for trajectory in sensor_records:
            dataframe = pd.DataFrame(sensor_records[trajectory]["trajectory_segments"],
                                     columns=["sequence_length", "start_frame_index"])
            expected_length = len(
                list(range(0, data_segmenter.__get_trajectory_length__(sensor_records[trajectory]) - 4, 4)))

            self.assertEqual(len(dataframe), expected_length)

            expected_start_Frame_index = 0
            for i, row in dataframe.iterrows():
                self.assertEqual(row["sequence_length"], 4)
                self.assertEqual(row["start_frame_index"], expected_start_Frame_index)
                expected_start_Frame_index += 4

    def test_segment_givenNoneSegmentLength_shouldSegmentWillBeTrajectory(self):
        data_segmenter = MidAirDataSegmenter("./Ressources/MidAir/Kite_test")
        data_segmenter.segment(None)
        sensor_records = h5py.File("./Ressources/MidAir/Kite_test/cloudy/sensor_records.hdf5", "r+")

        for trajectory in sensor_records:
            dataframe = pd.DataFrame(sensor_records[trajectory]["trajectory_segments"],
                                     columns=["sequence_length", "start_frame_index"])
            expected_length = 1
            self.assertEqual(len(dataframe), expected_length)

            for i, row in dataframe.iterrows():
                self.assertEqual(row["sequence_length"], data_segmenter
                                 .__get_trajectory_length__(sensor_records[trajectory]))
                self.assertEqual(row["start_frame_index"], 0)

    def test_segment_givenSingleSegmentLengthAndOverlap_shouldProduceFixedLengthOverlappingSegments(self):
        data_segmenter = MidAirDataSegmenter("./Ressources/MidAir/Kite_test")
        data_segmenter.segment((4,), 1)
        sensor_records = h5py.File("./Ressources/MidAir/Kite_test/cloudy/sensor_records.hdf5", "r+")

        for trajectory in sensor_records:
            dataframe = pd.DataFrame(sensor_records[trajectory]["trajectory_segments"],
                                     columns=["sequence_length", "start_frame_index"])
            expected_length = len(
                list(range(0, data_segmenter.__get_trajectory_length__(sensor_records[trajectory]) - 4, 3)))

            self.assertEqual(len(dataframe), expected_length)

            expected_start_Frame_index = 0
            for i, row in dataframe.iterrows():
                self.assertEqual(row["sequence_length"], 4)
                self.assertEqual(row["start_frame_index"], expected_start_Frame_index)
                expected_start_Frame_index += 3

    def test_segment_givenRangeSegmentLength_shouldProduceVariableLengthOverlappingSegments(self):
        data_segmenter = MidAirDataSegmenter("./Ressources/MidAir/Kite_test")
        data_segmenter.segment((2, 4), 1)
        sensor_records = h5py.File("./Ressources/MidAir/Kite_test/cloudy/sensor_records.hdf5", "r+")

        for trajectory in sensor_records:
            dataframe = pd.DataFrame(sensor_records[trajectory]["trajectory_segments"],
                                     columns=["sequence_length", "start_frame_index"])

            self.assertTrue(3 in dataframe["sequence_length"])
            self.assertTrue(4 in dataframe["sequence_length"])
            self.assertTrue(2 in dataframe["sequence_length"])

            expected_start_Frame_index = 0
            for i, row in dataframe.iterrows():
                self.assertTrue(row["sequence_length"] <= 4 or row["sequence_length"] >= 2)
                self.assertEqual(row["start_frame_index"], expected_start_Frame_index)
                expected_start_Frame_index += row["sequence_length"] - 1

    def test_segment_givenRangeSegmentLength_shouldProduceVariableLengthSegments(self):
        data_segmenter = MidAirDataSegmenter("./Ressources/MidAir/Kite_test")
        data_segmenter.segment((2, 4), 0)
        sensor_records = h5py.File("./Ressources/MidAir/Kite_test/cloudy/sensor_records.hdf5", "r+")

        for trajectory in sensor_records:
            dataframe = pd.DataFrame(sensor_records[trajectory]["trajectory_segments"],
                                     columns=["sequence_length", "start_frame_index"])

            self.assertTrue(3 in dataframe["sequence_length"])
            self.assertTrue(4 in dataframe["sequence_length"])
            self.assertTrue(2 in dataframe["sequence_length"])

            expected_start_Frame_index = 0
            for i, row in dataframe.iterrows():
                self.assertTrue(row["sequence_length"] <= 4 or row["sequence_length"] >= 2)
                self.assertEqual(row["start_frame_index"], expected_start_Frame_index)
                expected_start_Frame_index += row["sequence_length"]