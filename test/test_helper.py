import unittest
import re
import sys
# Custom utility module import
sys.path.append('E:/OCR/aadhar_ocr/src')
import unittest
from unittest.mock import patch, MagicMock, mock_open
import numpy as np
import pandas as pd
import cv2
import os
import json

import common_utils


class TestCommonUtils(unittest.TestCase):

    def test_check_date_valid(self):
        self.assertTrue(common_utils.check_date("12/31/2020"))
        self.assertTrue(common_utils.check_date("1/1/23"))

    def test_check_date_invalid(self):
        self.assertFalse(common_utils.check_date("31-12-2020"))
        self.assertFalse(common_utils.check_date("abcd"))

    def test_rot_img(self):
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        df = pd.DataFrame({'xmin': [10], 'ymin': [20], 'xmax': [60], 'ymax': [80]})
        rotated_img, rotated_df = common_utils.rot_img(img, 90, df.copy())
        self.assertEqual(rotated_img.shape[2], 3)
        self.assertIn('xmin', rotated_df)

    def test_resize_and_bbox_scaling(self):
        img = np.ones((800, 600, 3), dtype=np.uint8)
        df = pd.DataFrame({'xmin': [100], 'ymin': [150], 'xmax': [300], 'ymax': [400]})
        resized_img, resized_df = common_utils.resize(img, df.copy())
        self.assertEqual(resized_img.shape, (400, 600, 3))
        self.assertTrue('xmin' in resized_df)

    def test_find_iou(self):
        box1 = (10, 10, 50, 50)
        box2 = (30, 30, 70, 70)
        iou = common_utils.find_iou(box1, box2)
        self.assertTrue(0 < iou < 1)

        # Completely outside
        box3 = (100, 100, 120, 120)
        self.assertEqual(common_utils.find_iou(box1, box3), 0.0)

    def test_date_rectifier(self):
        self.assertEqual(common_utils.date_rectifier("02/SZ/2O2I"), "02/52/2021")

    def test_parse_gender(self):
        self.assertEqual(common_utils.parse_gender("M/"), "Male")
        self.assertEqual(common_utils.parse_gender("F/"), "Female")
        self.assertEqual(common_utils.parse_gender(""), "")

    def test_parse_name_basic(self):
        self.assertEqual(common_utils.parse_name("C/O John Doe"), "John Doe")
        self.assertEqual(common_utils.parse_name(": Jane Smith"), "Jane Smith")

    def test_parse_date_good_dates(self):
        self.assertEqual(common_utils.parse_date("DOB: 11/12/1990"), "11/12/1990")

    @patch('common_utils.get_close_matches', return_value=["123456"])
    def test_find_pincode_with_match(self, mock_fuzzy):
        self.assertEqual(common_utils.find_pincode("12456", ["123456", "654321"]), "123456")

    @patch('common_utils.get_close_matches', return_value=[])
    def test_find_pincode_no_match(self, mock_fuzzy):
        self.assertEqual(common_utils.find_pincode("999999", ["123456", "654321"]), "999999")

    def test_parse_address_valid(self):
        address = "C/O John Doe, 123 Street, City, 560034, Karnataka"
        data = {"560034": "Karnataka"}
        pincode, state, name = common_utils.parse_address(address, data)
        self.assertEqual(pincode, "560034")
        self.assertEqual(state, "Karnataka")
        self.assertTrue("John" in name or "Doe" in name)

    @patch.dict(os.environ, {"CONFIG_FILE": "test_config.json"})
    @patch("builtins.open", new_callable=mock_open, read_data=json.dumps({
        "applications": [{"application_name": "test_app", "database": {"host": "localhost"}}],
        "database": {"port": 5432}
    }))
    def test_get_config_merged(self, mocked_file):
        conf = common_utils.get_config("database", "test_app", {})
        self.assertEqual(conf["host"], "localhost")
        self.assertEqual(conf["port"], 5432)

    @patch.dict(os.environ, {}, clear=True)
    def test_get_config_missing_env(self):
        conf = common_utils.get_config("database", "test_app", default={"default": True})
        self.assertEqual(conf, {"default": True})


if __name__ == '__main__':
    unittest.main()
