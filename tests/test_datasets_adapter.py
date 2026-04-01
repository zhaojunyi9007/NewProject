#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest

from pipeline.datasets import KittiAdapter, OSDaR23Adapter, get_adapter


class DatasetsAdapterTest(unittest.TestCase):
    def test_get_adapter_kitti(self):
        cfg = {"data": {"dataset_format": "kitti"}}
        self.assertIsInstance(get_adapter(cfg), KittiAdapter)

    def test_get_adapter_osdar_aliases(self):
        for fmt in ("osdar23", "osdar"):
            cfg = {"data": {"dataset_format": fmt}}
            self.assertIsInstance(get_adapter(cfg), OSDaR23Adapter)

    def test_kitti_optimizer_env_has_format(self):
        cfg = {"data": {"dataset_format": "kitti"}}
        env = get_adapter(cfg).get_optimizer_env()
        self.assertEqual(env.get("EDGECALIB_DATASET_FORMAT"), "kitti")

    def test_osdar_optimizer_env_has_camera_and_format(self):
        cfg = {"data": {"dataset_format": "osdar23", "image_sensor": "rgb_center"}}
        env = get_adapter(cfg).get_optimizer_env()
        self.assertEqual(env.get("EDGECALIB_DATASET_FORMAT"), "osdar23")
        self.assertEqual(env.get("EDGECALIB_OSDAR_CAMERA"), "rgb_center")


if __name__ == "__main__":
    unittest.main()
