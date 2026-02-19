#!/usr/bin/env python3

import unittest
import numpy as np
import os
import sys
from volume_renderer import VolumeRenderer


class TestVolumeRenderer(unittest.TestCase):
    
    def setUp(self):
        """Create a VolumeRenderer instance for each test"""
        self.renderer = VolumeRenderer()
    
    def test_load_volume_simple(self):
        """Tests loading a simple 8x8x8 volume file"""
        test_file = '../test_data/sphere_8x8x8.vol'
        self.renderer.load_volume(test_file)
        self.assertIsNotNone(self.renderer.volume_data)
        dims = self.renderer.get_dimensions()
        self.assertEqual(dims, (8, 8, 8))
    
    def test_load_volume_dimensions(self):
        """Verifies correct dimension parsing"""
        test_file = '../test_data/sphere_8x8x8.vol'
        self.renderer.load_volume(test_file)
        width, height, depth = self.renderer.get_dimensions()
        self.assertEqual(width, 8)
        self.assertEqual(height, 8)
        self.assertEqual(depth, 8)
    
    def test_load_volume_data_count(self):
        """Checks that the correct number of data values are loaded"""
        test_file = '../test_data/sphere_8x8x8.vol'
        self.renderer.load_volume(test_file)
        width, height, depth = self.renderer.get_dimensions()
        expected_count = width * height * depth
        self.assertEqual(self.renderer.volume_data.size, expected_count)
    
    def test_get_dimensions(self):
        """Tests the get_dimensions method returns correct tuple"""
        test_file = '../test_data/gradient_16x16x16.vol'
        self.renderer.load_volume(test_file)
        dims = self.renderer.get_dimensions()
        self.assertIsInstance(dims, tuple)
        self.assertEqual(len(dims), 3)
        self.assertEqual(dims, (16, 16, 16))
    
    def test_sample_volume_at_grid_points(self):
        """Tests sampling at integer grid coordinates"""
        test_file = '../test_data/uniform_8x8x8.vol'
        self.renderer.load_volume(test_file)
        
        # Sample at a few grid points
        value1 = self.renderer.sample_volume(0, 0, 0)
        self.assertIsNotNone(value1)
        
        value2 = self.renderer.sample_volume(4, 4, 4)
        self.assertIsNotNone(value2)
        
        value3 = self.renderer.sample_volume(7, 7, 7)
        self.assertIsNotNone(value3)
    
    def test_sample_volume_interpolation(self):
        """Tests trilinear interpolation at fractional coordinates"""
        test_file = '../test_data/gradient_16x16x16.vol'
        self.renderer.load_volume(test_file)
        
        # Sample at fractional coordinates
        value1 = self.renderer.sample_volume(2.5, 3.5, 4.5)
        self.assertIsNotNone(value1)
        
        # Check that interpolation gives values between grid points
        val_low = self.renderer.sample_volume(2, 3, 4)
        val_high = self.renderer.sample_volume(3, 4, 5)
        
        # Interpolated value should be between the extremes
        min_val = min(val_low, val_high)
        max_val = max(val_low, val_high)
        self.assertGreaterEqual(value1, min_val - 0.1)
        self.assertLessEqual(value1, max_val + 0.1)
    
    def test_sample_volume_boundaries(self):
        """Tests boundary condition handling"""
        test_file = '../test_data/sphere_8x8x8.vol'
        self.renderer.load_volume(test_file)
        
        # Test outside boundaries - should return 0 or handle gracefully
        value_outside = self.renderer.sample_volume(-1, -1, -1)
        self.assertEqual(value_outside, 0.0)
        
        value_outside2 = self.renderer.sample_volume(10, 10, 10)
        self.assertEqual(value_outside2, 0.0)
        
        # Test at exact boundaries
        value_edge = self.renderer.sample_volume(7.9, 7.9, 7.9)
        self.assertIsNotNone(value_edge)
    
    def test_ray_box_intersection_hit(self):
        """Tests ray-box intersection when ray hits the volume"""
        test_file = '../test_data/sphere_8x8x8.vol'
        self.renderer.load_volume(test_file)
        
        # Ray starting outside, pointing at the volume center
        ray_origin = np.array([-5.0, 4.0, 4.0])
        ray_direction = np.array([1.0, 0.0, 0.0])
        ray_direction = ray_direction / np.linalg.norm(ray_direction)
        
        t_near, t_far = self.renderer.ray_box_intersection(ray_origin, ray_direction)
        
        self.assertIsNotNone(t_near)
        self.assertIsNotNone(t_far)
        self.assertGreater(t_far, t_near)
        self.assertGreaterEqual(t_near, 0.0)
    
    def test_ray_box_intersection_miss(self):
        """Tests ray-box intersection when ray misses"""
        test_file = '../test_data/sphere_8x8x8.vol'
        self.renderer.load_volume(test_file)
        
        # Ray pointing away from the volume
        ray_origin = np.array([-5.0, 4.0, 4.0])
        ray_direction = np.array([-1.0, 0.0, 0.0])
        ray_direction = ray_direction / np.linalg.norm(ray_direction)
        
        t_near, t_far = self.renderer.ray_box_intersection(ray_origin, ray_direction)
        
        # Should return None or negative values indicating no intersection
        self.assertTrue(t_near is None or t_near < 0 or t_near > t_far)
    
    def test_ray_march_uniform_field(self):
        """Tests ray marching through a uniform scalar field"""
        test_file = '../test_data/uniform_8x8x8.vol'
        self.renderer.load_volume(test_file)
        
        ray_origin = np.array([4.0, 4.0, -5.0])
        ray_direction = np.array([0.0, 0.0, 1.0])
        
        color = self.renderer.ray_march(ray_origin, ray_direction, step_size=0.5)
        
        self.assertIsNotNone(color)
        self.assertIsInstance(color, (float, np.floating))
        self.assertGreaterEqual(color, 0.0)
        self.assertLessEqual(color, 1.0)
    
    def test_ray_march_gradient(self):
        """Tests ray marching through a gradient field"""
        test_file = '../test_data/gradient_16x16x16.vol'
        self.renderer.load_volume(test_file)
        
        ray_origin = np.array([8.0, 8.0, -5.0])
        ray_direction = np.array([0.0, 0.0, 1.0])
        
        color = self.renderer.ray_march(ray_origin, ray_direction, step_size=0.3)
        
        self.assertIsNotNone(color)
        self.assertIsInstance(color, (float, np.floating))
        self.assertGreaterEqual(color, 0.0)
        self.assertLessEqual(color, 1.0)
    
    def test_render_image_basic(self):
        """Tests basic image rendering producing expected output dimensions"""
        test_file = '../test_data/sphere_8x8x8.vol'
        self.renderer.load_volume(test_file)
        
        width = 64
        height = 64
        
        image = self.renderer.render_image(width, height)
        
        self.assertIsNotNone(image)
        self.assertEqual(image.shape, (height, width))
        
        # Check that image values are in valid range
        self.assertGreaterEqual(np.min(image), 0.0)
        self.assertLessEqual(np.max(image), 1.0)
        
        # Check that not all values are the same (something was rendered)
        self.assertGreater(np.std(image), 0.001)


if __name__ == '__main__':
    unittest.main()