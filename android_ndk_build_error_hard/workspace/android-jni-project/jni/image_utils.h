#ifndef IMAGE_UTILS_H
#define IMAGE_UTILS_H

#include <vector>
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Apply Gaussian blur to image data
 * @param data Pointer to image pixel data (RGBA format)
 * @param width Image width in pixels
 * @param height Image height in pixels
 * @param sigma Blur strength parameter
 */
void applyGaussianBlur(unsigned char* data, int width, int height, float sigma);

/**
 * Adjust brightness of image
 * @param data Pointer to image pixel data (RGBA format)
 * @param width Image width in pixels
 * @param height Image height in pixels
 * @param delta Brightness adjustment value (-255 to 255)
 */
void adjustBrightness(unsigned char* data, int width, int height, int delta);

/**
 * Convert color image to grayscale
 * @param data Pointer to image pixel data (RGBA format)
 * @param width Image width in pixels
 * @param height Image height in pixels
 */
void convertToGrayscale(unsigned char* data, int width, int height);

/**
 * Apply sharpening filter to image
 * @param data Pointer to image pixel data (RGBA format)
 * @param width Image width in pixels
 * @param height Image height in pixels
 * @param strength Sharpening strength (0.0 to 1.0)
 */
void sharpenImage(unsigned char* data, int width, int height, float strength);

#ifdef __cplusplus
}
#endif

#endif // IMAGE_UTILS_H