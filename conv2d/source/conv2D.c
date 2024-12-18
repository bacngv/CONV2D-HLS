#include "conv2D.h"

// Function to perform 2D convolution
void conv2d(image_t input[IMAGE_ROWS][IMAGE_COLS],
            kernel_t kernel[KERNEL_ROWS][KERNEL_COLS],
            result_t output[IMAGE_ROWS - KERNEL_ROWS + 1][IMAGE_COLS - KERNEL_COLS + 1]) {
    // Slide kernel over the input image
    for (int x = 0; x < IMAGE_ROWS - KERNEL_ROWS + 1; x++) {
        for (int y = 0; y < IMAGE_COLS - KERNEL_COLS + 1; y++) {
            int sum = 0; // Accumulator for the convolution result
            for (int i = 0; i < KERNEL_ROWS; i++) {
                for (int j = 0; j < KERNEL_COLS; j++) {
                    sum += input[x + i][y + j] * kernel[i][j];
                }
            }
            output[x][y] = sum; // Store the result
        }
    }
}
