#include <stdio.h>
#include "conv2D.h"

int main() {
    // Input image (5x5)
    image_t input[IMAGE_ROWS][IMAGE_COLS] = {
        {1, 2, 3, 4, 5},
        {6, 7, 8, 9, 10},
        {11, 12, 13, 14, 15},
        {16, 17, 18, 19, 20},
        {21, 22, 23, 24, 25}
    };

    // Kernel (3x3)
    kernel_t kernel[KERNEL_ROWS][KERNEL_COLS] = {
        {1, 0, -1},
        {1, 0, -1},
        {1, 0, -1}
    };

    // Output image (3x3)
    result_t output[IMAGE_ROWS - KERNEL_ROWS + 1][IMAGE_COLS - KERNEL_COLS + 1] = {0};

    // Perform 2D convolution
    conv2d(input, kernel, output);

    // Print the output image
    printf("Output Image:\n");
    for (int i = 0; i < IMAGE_ROWS - KERNEL_ROWS + 1; i++) {
        for (int j = 0; j < IMAGE_COLS - KERNEL_COLS + 1; j++) {
            printf("%4d ", output[i][j]);
        }
        printf("\n");
    }

    return 0;
}
