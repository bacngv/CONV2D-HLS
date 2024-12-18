#ifndef __CONV2D_H__
#define __CONV2D_H__

#define IMAGE_ROWS 5      // Rows in the input image
#define IMAGE_COLS 5      // Columns in the input image
#define KERNEL_ROWS 3     // Rows in the kernel
#define KERNEL_COLS 3     // Columns in the kernel

typedef int image_t;      // Data type for image pixels
typedef int kernel_t;     // Data type for kernel elements
typedef int result_t;     // Data type for result elements

// Function prototype for 2D convolution
void conv2d(image_t input[IMAGE_ROWS][IMAGE_COLS],
            kernel_t kernel[KERNEL_ROWS][KERNEL_COLS],
            result_t output[IMAGE_ROWS - KERNEL_ROWS + 1][IMAGE_COLS - KERNEL_COLS + 1]);

#endif // __CONV2D_H__
