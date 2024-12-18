#include "conv2D.h"

// Function to perform 2D convolution using line buffers with HLS pragmas for optimization
void conv2d(image_t input[IMAGE_ROWS][IMAGE_COLS],
            kernel_t kernel[KERNEL_ROWS][KERNEL_COLS],
            result_t output[IMAGE_ROWS - KERNEL_ROWS + 1][IMAGE_COLS - KERNEL_COLS + 1]) {
    
    // Line buffer for storing a row of the input image
    image_t line_buffer[KERNEL_ROWS][IMAGE_COLS];
    
    // HLS directive: Pipeline the outer loop for better throughput
    #pragma HLS PIPELINE II=1

    // Slide kernel over the input image
    for (int x = 0; x < IMAGE_ROWS - KERNEL_ROWS + 1; x++) {
        
        // HLS directive: Partition the input image for better memory access
        #pragma HLS ARRAY_PARTITION variable=line_buffer complete dim=1
        
        // Shift the line buffer for each row
        if (x == 0) {
            // Load the first KERNEL_ROWS rows into the line buffer
            for (int i = 0; i < KERNEL_ROWS; i++) {
                #pragma HLS UNROLL
                for (int j = 0; j < IMAGE_COLS; j++) {
                    line_buffer[i][j] = input[i][j];
                }
            }
        } else {
            // Shift the buffer up by one row (slide window effect)
            for (int i = 1; i < KERNEL_ROWS; i++) {
                #pragma HLS UNROLL
                for (int j = 0; j < IMAGE_COLS; j++) {
                    line_buffer[i-1][j] = line_buffer[i][j];
                }
            }
            // Load the next row of the input image
            for (int j = 0; j < IMAGE_COLS; j++) {
                line_buffer[KERNEL_ROWS-1][j] = input[x + KERNEL_ROWS-1][j];
            }
        }

        // Convolve the kernel over the buffered rows
        for (int y = 0; y < IMAGE_COLS - KERNEL_COLS + 1; y++) {
            int sum = 0; // Accumulator for the convolution result

            // HLS directive: Unroll the inner loop to reduce loop overhead
            #pragma HLS UNROLL

            for (int i = 0; i < KERNEL_ROWS; i++) {
                for (int j = 0; j < KERNEL_COLS; j++) {
                    sum += line_buffer[i][y + j] * kernel[i][j];
                }
            }
            output[x][y] = sum; // Store the result
        }
    }
}
