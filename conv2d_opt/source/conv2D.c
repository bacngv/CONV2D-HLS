#include "conv2D.h"

void conv2d(image_t input[IMAGE_ROWS][IMAGE_COLS],
            kernel_t kernel[KERNEL_ROWS][KERNEL_COLS],
            result_t output[IMAGE_ROWS - KERNEL_ROWS + 1][IMAGE_COLS - KERNEL_COLS + 1]) {
    // Pragma directives for HLS optimization
    #pragma HLS ARRAY_PARTITION variable=input dim=2 complete
    #pragma HLS ARRAY_PARTITION variable=kernel dim=2 complete
    #pragma HLS ARRAY_PARTITION variable=output dim=2 complete

    // Line buffer for storing previous lines
    static image_t line_buffer[KERNEL_ROWS - 1][IMAGE_COLS];
    #pragma HLS ARRAY_PARTITION variable=line_buffer dim=1 complete

    // Window to store the current processing window
    static image_t window[KERNEL_ROWS][KERNEL_COLS];
    #pragma HLS ARRAY_PARTITION variable=window complete

    // Initialize line buffer
    INIT_LINE_BUFFER_ROW: for (int lb_row = 0; lb_row < KERNEL_ROWS - 1; lb_row++) {
        #pragma HLS PIPELINE
        INIT_LINE_BUFFER_COL: for (int col = 0; col < IMAGE_COLS; col++) {
            #pragma HLS UNROLL
            line_buffer[lb_row][col] = input[lb_row][col];
        }
    }

    // Convolution processing
    ROW_LOOP: for (int y = 0; y < IMAGE_ROWS - KERNEL_ROWS + 1; y++) {
        COL_LOOP: for (int x = 0; x < IMAGE_COLS - KERNEL_COLS + 1; x++) {
            #pragma HLS PIPELINE

            // Shift window vertically
            SHIFT_WINDOW_VERT: for (int win_row = 0; win_row < KERNEL_ROWS - 1; win_row++) {
                #pragma HLS UNROLL
                SHIFT_WINDOW_HORZ: for (int win_col = 0; win_col < KERNEL_COLS; win_col++) {
                    #pragma HLS UNROLL
                    window[win_row][win_col] = window[win_row + 1][win_col];
                }
            }

            // Fill last row of window
            FILL_WINDOW_LAST_ROW: for (int win_col = 0; win_col < KERNEL_COLS; win_col++) {
                #pragma HLS UNROLL
                window[KERNEL_ROWS - 1][win_col] =
                    (y + KERNEL_ROWS - 1 < IMAGE_ROWS)
                    ? input[y + KERNEL_ROWS - 1][x + win_col]
                    : line_buffer[KERNEL_ROWS - 2][x + win_col];
            }

            // Compute convolution
            result_t sum = 0;
            CONV_KERNEL_ROW: for (int kernel_row = 0; kernel_row < KERNEL_ROWS; kernel_row++) {
                #pragma HLS UNROLL
                CONV_KERNEL_COL: for (int kernel_col = 0; kernel_col < KERNEL_COLS; kernel_col++) {
                    #pragma HLS UNROLL
                    sum += window[kernel_row][kernel_col] * kernel[kernel_row][kernel_col];
                }
            }
            output[y][x] = sum;
        }

        // Update line buffer for next iteration
        if (y + KERNEL_ROWS < IMAGE_ROWS) {
            UPDATE_LINE_BUFFER_OUTER: for (int col = 0; col < IMAGE_COLS; col++) {
                #pragma HLS PIPELINE
                UPDATE_LINE_BUFFER_INNER: for (int lb_row = 0; lb_row < KERNEL_ROWS - 1; lb_row++) {
                    #pragma HLS UNROLL
                    line_buffer[lb_row][col] =
                        (lb_row + 1 < KERNEL_ROWS - 1)
                        ? line_buffer[lb_row + 1][col]
                        : input[y + KERNEL_ROWS - 1][col];
                }
            }
        }
    }
}
