#ifndef IMAGE_PROCESSING_H
#define IMAGE_PROCESSING_H

#include <opencv2/core.hpp>

// Original single-threaded baseline. Arithmetic and cropped borders are preserved.
cv::Mat grayscale(const cv::Mat& frame);
cv::Mat sobelFilter(const cv::Mat& grayFrame);

// Creates threadCount std::thread workers for THIS frame and joins all of them.
// The returned Sobel image has frame.rows - 2 rows and frame.cols - 2 columns.
cv::Mat processFrameMultithreaded(const cv::Mat& frame, int threadCount);

#endif
