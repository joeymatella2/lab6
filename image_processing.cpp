/************************************************************
 * File: image_processing.cpp
 * Original image-processing arithmetic: Joseph Matella
 * C++11 threading version: horizontal slices, shared buffers,
 * and new threads for every frame.
 ************************************************************/

#include "image_processing.h"

#include <algorithm>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

namespace
{
// C++11 has no built-in barrier. This small one-use barrier lets workers wait
// until every grayscale slice is finished. A fresh object is made each frame.
struct GrayBarrier
{
    std::mutex mutex;                  // Protects the arrived counter.
    std::condition_variable condition; // Lets early workers sleep.
    int arrived = 0;
    const int workerCount;

    explicit GrayBarrier(int count) : workerCount(count) {}

    void wait()
    {
        std::unique_lock<std::mutex> lock(mutex);
        ++arrived;

        if (arrived == workerCount)
        {
            condition.notify_all(); // Last worker wakes the waiting workers.
        }
        else
        {
            while (arrived < workerCount)
            {
                // Releases the mutex while sleeping, then reacquires it.
                // Recheck the count after waking, even for a spurious wakeup.
                condition.wait(lock);
            }
        }
        // The unique_lock automatically unlocks the mutex when it leaves scope.
    }
};

// endRow is exclusive: [startRow, endRow).
void grayscaleRows(const cv::Mat& frame, cv::Mat& grayFrame,
                   int startRow, int endRow)
{
    for (int y = startRow; y < endRow; ++y)
    {
        const uint8_t* row_ptr = frame.ptr<uint8_t>(y);
        uint8_t* gray_row_ptr = grayFrame.ptr<uint8_t>(y);

        for (int x = 0; x < frame.cols; ++x)
        {
            // OpenCV input is BGR. Keep your original fixed-point formula.
            gray_row_ptr[x] = static_cast<uint8_t>(
                (54 * row_ptr[x * 3 + 2]
                 + 183 * row_ptr[x * 3 + 1]
                 + 18 * row_ptr[x * 3]) >> 8);
        }
    }
}

void sobelRows(const cv::Mat& grayFrame, cv::Mat& sobelFrame,
               int startRow, int endRow)
{
    // Clamp ONLY at the actual image boundaries. Internal slice boundaries
    // still read their neighbors from the shared grayscale image.
    const int first = std::max(startRow, 1);
    const int last = std::min(endRow, grayFrame.rows - 1);

    for (int y = first; y < last; ++y)
    {
        const uint8_t* row_ptr_top = grayFrame.ptr<uint8_t>(y - 1);
        const uint8_t* row_ptr_mid = grayFrame.ptr<uint8_t>(y);
        const uint8_t* row_ptr_bot = grayFrame.ptr<uint8_t>(y + 1);
        uint8_t* sobel_row_ptr = sobelFrame.ptr<uint8_t>(y - 1);

        for (int x = 1; x < grayFrame.cols - 1; ++x)
        {
            int gx = (row_ptr_top[x + 1]
                      + 2 * row_ptr_mid[x + 1]
                      + row_ptr_bot[x + 1])
                   - (row_ptr_top[x - 1]
                      + 2 * row_ptr_mid[x - 1]
                      + row_ptr_bot[x - 1]);

            int gy = (row_ptr_bot[x - 1]
                      + 2 * row_ptr_bot[x]
                      + row_ptr_bot[x + 1])
                   - (row_ptr_top[x - 1]
                      + 2 * row_ptr_top[x]
                      + row_ptr_top[x + 1]);

            int abs_gx = (gx < 0) ? -gx : gx;
            int abs_gy = (gy < 0) ? -gy : gy;
            int magnitude = (abs_gx > abs_gy)
                ? abs_gx + (abs_gy >> 1)
                : abs_gy + (abs_gx >> 1);

            if (magnitude > 255)
            {
                magnitude = 255;
            }
            sobel_row_ptr[x - 1] = static_cast<uint8_t>(magnitude);
        }
    }
}

// std::thread accepts the worker's arguments directly. Passing addresses shares
// the original image buffers; it does not copy any image pixels.
void processSlice(const cv::Mat* frame, cv::Mat* grayFrame, cv::Mat* sobelFrame,
                  int startRow, int endRow, GrayBarrier* barrier)
{
    grayscaleRows(*frame, *grayFrame, startRow, endRow);
    barrier->wait(); // Neighboring grayscale rows must be ready before Sobel.
    sobelRows(*grayFrame, *sobelFrame, startRow, endRow);
    // Returning ends this worker. There is no persistent worker loop.
}
} // namespace

cv::Mat grayscale(const cv::Mat& frame)
{
    if (frame.empty() || frame.type() != CV_8UC3)
    {
        throw std::invalid_argument("grayscale needs a nonempty CV_8UC3 BGR image.");
    }
    cv::Mat grayFrame(frame.rows, frame.cols, CV_8UC1);
    grayscaleRows(frame, grayFrame, 0, frame.rows);
    return grayFrame;
}

cv::Mat sobelFilter(const cv::Mat& grayFrame)
{
    if (grayFrame.type() != CV_8UC1 || grayFrame.rows < 3 || grayFrame.cols < 3)
    {
        throw std::invalid_argument("Sobel needs a CV_8UC1 image at least 3x3.");
    }
    cv::Mat sobelFrame(grayFrame.rows - 2, grayFrame.cols - 2, CV_8UC1);
    sobelRows(grayFrame, sobelFrame, 1, grayFrame.rows - 1);
    return sobelFrame;
}

cv::Mat processFrameMultithreaded(const cv::Mat& frame, int threadCount)
{
    if (threadCount < 1)
    {
        throw std::invalid_argument("The thread count must be positive.");
    }
    if (frame.type() != CV_8UC3 || frame.rows < 3 || frame.cols < 3)
    {
        throw std::invalid_argument("Processing needs a CV_8UC3 image at least 3x3.");
    }

    // Allocate ONE shared grayscale image and ONE shared Sobel image.
    // No worker allocates or copies an image slice.
    cv::Mat grayFrame(frame.rows, frame.cols, CV_8UC1);
    cv::Mat sobelFrame(frame.rows - 2, frame.cols - 2, CV_8UC1);

    GrayBarrier barrier(threadCount);
    std::vector<std::thread> workers;
    workers.reserve(threadCount); // Reserve space; no threads start yet.

    for (int i = 0; i < threadCount; ++i)
    {
        // Integer division distributes remainder rows automatically.
        // A wide intermediate avoids overflow when multiplying i * rows.
        int startRow = static_cast<int>(
            static_cast<long long>(i) * frame.rows / threadCount);
        int endRow = static_cast<int>(
            (static_cast<long long>(i) + 1) * frame.rows / threadCount);

        // Constructing a std::thread STARTS processSlice with these arguments.
        workers.emplace_back(processSlice, &frame, &grayFrame, &sobelFrame,
                             startRow, endRow, &barrier);
    }

    // Create ALL workers before joining ANY worker. Workers run concurrently;
    // joining one does not stop the other workers from making progress.
    for (auto& worker : workers)
    {
        worker.join();
    }

    // Workers are finished, so the local buffers and barrier can safely expire.
    // cv::Mat manages the returned buffer's lifetime. grayFrame is released.
    return sobelFrame;
}
