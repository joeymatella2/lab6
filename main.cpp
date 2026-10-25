/************************************************************
 * Multithreaded performance run only.
 * Usage: ./image_pipeline 4
 * Set the video path, resolution, and display in settings.h.
 ************************************************************/

#include "image_processing.h"
#include "settings.h"
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

int main(int argc, char* argv[])
{
    if (argc != 2)
    {
        std::cerr << "Usage: " << argv[0] << " THREAD_COUNT\n";
        return 1;
    }

    try
    {
        // Only the worker count is selected at runtime.
        std::string argument = argv[1];
        std::size_t used = 0;
        const int threadCount = std::stoi(argument, &used);
        if (threadCount < 1 || used != argument.size())
        {
            throw std::invalid_argument("Thread count must be a positive integer.");
        }

        cv::setNumThreads(1);  // Keep OpenCV's own operations single-threaded.
        cv::VideoCapture video(VIDEO_PATH);
        if (!video.isOpened())
        {
            throw std::runtime_error("Could not open the video. Check settings.h.");
        }

        using Clock = std::chrono::steady_clock;
        double totalProcessingTime = 0.0;
        int frameCount = 0;
        cv::Mat frame;
        cv::Mat resizedFrame;

        while ((FRAME_LIMIT == 0 ||frameCount < FRAME_LIMIT )
                && video.read(frame) && !frame.empty())
        {
            cv::resize(frame, resizedFrame, cv::Size(FRAME_WIDTH, FRAME_HEIGHT));

            // Include allocation, thread creation, processing, barrier, and
            // joins. Decoding, resizing, and display are outside the timer.
            const auto start = Clock::now();
            cv::Mat sobelFrame = processFrameMultithreaded(resizedFrame, threadCount);
            const auto end = Clock::now();

            totalProcessingTime +=
                std::chrono::duration<double, std::milli>(end - start).count();
            ++frameCount;

            if (DISPLAY_VIDEO)
            {
                cv::imshow("Multithreaded Sobel", sobelFrame);
                if (cv::waitKey(1) >= 0)
                {
                    break;
                }
            }
        }

        video.release();
        if (DISPLAY_VIDEO)
        {
            cv::destroyAllWindows();
        }
        if (frameCount == 0)
        {
            throw std::runtime_error("No frames were processed.");
        }

        std::cout << std::fixed << std::setprecision(3)
                  << "Worker threads: " << threadCount << '\n'
                  << "Frames processed: " << frameCount << '\n'
                  << "Average grayscale + Sobel time: "
                  << totalProcessingTime / frameCount << " ms\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
