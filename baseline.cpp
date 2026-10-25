/************************************************************
 * Single-threaded baseline, measured in a SEPARATE run.
 * Usage: ./baseline
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

int main()
{
    try
    {
        cv::setNumThreads(1);
        cv::VideoCapture video(VIDEO_PATH);
        if (!video.isOpened())
        {
            throw std::runtime_error("Could not open the video. Check settings.h.");
        }

        using Clock = std::chrono::steady_clock;
        double totalGrayTime = 0.0;
        double totalSobelTime = 0.0;
        double totalProcessingTime = 0.0;
        int frameCount = 0;
        cv::Mat frame;
        cv::Mat resizedFrame;

        while ((FRAME_LIMIT == 0 || frameCount < FRAME_LIMIT) 
                && video.read(frame) && !frame.empty())
        {
            cv::resize(frame, resizedFrame, cv::Size(FRAME_WIDTH, FRAME_HEIGHT));

            const auto grayStart = Clock::now();
            cv::Mat grayFrame = grayscale(resizedFrame);
            const auto grayEnd = Clock::now();
            cv::Mat sobelFrame = sobelFilter(grayFrame);
            const auto sobelEnd = Clock::now();

            totalGrayTime +=
                std::chrono::duration<double, std::milli>(grayEnd - grayStart).count();
            totalSobelTime +=
                std::chrono::duration<double, std::milli>(sobelEnd - grayEnd).count();
            totalProcessingTime +=
                std::chrono::duration<double, std::milli>(sobelEnd - grayStart).count();
            ++frameCount;

            if (DISPLAY_VIDEO)
            {
                cv::imshow("Single-threaded Sobel", sobelFrame);
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
                  << "Frames processed: " << frameCount << '\n'
                  << "Average grayscale time: " << totalGrayTime / frameCount << " ms\n"
                  << "Average Sobel time: " << totalSobelTime / frameCount << " ms\n"
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
