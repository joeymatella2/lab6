/************************************************************
 * Correctness checks only. This is NOT a performance run.
 * Usage: ./validate 4
 ************************************************************/

#include "image_processing.h"
#include "settings.h"
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

// A Sobel center at input row y produces cropped output row y - 1.
// Check BOTH sides of each internal slice boundary, counting each row once.
long long boundaryDifferences(const cv::Mat& difference, int rows, int threads)
{
    std::vector<bool> boundaryRows(difference.rows, false);
    for (int i = 1; i < threads; ++i)
    {
        int split = static_cast<int>(static_cast<long long>(i) * rows / threads);
        for (int y = split - 1; y <= split; ++y)
        {
            if (y >= 1 && y < rows - 1)
            {
                boundaryRows[y - 1] = true;
            }
        }
    }

    long long count = 0;
    for (int y = 0; y < difference.rows; ++y)
    {
        if (boundaryRows[y])
        {
            count += cv::countNonZero(difference.row(y));
        }
    }
    return count;
}

int main(int argc, char* argv[])
{
    if (argc != 2)
    {
        std::cerr << "Usage: " << argv[0] << " THREAD_COUNT\n";
        return 1;
    }

    try
    {
        std::string argument = argv[1];
        std::size_t used = 0;
        const int threads = std::stoi(argument, &used);
        if (threads < 1 || used != argument.size())
        {
            throw std::invalid_argument("Thread count must be a positive integer.");
        }
        cv::setNumThreads(1);
        cv::VideoCapture video(VIDEO_PATH);
        if (!video.isOpened())
        {
            throw std::runtime_error("Could not open the video. Check settings.h.");
        }

        double maxDifference = 0.0;
        long long differingPixels = 0;
        long long boundaryPixels = 0;
        int frameCount = 0;
        cv::Mat frame;
        cv::Mat resizedFrame;

        while (video.read(frame) && !frame.empty())
        {
            cv::resize(frame, resizedFrame, cv::Size(FRAME_WIDTH, FRAME_HEIGHT));
            cv::Mat reference = sobelFilter(grayscale(resizedFrame));
            cv::Mat output = processFrameMultithreaded(resizedFrame, threads);
            cv::Mat difference;
            cv::absdiff(reference, output, difference);

            double frameMax = 0.0;
            cv::minMaxLoc(difference, nullptr, &frameMax);
            maxDifference = std::max(maxDifference, frameMax);
            differingPixels += cv::countNonZero(difference);
            boundaryPixels += boundaryDifferences(difference, resizedFrame.rows, threads);
            ++frameCount;

            if (DISPLAY_VIDEO)
            {
                cv::imshow("Baseline", reference);
                cv::imshow("Multithreaded", output);
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
        std::cout << "Worker threads: " << threads << '\n'
                  << "Frames checked: " << frameCount << '\n'
                  << "Maximum absolute pixel difference: " << maxDifference << '\n'
                  << "Differing pixels across all frames: " << differingPixels << '\n'
                  << "Differing pixels near slice boundaries: " << boundaryPixels << '\n';
        return differingPixels == 0 ? 0 : 2;
    }
    catch (const std::exception& error)
    {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
