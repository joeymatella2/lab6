#ifndef SETTINGS_H
#define SETTINGS_H

// Edit the video path here. A relative path starts from the terminal's
// current working directory, not the executable's directory.
constexpr const char* VIDEO_PATH = "../video/bbb_sunflower_2160p_60fps_normal.mp4";
constexpr int FRAME_WIDTH = 3840;
constexpr int FRAME_HEIGHT = 2160;

// Set false for runs without a display. This setting applies to all programs.
constexpr bool DISPLAY_VIDEO = true;
// How many frames to process before exiting. Set to 0 for no limit.
constexpr int FRAME_LIMIT = 2000;

#endif
