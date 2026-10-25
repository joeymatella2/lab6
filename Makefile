CXX = g++
PKG_CONFIG = pkg-config
CXXFLAGS = -std=c++11 -O3 -ftree-vectorize -Wall -Wextra -Wpedantic -pthread
OPENCV_CFLAGS = $(shell $(PKG_CONFIG) --cflags opencv4)
OPENCV_LIBS = $(shell $(PKG_CONFIG) --libs opencv4)

.PHONY: all clean
all: image_pipeline baseline

image_pipeline: main.cpp image_processing.cpp image_processing.h settings.h
	$(CXX) $(CXXFLAGS) $(OPENCV_CFLAGS) main.cpp image_processing.cpp -o $@ $(OPENCV_LIBS)

baseline: baseline.cpp image_processing.cpp image_processing.h settings.h
	$(CXX) $(CXXFLAGS) $(OPENCV_CFLAGS) baseline.cpp image_processing.cpp -o $@ $(OPENCV_LIBS)

# Build this separately when collecting the lab's correctness evidence.
validate: validate.cpp image_processing.cpp image_processing.h settings.h
	$(CXX) $(CXXFLAGS) $(OPENCV_CFLAGS) validate.cpp image_processing.cpp -o $@ $(OPENCV_LIBS)

clean:
	rm -f image_pipeline baseline validate
