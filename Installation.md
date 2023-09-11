# OpenCV Version
This will build with the most recent version of opencv, but in order to open large vocabularies, I needed to use a 3.X version. This can be installed alonglisde a system Opencv install with the following
1. Download the OpenCV version
2. Build it with
    - cd opencv-3x; mkdir build; cd build
    - `cmake -DCMAKE_INSTALL_PREFIX=install ..` this puts the `install` dir in (relative to) the build `folder`. You can do an absolute path
    - `cmake --build . --target install` builds and installs in the target install path
3. You can point DBoW2 to the correct version by changing `find_package(OpenCV REQUIRED)` to 
`find_package(OpenCV 3.4 REQUIRED PATHS /home/colin/Software/opencv-3.4.16/build/install)` in CMAKE