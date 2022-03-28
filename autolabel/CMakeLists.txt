CMAKE_MINIMUM_REQUIRED(VERSION 3.1)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

SET(OpenCV_DIR /home/roberto/Documentos/installation/OpenCV-master/lib/cmake/opencv4)

PROJECT(autolabel)

FIND_PACKAGE(OpenCV REQUIRED)

SET(autolabel_SRC "autolabel.cpp")
 
ADD_EXECUTABLE(autolabel ${autolabel_SRC})

TARGET_LINK_LIBRARIES(autolabel ${OpenCV_LIBS})											 

