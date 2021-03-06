#Execute the uname -s to get OS name
M_ARCH := $(shell uname -s)

#Name returned by OS X from uname -s
MAC_ARCH=Darwin

CC=g++
BCFLAGS=-c -g -D_REENTRANT -Wno-deprecated 
CFLAGS=$(BCFLAGS) `pkg-config opencv --cflags`  `pkg-config python --cflags` -Wall
LDFLAGS=-lpthread `pkg-config opencv --libs`  -L/usr/lib/python2.7/ `pkg-config python --libs`

#CFLAGS=$(BCFLAGS) -I/home/mike/OpenCV/build/include/opencv/ -I/home/mike/OpenCV/include/build/opencv/opencv2/ -I/home/mike/OpenCV/build/include/  `pkg-config python --cflags` -Wall
#LDFLAGS=-lpthread -L/home/mike/OpenCV/build/lib -lopencv_contrib -lopencv_legacy -lopencv_objdetect -lopencv_calib3d -lopencv_features2d -lopencv_video -lopencv_highgui -lopencv_ml -lopencv_imgproc -lopencv_flann -lopencv_core  -L/usr/lib/python2.7/ `pkg-config python --libs`

#Check Arch Mac
ifeq ($(M_ARCH), $(MAC_ARCH))  
	CFLAGS=$(BCFLAGS)  -I/usr/local/include/opencv/ -I/usr/include/python2.7/ -I/Library/Python/2.7/site-packages/numpy/core/include -I/usr/local/include/  -Wall -Wno-sign-compare -g
	LDFLAGS=-lpthread `pkg-config opencv --libs` -L/System/Library/Frameworks/Python.framework/Versions/2.7/Extras/lib/python/numpy/ -L/usr/local/lib/ -lpython2.7
endif

SOURCES= gist.cpp main.cpp PCA.cpp ../GISTHoming/alglib/*.o
OBJECTS=$(SOURCES:.cpp=.o)


EXECUTABLE=main

all: $(SOURCES) $(EXECUTABLE)
#python: gist.o PCA.o gist_Python.cpp
#	@echo CC LINK libgist
#	@g++ $(CFLAGS) -fPIC gist_Python.cpp   -o libgist.o
#	@g++ $(LDFLAGS) -msse -msse2 -mfpmath=sse4 -shared -Wl libgist.o gist.o PCA.o -o libgist.so 

python: python_gist.cpp python_gist.h
	@echo CC python_gist.cpp
	@g++ $(CFLAGS) -fPIC python_gist.cpp -o libgist.o
	@g++ -shared libgist.o -o libgist.so $(LDFLAGS) 	# -ffast-math

FAST: FAST_GPU.cpp
	@echo CC FAST_GPU.cpp
	@g++ $(CFLAGS) -c FAST_GPU.cpp
	@echo LD FAST_GPU.cpp
	@g++ $(LDFLAGS) FAST_GPU.o -o fast_test

$(EXECUTABLE): $(OBJECTS)
	@echo LINKING $(EXECUTABLE)
	@$(CC) $(LDFLAGS) $(OBJECTS) -o $@ 

.cpp.o:
	@echo CC $(BUILD)$@
	@$(CC) $(CFLAGS) $< -o $@


clean:
	@echo CLEAN
	@rm -f $(BUILD)*.o $(BUILD)*.a $(EXECUTABLE)
	@rm -f ./BRIEF/*.o


