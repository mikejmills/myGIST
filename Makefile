#Execute the uname -s to get OS name
M_ARCH := $(shell uname -s)

#Name returned by OS X from uname -s
MAC_ARCH=Darwin

CC=g++
BCFLAGS=-c -g -D_REENTRANT -Wno-deprecated 
CFLAGS=$(BCFLAGS) `pkg-config opencv --cflags`  -Wall -O3
LDFLAGS=-lpthread `pkg-config opencv --libs`

#Check Arch Mac
ifeq ($(M_ARCH), $(MAC_ARCH))  
CFLAGS=$(BCFLAGS)  -I/usr/local/include/opencv -I/usr/include/python2.7/ -I/Library/Python/2.7/site-packages/numpy/core/include -I/usr/local/include/  -Wall -Wno-sign-compare -g
LDFLAGS=-lpthread /usr/local/lib/libopencv_core.dylib /usr/local/lib/libopencv_highgui.dylib  /usr/local/lib/libopencv_imgproc.dylib -L/Library/Python/2.7/site-packages/numpy/core/include/numpy/ -L/usr/local/lib/ -lfftw3f -lpython2.7
endif

SOURCES= gist.cpp main.cpp PCA.cpp ../GISTHoming/alglib/*.o
OBJECTS=$(SOURCES:.cpp=.o)


EXECUTABLE=main

all: $(MINILZO) $(SOURCES) $(EXECUTABLE)
python: gist.o PCA.o gist_Python.cpp
	@echo CC LINK libgist
	@g++ $(CFLAGS) -fPIC gist_Python.cpp   -o libgist.o
	@g++ $(LDFLAGS) -shared -Wl libgist.o gist.o PCA.o -o libgist.so 

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


