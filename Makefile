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
CFLAGS=$(BCFLAGS)  -I/usr/local/include/opencv  -Wall -Wno-sign-compare
LDFLAGS=-lpthread /usr/local/lib/libopencv_core.dylib /usr/local/lib/libopencv_highgui.dylib /usr/local/lib/libopencv_imgproc.dylib -lfftw3f
endif

SOURCES= gist.cpp main.cpp
OBJECTS=$(SOURCES:.cpp=.o)


EXECUTABLE=main

all: $(MINILZO) $(SOURCES) $(EXECUTABLE)


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


