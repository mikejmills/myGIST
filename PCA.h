#ifndef PCA_HOMING
#define PCA_HOMING

#include <stdlib.h>
#include "../myGIST/gist.h"
#include <opencv2/highgui/highgui.hpp>
#include <cv.hpp>
#include <stdio.h>

#define PCA_DIM 10

void     PCA_BuildSave(char *image_dir, int blocks);
cv::PCA *PCA_LoadData(int blocks);
void PCA_Free(cv::PCA *p);

#endif