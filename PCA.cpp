#include "PCA.h"


void PCA_BuildSave(char *image_dir, int blocks)
{
	char path[40];
	int fcount = 30, descsize;
	double *desc;
	cv::Mat cinput, coutput;

	sprintf(path, "%s/image1", image_dir);
	cinput = cv::imread(path);
	
	format_image(cinput, coutput);
	
	Gist_Processor proc(coutput, blocks);

	cv::Mat PCAm(fcount, proc.base_descsize, CV_64FC1);

	for (int i=0; i < fcount; i++) {

		sprintf(path, "%s/image%d", image_dir, i);
		printf("Image %d\n", i);
		cinput = cv::imread(path);
	
		format_image(cinput, coutput);
		proc.Process(coutput);

		descsize = proc.Get_Descriptor(&desc,blocks);
		
		
		for (int d=0; d < descsize; d++) {
			//((double *)Row.data)[d] = desc[d];
			PCAm.at<double>(i,d) = desc[d];
			
		}

		free(desc);
		
	}
	
	cv::PCA pca_obj(PCAm, cv::Mat(), CV_PCA_DATA_AS_ROW, PCA_DIM);
	
	printf("type %d %d %d\n", pca_obj.eigenvectors.type(), CV_64FC1, CV_64FC1);
	
	sprintf(path, "./PCAeigenvectors%d.mat", blocks);

	FILE *fd = fopen(path, "w+");
	
	if (!fd) {
			printf("%s\n", path);
            perror("Error opening file for loading %s\n");
            return;
    }

	fprintf(fd, "%d\n%d\n", pca_obj.eigenvectors.rows, pca_obj.eigenvectors.cols);

	for (int j=0; j < pca_obj.eigenvectors.rows; j++) {
		for (int i=0; i < pca_obj.eigenvectors.cols; i++) {
			fprintf(fd, "%f\n", pca_obj.eigenvectors.at<double>(j,i));
		}
	}
	fclose(fd);


	sprintf(path, "./PCAmean%d.mat", blocks);
	fd = fopen(path, "w+");
	if (!fd) {
            perror("Error opening file for loading\n");
            return;
    }	
	fprintf(fd, "%d\n%d\n", pca_obj.mean.rows, pca_obj.mean.cols);	

	for (int j=0; j < pca_obj.mean.rows; j++) {
		for (int i=0; i < pca_obj.mean.cols; i++) {
			fprintf(fd, "%f\n", pca_obj.mean.at<double>(j,i));
		}
	}
	fclose(fd);
	
}

cv::PCA *PCA_LoadData(int blocks)
{
	char path[40];
	
	int rows, cols;

	sprintf(path, "./PCAeigenvectors%d.mat", blocks);

	FILE *fd = fopen(path, "r+");
	
	if (!fd) {

            perror("Error opening file for loading\n");
            return NULL;
    }

	fscanf(fd, "%d", &rows);
	fscanf(fd, "%d", &cols);
	cv::Mat eigenvectors(rows, cols, CV_64FC1);

	for (int j=0; j < eigenvectors.rows; j++) {
		for (int i=0; i < eigenvectors.cols; i++) {
			
			fscanf(fd, "%lf", &(eigenvectors.at<double>(j,i)));
		}
	}
	fclose(fd);


	sprintf(path, "./PCAmean%d.mat", blocks);
	fd = fopen(path, "r+");

	if (!fd) {
			printf("blah %s\n", path);
            perror("Error opening file for loading huh\n");
            return NULL;
    }	
	
	
	fscanf(fd, "%d", &rows);
	fscanf(fd, "%d", &cols);
	cv::Mat mean(rows, cols, CV_64FC1);
	
	for (int j=0; j < mean.rows; j++) {
		for (int i=0; i < mean.cols; i++) {
			fscanf(fd, "%lf", &(mean.at<double>(j,i)));
		}
	}
	
	fclose(fd);



	cv::PCA *pca_obj = new cv::PCA();
	
	pca_obj->eigenvectors = eigenvectors;
	pca_obj->mean         = mean;
	
	return pca_obj;
}

void PCA_Free(cv::PCA *p)
{
	delete p;
}
