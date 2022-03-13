#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>
#include "FreeImage.h"
#include <math.h>
#include <omp.h>
#include <string.h>


#define MAX_SOURCE_SIZE	16384

#define PATH_MAX 256
#define INPUT "/ceph/grid/home/ag0001/Stiskanje-Slike-PS/images/jpg/"
#define OUTPUT "images/"

int K = 256;
int ITTERATIONS = 20;
int WORKGROUP_SIZE = 256;

unsigned char *imageIn;     //pointer to image
unsigned char *centroids;   //pointer to centroids array
int imageSize = 0;          //size in bytes
int width = 0;
int height = 0;
int pitch = 0;

void initCentroids()
{
    centroids = (unsigned char *) malloc( K * 3 * sizeof(unsigned char));
    for (int i = 0; i < K; i++)
    {
        int index = rand() % (width * height);
        centroids[i * 3]        = imageIn[index * 4];
        centroids[i * 3 + 1]    = imageIn[index * 4 + 1];
        centroids[i * 3 + 2]    = imageIn[index * 4 + 2];
    }
}
void readImage()
{
    char inputPath[PATH_MAX];
    strcpy(inputPath, INPUT);
    strcat(inputPath, "bird.jpg"); // TODO change to arg

    // TODO K and INTERATIONS as args

    //printf("Loading image %s\n", inputPath);

    FIBITMAP *imageBitmap = FreeImage_Load(FIF_JPEG, inputPath, 0);
    //Convert it to a 32-bit image
    FIBITMAP *imageBitmap32 = FreeImage_ConvertTo32Bits(imageBitmap);

    width = FreeImage_GetWidth(imageBitmap32);
    height = FreeImage_GetHeight(imageBitmap32);
    pitch = FreeImage_GetPitch(imageBitmap32);
    imageSize = height * pitch;

    //Preapare room for a raw data copy of the image
    imageIn = (unsigned char *)malloc(height * pitch * sizeof(unsigned char));
    FreeImage_ConvertToRawBits(imageIn, imageBitmap, pitch, 32,
            FI_RGBA_RED_MASK, FI_RGBA_GREEN_MASK, FI_RGBA_BLUE_MASK, TRUE);
}
int main(int argc, char* argv[])
{
    readImage();
    initCentroids();
    K = atoi(argv[1]);
    ITTERATIONS = atoi(argv[2]);
    WORKGROUP_SIZE = atoi(argv[3]);

    FILE *fp;
    char *source_str;
    size_t source_size;

    char ch;
    int i;
	cl_int ret;

	//branje "s"cepca
    fp = fopen("kernel.cl", "r");
    if (!fp)
	{
		fprintf(stderr, ":-(#\n");
        exit(1);
    }
    source_str = (char*)malloc(MAX_SOURCE_SIZE);
    source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
	source_str[source_size] = '\0';
    fclose( fp );


    // Pridobi podatke o platformi
    cl_platform_id	platform_id[10];
    cl_uint			ret_num_platforms;
	char			*buf;
	size_t			buf_len;
	ret = clGetPlatformIDs(10, platform_id, &ret_num_platforms);
	// max. "stevilo platform, kazalec na platforme, dejansko "stevilo platform

	// Podatki o napravi
	cl_device_id	device_id[10];
	cl_uint			ret_num_devices;
	// Delali bomo s platform_id[0] na GPU
	ret = clGetDeviceIDs(platform_id[0], CL_DEVICE_TYPE_GPU, 10,
						 device_id, &ret_num_devices);
			// izbrana platforma, tip naprave, koliko naprav nas zanima
			// kazalec na naprave, dejansko "stevilo naprav

    // Kontekst
    cl_context context = clCreateContext(NULL, 1, &device_id[0], NULL, NULL, &ret);
			// kontekst: vklju"cene platforme - NULL je privzeta, "stevilo naprav,
			// kazalci na naprave, kazalec na call-back funkcijo v primeru napake
			// dodatni parametri funkcije, "stevilka napake

    // Ukazna vrsta
    cl_command_queue command_queue = clCreateCommandQueue(context, device_id[0], 0, &ret);
			// kontekst, naprava, INORDER/OUTOFORDER, napake

	// Delitev dela 2D nacin
	//size_t local_item_size[2] = {WORKGROUP_SIZE, WORKGROUP_SIZE};
	//size_t num_groups[2] = {((height - 1) / local_item_size[0] + 1), ((width - 1) / local_item_size[1] + 1)};
	//size_t global_item_size[2] = {num_groups[0] * local_item_size[0], num_groups[1] * local_item_size[1]};

    size_t local_item_size = WORKGROUP_SIZE;
	size_t num_groups = (height * width - 1) / (local_item_size + 1);
	size_t global_item_size = num_groups * local_item_size;

    // Alokacija pomnilnika na napravi
	// to ni vec potrebno
	cl_mem image_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, width * height * sizeof(unsigned int), imageIn, &ret);
    cl_mem centroids_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, K * 3 * sizeof(unsigned char), centroids, &ret);
    cl_mem centroids_sums = clCreateBuffer(context, CL_MEM_READ_WRITE, K * 3 * sizeof(unsigned int), NULL, &ret);
    cl_mem centroids_popularity = clCreateBuffer(context, CL_MEM_READ_WRITE, K * sizeof(unsigned int), NULL, &ret);



    // Priprava programa
    cl_program program = clCreateProgramWithSource(context,	1, (const char **)&source_str,
												   NULL, &ret);
			// kontekst, "stevilo kazalcev na kodo, kazalci na kodo,
			// stringi so NULL terminated, napaka

    // Prevajanje
    ret = clBuildProgram(program, 1, &device_id[0], NULL, NULL, NULL);
			// program, "stevilo naprav, lista naprav, opcije pri prevajanju,
			// kazalec na funkcijo, uporabni"ski argumenti

	// Log
	size_t build_log_len;
	char *build_log;
	ret = clGetProgramBuildInfo(program, device_id[0], CL_PROGRAM_BUILD_LOG,
								0, NULL, &build_log_len);
			// program, "naprava, tip izpisa,
			// maksimalna dol"zina niza, kazalec na niz, dejanska dol"zina niza
	build_log =(char *) malloc (sizeof(char)*(build_log_len+1));
	ret = clGetProgramBuildInfo(program, device_id[0], CL_PROGRAM_BUILD_LOG,
							    build_log_len, build_log, NULL);
	//printf("%s\n", build_log);
	free(build_log);
	if(build_log_len > 2)
		return 1;

    // "s"cepec: priprava objekta
    cl_kernel kernel = clCreateKernel(program, "kMeansAlgorithm", &ret);
			// program, ime "s"cepca, napaka

    // "s"cepec: argumenti
    ret  = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&image_mem_obj);
    ret |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&centroids_mem_obj);
    ret |= clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&centroids_sums);
    ret |= clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&centroids_popularity);
    ret |= clSetKernelArg(kernel, 4, K * 3 * sizeof(unsigned int), NULL);	
    ret |= clSetKernelArg(kernel, 5, K * sizeof(unsigned int), NULL);		
    ret |= clSetKernelArg(kernel, 6, sizeof(cl_int), (void *)&height);
    ret |= clSetKernelArg(kernel, 7, sizeof(cl_int), (void *)&width);
    ret |= clSetKernelArg(kernel, 8, sizeof(cl_int), (void *)&K);
    ret |= clSetKernelArg(kernel, 9, sizeof(cl_int), (void *)&ITTERATIONS);
    
			// "s"cepec, "stevilka argumenta, velikost podatkov, kazalec na podatke
	double dt = omp_get_wtime();
	// "s"cepec: zagon
    ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL,
								 &global_item_size, &local_item_size, 0, NULL, NULL);
			// vrsta, "s"cepec, dimenzionalnost, mora biti NULL,
			// kazalec na "stevilo vseh niti, kazalec na lokalno "stevilo niti,
			// dogodki, ki se morajo zgoditi pred klicem

	// Kopiranje rezultatov
    ret = clEnqueueReadBuffer(command_queue, image_mem_obj, CL_TRUE, 0,
							  4 * width * height * sizeof(unsigned char), imageIn, 0, NULL, NULL);
			// branje v pomnilnik iz naparave, 0 = offset
			// zadnji trije - dogodki, ki se morajo zgoditi prej
    dt = omp_get_wtime() - dt;
	//printf("Cas: %lf\n", dt);
    pitch = ((32 * width + 31) / 32) * 4;

    printf("OCL\tO2\t%d\t%d\t%d\t%.0f\n", K, ITTERATIONS, WORKGROUP_SIZE, dt * 1000);


    FIBITMAP *dst = FreeImage_ConvertFromRawBits(imageIn, width, height, pitch,
		32, FI_RGBA_RED_MASK, FI_RGBA_GREEN_MASK, FI_RGBA_BLUE_MASK, TRUE);

    char outputPath[PATH_MAX];
    sprintf(outputPath, "%s%s_K%d_IT%d.png", OUTPUT, "stisnjena", K, ITTERATIONS);
	FreeImage_Save(FIF_PNG, dst, "output.png", 0);

	//printHistogram(H);
    // "ci"s"cenje
    ret = clFlush(command_queue);
    ret = clFinish(command_queue);
    ret = clReleaseKernel(kernel);
    ret = clReleaseProgram(program);
    ret = clReleaseMemObject(image_mem_obj);
    ret = clReleaseMemObject(centroids_mem_obj);
    ret = clReleaseMemObject(centroids_sums);
    ret = clReleaseMemObject(centroids_popularity);
    
    ret = clReleaseCommandQueue(command_queue);
    ret = clReleaseContext(context);

    free(imageIn);
    free(centroids);

    return 0;
}
