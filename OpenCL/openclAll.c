/**
 * @file OpenMP.c
 * @author FallingLights (fallinglights@protonmail.com)
 * @brief
 * @version 0.1
 * @date 2022-02-12
 *
 * Uporabil sem schedule(runtime) moreš nastaviti $ export OMP_SCHEDULE=static
 * @copyright
 * @compile gcc -Wall -o openclAll openclAll.c -lm -lOpenCL -fopenmp -O2
 * @run srun -n1 -G1 --reservation=fri openclAll -k 30 -i 20 -w 128 -s 123456 -o out.png /ceph/grid/home/ag0001/Stiskanje-Slike-PS/images/png/bled.png
 */

/**
 * 
 * https://hpi.de/fileadmin/user_upload/fachgebiete/rabl/publications/2018/KMeansDatenbankspektrum.pdf
 * https://github.com/muncok/Kmeans-OpenCL
 * https://github.com/ronak197/Parallel-K-means-Clustering
 * https://github.com/LaurabelleKay/Parallel-kMeans/tree/master/OpenCL
 * https://bitbucket.org/malthejorgensen/kmeans-gpu-nbi/src/master/kernels/
 * https://repo.infor.uva.es/ismael/rodinia/-/tree/master/opencl/kmeans
 * http://www.goldsborough.me/c++/python/cuda/2017/09/10/20-32-46-exploring_k-means_in_python,_c++_and_cuda/
 */
#include <errno.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
//#include <sys/time.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>
#include <omp.h>
#include <CL/cl.h>

#define STB_IMAGE_IMPLEMENTATION
#include "include/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "include/stb_image_write.h"

#define N_THREADS (2)
#define WORKGROUP_SIZE 128
#define MAX_SOURCE_SIZE 16384

typedef unsigned char byte_t;

int seed;

// global variables for the device and program
//void *deviceID;
//void *deviceCtx;
cl_command_queue command_queue;
//void *program;


byte_t *img_load(char *img_file, int *width, int *height, int *n_channels)
{
    byte_t *data;

    data = stbi_load(img_file, width, height, n_channels, 0);
    if (data == NULL) {
        fprintf(stderr, "ERROR LOADING IMAGE: << Invalid file name or format >> \n");
        exit(EXIT_FAILURE);
    }

    return data;
}

void img_save(char *img_file, byte_t *data, int width, int height, int n_channels)
{
    char *ext;

    ext = strrchr(img_file, '.');

    if (!ext) {
        fprintf(stderr, "ERROR SAVING IMAGE: << Unspecified format >> \n\n");
        return;
    }

    printf("Savin Image...\n");

    if ((strcmp(ext, ".jpeg") == 0) || (strcmp(ext, ".jpg") == 0)) {
        stbi_write_jpg(img_file, width, height, n_channels, data, 100);
    } else if (strcmp(ext, ".png") == 0) {
        stbi_write_png(img_file, width, height, n_channels, data, width * n_channels);
    } else if (strcmp(ext, ".bmp") == 0) {
        stbi_write_bmp(img_file, width, height, n_channels, data);
    } else if (strcmp(ext, ".tga") == 0) {
        stbi_write_tga(img_file, width, height, n_channels, data);
    } else {
        fprintf(stderr, "ERROR SAVING IMAGE: << Unsupported format >> \n\n");
    }
}

int setBufferData(cl_mem buf, size_t size, void *data)
{
	cl_event event;
	cl_int ret = clEnqueueWriteBuffer(command_queue, buf, 1, 0, size, data, 0, NULL, &event);
	ret |= clWaitForEvents(1, &event);
	ret |= clReleaseEvent(event);
	return ret;
}

int getBufferData(cl_mem buf, size_t size, void *ptr)
{
	cl_event event;
	cl_int ret = clEnqueueReadBuffer(command_queue, buf, 1, 0, size, ptr, 0, NULL, &event);
	ret |= clWaitForEvents(1, &event);
	ret |= clReleaseEvent(event);
	return ret;
}

void init_centers(byte_t *data, double *centers, int n_px, int n_ch, int n_clus)
{
    int k, ch, rnd;

    #pragma omp parallel
    {
        srand((int)(seed) ^ omp_get_thread_num());
        #pragma omp for private(k, ch, rnd)
        for (k = 0; k < n_clus; k++) {
            rnd = rand() % n_px;
            for (ch = 0; ch < n_ch; ch++) {
                centers[k * n_ch + ch] = data[rnd * n_ch + ch];
            }
        }
    }
}


int label_pixels_CL(void *labelPixelsKernel, 
                    cl_mem data, 
                    cl_mem centers, 
                    cl_mem labels, 
                    cl_mem dists, 
                    cl_mem changed, 
                    cl_int n_px, 
                    cl_int n_ch, 
                    cl_int n_clus,
                    int workgroupsize
) {
    size_t local_item_size = workgroupsize;
	size_t num_groups = n_px / local_item_size;
	size_t global_item_size = num_groups * local_item_size;

    // ščepec: argumenti
    cl_int ret;
    ret = clSetKernelArg(labelPixelsKernel, 0, sizeof(cl_mem), (void *)&data);
    ret |= clSetKernelArg(labelPixelsKernel, 1, sizeof(cl_mem), (void*)&centers);
    ret |= clSetKernelArg(labelPixelsKernel, 2, sizeof(cl_mem), (void*)&labels);
    ret |= clSetKernelArg(labelPixelsKernel, 3, sizeof(cl_mem), (void*)&dists);
    //ret |= clSetKernelArg(labelPixelsKernel, 4, sizeof(cl_mem), (void*)&changed);

    ret |= clSetKernelArg(labelPixelsKernel, 4, sizeof(cl_int), (void*)&n_px);
    ret |= clSetKernelArg(labelPixelsKernel, 5, sizeof(cl_int), (void*)&n_ch);
    ret |= clSetKernelArg(labelPixelsKernel, 6, sizeof(cl_int), (void*)&n_clus);
    if (ret != 0)
		return ret;
    cl_event event;
	size_t globalSize = n_px;
	ret |= clEnqueueNDRangeKernel(command_queue, labelPixelsKernel, 1, 0, &global_item_size, &local_item_size, 0, NULL, &event);
	ret |= clWaitForEvents(1, &event);
	ret |= clReleaseEvent(event);
	return ret;
}

int update_centers_CL(void *update_centersKernel,
                      cl_mem data, 
                      cl_mem centers, 
                      cl_mem labels, 
                      cl_mem dists,
                      cl_mem global_sum,
                      cl_mem global_count,
                      int n_px, 
                      int n_ch, 
                      int n_clus) 
{
    // ščepec: argumenti
    cl_int ret;
	ret = clSetKernelArg(update_centersKernel, 0, sizeof(cl_mem), (void *)&data); //data
    ret |= clSetKernelArg(update_centersKernel, 1, sizeof(cl_mem), (void*)&centers); //centers
    ret |= clSetKernelArg(update_centersKernel, 2, sizeof(cl_mem), (void*)&labels); //labels
    ret |= clSetKernelArg(update_centersKernel, 3, sizeof(cl_mem), (void*)&dists); //dists
    ret |= clSetKernelArg(update_centersKernel, 4, sizeof(cl_mem), (void*)&global_sum); //global_sum
    ret |= clSetKernelArg(update_centersKernel, 5, sizeof(cl_mem), (void*)&global_count); //global_count
    ret |= clSetKernelArg(update_centersKernel, 6, n_px * sizeof(double), NULL); //local_sum
    ret |= clSetKernelArg(update_centersKernel, 7, (n_clus + 1) * sizeof(cl_uint), NULL); //local_count

    ret |= clSetKernelArg(update_centersKernel, 8, sizeof(cl_int), (void*)&n_px); //n_px
    ret |= clSetKernelArg(update_centersKernel, 9, sizeof(cl_int), (void*)&n_ch); //n_ch
    ret |= clSetKernelArg(update_centersKernel, 10, sizeof(cl_int), (void*)&n_clus); //n_clus
	if (ret != 0)
		return ret;
	int window = 8;
	size_t globalSize = (n_px + 1) / window;
	size_t localSize = n_clus;
	cl_event event;
	ret = clEnqueueNDRangeKernel(command_queue, update_centersKernel, 1,	// work_dim
					 0,	// global_work_offset
					 &globalSize,	// global_work_size
					 &localSize,	// local_work_size
					 0,	// num_events_in_wait_list
					 NULL,	// event_wait_list
					 &event	//event
	    );
	ret |= clWaitForEvents(1, &event);
	ret |= clReleaseEvent(event);
	if (ret != 0)
		return ret;
}

void update_centers(byte_t *data, double *centers, int *labels, double *dists, int n_px, int n_ch, int n_clus)
{
    int px, ch, k;
    int *counts;
    int min_k, far_px;
    double max_dist;

    counts = malloc(n_clus * sizeof(int));
	//printf("start\n");

    //Resetting arrays
    for (k = 0; k < n_clus; k++) {
        for (ch = 0; ch < n_ch; ch++) {
            centers[k * n_ch + ch] = 0;
        }

        counts[k] = 0;
    }

    //Calculating sums and updating cluster counters
    #pragma omp parallel for private(px, ch, min_k) reduction(+:centers[:n_clus * n_ch],counts[:n_clus])
    for (px = 0; px < n_px; px++) {
        min_k = labels[px];

        for (ch = 0; ch < n_ch; ch++) {
            centers[min_k * n_ch + ch] += data[px * n_ch + ch];
        }

        counts[min_k]++;
    }

    //means
    #pragma omp parallel for private(px, ch, min_k, k)
    for (k = 0; k < n_clus; k++) {
        if (counts[k]) {
            for (ch = 0; ch < n_ch; ch++) {
                centers[k * n_ch + ch] /= counts[k];
            }
        } else {

            max_dist = 0;

            for (px = 0; px < n_px; px++) {
                if (dists[px] > max_dist) {
                    max_dist = dists[px];
                    far_px = px;
                }
            }

            for (ch = 0; ch < n_ch; ch++) {
                centers[k * n_ch + ch] = data[far_px * n_ch + ch];
            }

            dists[far_px] = 0;
        }
    }

    free(counts);
}

void update_image(byte_t *data, double *centers, int *labels, int n_px, int n_ch)
{
    int px, ch, min_k;

    for (px = 0; px < n_px; px++) {
        min_k = labels[px];

        for (ch = 0; ch < n_ch; ch++) {
            data[px * n_ch + ch] = (byte_t)round(centers[min_k * n_ch + ch]);
        }
    }
}

void kmeans(byte_t *data, int width, int height, int n_channels, int n_clus, int *n_iters, int n_threads, int workgroupsize)
{
    int n_px;
    int iter, max_iters;
    int *changes;
    int *labels;
    double *centers;
    double *dists;
    omp_set_num_threads(n_threads);

    max_iters = *n_iters;

    n_px = width * height;

    //Initialize Arrays
    labels = malloc(n_px * sizeof(int));
    centers = malloc(n_clus * n_channels * sizeof(double));
    dists = malloc(n_px * sizeof(double));

    //Randomly set centers
    init_centers(data, centers, n_px, n_channels, n_clus);

    //READ KERNEL
    FILE *fp;
    char *source_str;
    size_t source_size;
    fp = fopen("kernel.cl", "r");
    if (!fp) {
            fprintf(stderr, "Failed to read kernel\n");
            exit(1);
    }
    source_str = (char*)malloc(MAX_SOURCE_SIZE);
    source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
    source_str[source_size] = '\0';
    fclose( fp );


    size_t local_item_size = workgroupsize;
	size_t num_groups = n_px / local_item_size;
	size_t global_item_size = num_groups * local_item_size;

    // Pridobi podatke o platformi
    cl_int ret;
	cl_platform_id	platform_id = NULL;
    cl_uint			ret_num_platforms;

	ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
    if (ret != CL_SUCCESS)
    {
        printf("clGetPlatformIDs Error: %d\n", ret);
        exit(1);
    }

    // Pridobi podatke o napravi
	cl_device_id	device_id = NULL;
	cl_uint			ret_num_devices;
	ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, &ret_num_devices);
    if (ret != CL_SUCCESS)
    {
        printf("clGetDeviceIDs Error: %d\n", ret);
        exit(1);
    }

    // Kontekst
	cl_context context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);
    if (ret != CL_SUCCESS)
    {
        printf("CreateContext Error: %d\n", ret);
        exit(1);
    }
    // Ukazna vrsta
	command_queue = clCreateCommandQueue(context, device_id, 0, &ret);
    if (ret != CL_SUCCESS)
    {
        printf("clCreateCommandQueue Error: %d\n", ret);
        exit(1);
    }

    // Alokacija pomnilnika na napravi
    cl_mem gpu_data = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, n_px * n_channels, data, &ret);
    cl_mem gpu_centers = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, n_clus * n_channels * sizeof(double), centers, &ret);
    cl_mem gpu_labels = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, n_px * sizeof(int), labels, &ret);
    cl_mem gpu_dists = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, n_px * sizeof(double), dists, &ret);
    if (!gpu_data || !gpu_centers || !gpu_labels || !gpu_dists)
    {
        printf("Failed to allocate memory\n");
        exit(1);
    }
    cl_mem global_sum = clCreateBuffer(context, CL_MEM_READ_WRITE, n_px * sizeof(cl_ulong), NULL, &ret);
    cl_mem global_count = clCreateBuffer(context, CL_MEM_READ_WRITE, n_px * sizeof(cl_ulong), NULL, &ret);
    cl_int changed = 0;
    cl_mem changedBuf = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_int), NULL, &ret);
    if (!global_sum || !global_count || !changedBuf)
    {
        printf("Failed to allocate memory\n");
        exit(1);
    }

    // Priprava programa
    cl_program program = clCreateProgramWithSource(context,	1, (const char **)&source_str, (const size_t *)&source_size, &ret);
    if (ret != CL_SUCCESS)
    {
        printf("CreateProgram Error %d\n", ret);
        exit(1);
    }

    // Prevajanje
	cl_int retBuild = clBuildProgram(program, 1, &device_id, "-cl-nv-verbose", NULL, NULL);

    // Log 
	size_t build_log_len;
	char *build_log;
	ret = clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &build_log_len);
	build_log =(char *)malloc(sizeof(char)*(build_log_len+1));
	ret = clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, build_log_len, build_log, NULL);
	printf("%s\n", build_log);
	free(build_log);
    if (retBuild != CL_SUCCESS)
    {
        printf("BuildProgram Error %d\n", ret);
        exit(1);
    }

    // ščepec: priprava objekta
    cl_kernel labelPixelsKernel = clCreateKernel(program, "labelPixels", &ret);
    if (ret != CL_SUCCESS)
    {
        printf("CreateKernel Error %d\n", ret);
        exit(1);
    }
	size_t buf_size_t;
	clGetKernelWorkGroupInfo(labelPixelsKernel, device_id, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,sizeof(buf_size_t), &buf_size_t, NULL);
    
    cl_kernel updateCentersKernel = clCreateKernel(program, "update_centers", &ret);
    if (ret != CL_SUCCESS)
    {
        printf("CreateKernel Error %d\n", ret);
        exit(1);
    }

    /*ret = clSetKernelArg(labelPixelsKernel, 0, sizeof(cl_mem), (void *)&gpu_data);
    ret |= clSetKernelArg(labelPixelsKernel, 1, sizeof(cl_mem), (void*)&gpu_centers);
    ret |= clSetKernelArg(labelPixelsKernel, 2, sizeof(cl_mem), (void*)&gpu_labels);
    ret |= clSetKernelArg(labelPixelsKernel, 3, sizeof(cl_mem), (void*)&gpu_dists);
    ret |= clSetKernelArg(labelPixelsKernel, 4, sizeof(cl_int), (void*)&n_px);
    ret |= clSetKernelArg(labelPixelsKernel, 5, sizeof(cl_int), (void*)&n_channels);
    ret |= clSetKernelArg(labelPixelsKernel, 6, sizeof(cl_int), (void*)&n_clus);*/

    //Training
    double labeling = 0;
    double updating = 0;
    printf("Training...\n");
    for (iter = 0; iter < max_iters; iter++) {
	    
        double dt;
        dt = omp_get_wtime();

        changed = 0;
		ret = setBufferData(changedBuf, sizeof(cl_int), (void *)(&changed));
        ret = label_pixels_CL(labelPixelsKernel, gpu_data, gpu_centers, gpu_labels, gpu_dists, changedBuf, n_px, n_channels, n_clus, workgroupsize);
        ret = getBufferData(changedBuf, sizeof(cl_int), (void *)(&changed));
        //ret = clEnqueueNDRangeKernel(command_queue, labelPixelsKernel, 1, NULL, &global_item_size, &local_item_size, 0, NULL, NULL);
        /*if (changed == 0)
			break;*/
		ret = update_centers_CL(updateCentersKernel, gpu_data, gpu_centers, gpu_labels, gpu_dists, global_sum, global_count, n_px, n_channels, n_clus);
        
        // !  OpenMP
        /*
        ret = clEnqueueReadBuffer(command_queue, gpu_data, CL_TRUE, 0, n_px * n_channels, data, 0, NULL, NULL);
        ret = clEnqueueReadBuffer(command_queue, gpu_labels, CL_TRUE, 0, n_px * sizeof(int), labels, 0, NULL, NULL);
        ret = clEnqueueReadBuffer(command_queue, gpu_dists, CL_TRUE, 0, n_px * sizeof(double), dists, 0, NULL, NULL);
        update_centers(data, centers, labels, dists, n_px, n_channels, n_clus);*/
        
        /*
        // ščepec: zagon
        ret = clEnqueueNDRangeKernel(command_queue, labelPixelsKernel, 1, NULL, &global_item_size, &local_item_size, 0, NULL, NULL);

        // Kopiranje rezultatov
        ret = clEnqueueReadBuffer(command_queue, gpu_data, CL_TRUE, 0, n_px * n_channels, data, 0, NULL, NULL);
        ret = clEnqueueReadBuffer(command_queue, gpu_labels, CL_TRUE, 0, n_px * sizeof(int), labels, 0, NULL, NULL);
        ret = clEnqueueReadBuffer(command_queue, gpu_dists, CL_TRUE, 0, n_px * sizeof(double), dists, 0, NULL, NULL);

        labeling += omp_get_wtime() - dt;


        dt = omp_get_wtime();
            update_centers(data, centers, labels, dists, n_px, n_channels, n_clus);
        updating += omp_get_wtime() - dt;*/
   
    }
    ret = clEnqueueReadBuffer(command_queue, gpu_data, CL_TRUE, 0, n_px * n_channels, data, 0, NULL, NULL);
    ret = clEnqueueReadBuffer(command_queue, gpu_labels, CL_TRUE, 0, n_px * sizeof(int), labels, 0, NULL, NULL);
    ret = clEnqueueReadBuffer(command_queue, gpu_dists, CL_TRUE, 0, n_px * sizeof(double), dists, 0, NULL, NULL);
    //Get data
    update_image(data, centers, labels, n_px, n_channels);

    *n_iters = iter;
    printf("Labeling %f\nUpdating %f\n", labeling, updating);

    ////////////////////////////////////////////////////////
    //Clean-Up
    ret = clFlush(command_queue);
    ret = clFinish(command_queue);
    ret = clReleaseKernel(labelPixelsKernel);
    ret = clReleaseProgram(program);
    ret = clReleaseMemObject(gpu_data);
    ret = clReleaseMemObject(gpu_labels);
    ret = clReleaseMemObject(gpu_centers);
    ret = clReleaseMemObject(gpu_dists);
    ret = clReleaseCommandQueue(command_queue);
    ret = clReleaseContext(context);
    ret = clReleaseMemObject(global_sum);
    ret = clReleaseMemObject(global_count);
    ret = clReleaseMemObject(changedBuf);
    free(centers);
    free(labels);
    free(dists);
}

void print_usage(char *pgr_name)
{
    char *usage = "\nPROGRAM USAGE \n\n"
                  "   %s [-h] [-k num_clusters] [-i max_iters] [-o output_img] [-s seed] input_image \n\n"
                  "OPTIONAL PARAMETERS \n\n"
                  "   -k num_clusters : number of clusters, must be bigger than 1. Default is %d. \n"
                  "   -i max_iters    : Max Number of iterations, must be bigger than 0. defualt is %d \n"
                  "   -o output_image : output image path, include extenstion. \n"
                  "   -s seed         : seed for random function. \n"
                  "   -t threads      : number of threads for OpenMP\n"
                  "   -w workgroup    : size of the workgroup\n"
                  "   -h              : print help. \n\n";

    fprintf(stderr, usage, pgr_name, 4, 150);
}

void print_exec(int width, int height, int n_ch, int n_clus, int n_iters, off_t inSize, off_t outSize, double dt, int workgroupsize)
{
    char *details = "\nEXECUTION\n\n"
                    "  Image size              : %d x %d\n"
                    "  Color channels          : %d\n"
                    "  Number of clusters      : %d\n"
                    "  Number of iterations    : %d\n"
                    "  Input Image Size        : %ld KB\n"
                    "  Output Image Size       : %ld KB\n"
                    "  Size diffrance          : %ld KB\n"
                    "  Runtime                 : %f\n"
		            "  WorkGroupSize           : %d\n\n";

    fprintf(stdout, details, width, height, n_ch, n_clus, n_iters, inSize / 1000, outSize / 1000, (inSize - outSize) / 1000, dt, workgroupsize);
}

off_t fsize(const char *filename)
{ // https://stackoverflow.com/questions/8236/how-do-you-determine-the-size-of-a-file-in-c
    struct stat st;

    if (stat(filename, &st) == 0)
        return st.st_size;

    fprintf(stderr, "Cannot determine size of %s: %s\n",
            filename, strerror(errno));

    return -1;
}

int main(int argc, char **argv)
{
    int n_iters = 150;
    int n_clus = 4;
    char *out_path = "result.png";
    int width, height, n_ch, n_threads = N_THREADS;
    byte_t *data;
    seed = time(NULL);
    char *in_path = NULL;
    int workgroupsize = WORKGROUP_SIZE;

    // Parsing arguments and optional parameters
    // https://www.gnu.org/software/libc/manual/html_node/Example-of-Getopt.html
    char optchar;
    while ((optchar = getopt(argc, argv, "k:i:o:s:t:w:h")) != -1) {
        switch (optchar) {
        case 'k':
            n_clus = strtol(optarg, NULL, 10);
            break;
        case 'i':
            n_iters = strtol(optarg, NULL, 10);
            break;
        case 'o':
            out_path = optarg;
            break;
        case 's':
            seed = strtol(optarg, NULL, 10);
            break;
        case 't':
            n_threads = strtol(optarg, NULL, 10);
            break;
        case 'w':
            workgroupsize = strtol(optarg, NULL, 10);
            break;
        case 'h':
        default:
            print_usage(argv[0]);
            exit(EXIT_FAILURE);
            break;
        }
    }

    in_path = argv[optind];

    // Validating inputs
    if (in_path == NULL) {
        print_usage(argv[0]);
        exit(EXIT_FAILURE);
    }
    if (n_clus < 2) {
        fprintf(stderr, "INPUT ERROR: << Invalid number of clusters >> \n");
        exit(EXIT_FAILURE);
    }
    if (n_iters < 1) {
        fprintf(stderr, "INPUT ERROR: << Invalid maximum number of iterations >> \n");
        exit(EXIT_FAILURE);
    }

    //Seed rnd function
    //srand(seed);
    ///////////////////////////////////////////////////
    //Loading image
    data = img_load(in_path, &width, &height, &n_ch);

    // Executing k-means segmentation
    double dt = omp_get_wtime();
    //init_centers(data, centers, n_px, n_channels, n_clus);
    kmeans(data, width, height, n_ch, n_clus, &n_iters, n_threads, workgroupsize);
    dt = omp_get_wtime() - dt;

    // Saving Image
    img_save(out_path, data, width, height, n_ch);

    //Statistics
    off_t inSize = fsize(in_path);
    off_t outSize = fsize(out_path);
    print_exec(width, height, n_ch, n_clus, n_iters, inSize, outSize, dt, workgroupsize);


    //Cleaning up
    free(data);

    return EXIT_SUCCESS;
}

