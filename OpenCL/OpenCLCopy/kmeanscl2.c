#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "kMeansCL.h"
#include <time.h>

#pragma GCC diagnostic ignored "-Wunused-result"

typedef long long hrtime_t;

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#define GROUPSIZE 1000

#define MAX_SOURCE_SIZE (0x10000)
void readFiles(float *X, float *Y, char *x, char *y, int n);
void writeFiles(int *C, float *CX, float *CY, int k, int n);

hrtime_t gethrtime()
{
    hrtime_t elapsed;
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);          //multicore - safe, but includes outside interference
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t); // maynot be multicore -safe if process migrates cores
    elapsed = (hrtime_t)t.tv_sec * (hrtime_t)1e9 + (hrtime_t)t.tv_nsec;
    return elapsed;
}

int main(int argc, char **argv)
{
    int n;
    int k;

    char *xFilename;
    char *yFilename;
    if (argc >= 2)
    {
        n = atoi(argv[1]);
    }
    else
    {
        n = 100;
    }
    printf("n: %d\n", n);
    if (argc >= 3)
    {
        k = atoi(argv[2]);
    }
    else
    {
        k = 4;
    }
    printf("k: %d\n", k);
    if (argc >= 5)
    {
        xFilename = argv[3];
        yFilename = argv[4];
    }
    else
    {
        xFilename = "Data/X100.txt";
        yFilename = "Data/Y100.txt";
    }

    int numGroups = n / GROUPSIZE;

    float *X = malloc(n * sizeof(float));
    float *Y = malloc(n * sizeof(float));
    float *CX = malloc(k * sizeof(float));
    float *CY = malloc(k * sizeof(float));
    float *oldX = malloc(k * sizeof(float));
    float *oldY = malloc(k * sizeof(float));
    int *counts = malloc(numGroups * k * sizeof(int));
    int *flag = malloc(sizeof(int));
    int *clusters = malloc(n * sizeof(int));
    flag[0] = 0;

    for (int i = 0; i < k; i++)
    {
        float range = 8;
        CX[i] = range * ((float)rand() / (float)RAND_MAX);
        CY[i] = range * ((float)rand() / (float)RAND_MAX);
    }

    readFiles(X, Y, xFilename, yFilename, n);

    FILE *file = fopen("./kMeans.cl", "r");
    if (!file)
    {
        printf("Failed to open kernel\n");
        exit(1);
    }
    char *source = malloc(MAX_SOURCE_SIZE);
    size_t sourceSize = fread(source, 1, MAX_SOURCE_SIZE, file);
    fclose(file);

    cl_platform_id platformID = NULL;
    cl_device_id deviceID = NULL;
    cl_uint numDevices;
    cl_uint NumPlatforms;
    cl_uint ret;

    ret = clGetPlatformIDs(1, &platformID, &NumPlatforms);
    if (ret != CL_SUCCESS)
    {
        printf("GetPlatformID Error: %d\n", ret);
        exit(1);
    }
    ret = clGetDeviceIDs(platformID, CL_DEVICE_TYPE_GPU, 1, &deviceID, &numDevices);
    if (ret != CL_SUCCESS)
    {
        printf("GetDeviceID Error: %d\n", ret);
        exit(1);
    }
    cl_context context = clCreateContext(NULL, 1, &deviceID, NULL, NULL, &ret);
    if (ret != CL_SUCCESS)
    {
        printf("CreateContext Error: %d\n", ret);
        exit(1);
    }
    cl_command_queue commandQueue = clCreateCommandQueue(context, deviceID, NULL, &ret);
    if (ret != CL_SUCCESS)
    {
        printf("CommandQueue Error: %d\n", ret);
        exit(1);
    }

    cl_mem xMem = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, n * sizeof(float), X, &ret);
    cl_mem yMem = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, n * sizeof(float), Y, &ret);
    cl_mem cxMem = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, k * sizeof(float), CX, &ret);
    cl_mem cyMem = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, k * sizeof(float), CY, &ret);
    cl_mem oxMem = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, k * sizeof(float), oldX, &ret);
    cl_mem oyMem = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, k * sizeof(float), oldY, &ret);
    cl_mem gcMem = clCreateBuffer(context, CL_MEM_READ_WRITE, numGroups * k * sizeof(int), NULL, &ret);
    cl_mem sxMem = clCreateBuffer(context, CL_MEM_READ_WRITE, numGroups * k * sizeof(int), NULL, &ret);
    cl_mem syMem = clCreateBuffer(context, CL_MEM_READ_WRITE, numGroups * k * sizeof(int), NULL, &ret);
    cl_mem fMem = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(int), flag, &ret);
    cl_mem clMem = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, n * sizeof(int), clusters, &ret);
    if (!xMem || !yMem || !cxMem || !cyMem || !gcMem || !sxMem || !syMem || !oxMem || !oyMem || !fMem || !clMem)
    {
        printf("Failed to allocate memory\n");
        exit(1);
    }

    cl_program program = clCreateProgramWithSource(context, 1, (const char **)&source, (const size_t *)&sourceSize, &ret);
    if (ret != CL_SUCCESS)
    {
        printf("CreateProgram Error %d\n", ret);
        exit(1);
    }
    ret = clBuildProgram(program, 1, &deviceID, "-cl-nv-verbose", NULL, NULL);
    if (ret != CL_SUCCESS)
    {
        printf("BuildProgram Error %d\n", ret);
        exit(1);
    }
    cl_kernel kernel = clCreateKernel(program, "assignCluster", &ret);
    if (ret != CL_SUCCESS)
    {
        printf("CreateKernel Error %d\n", ret);
        exit(1);
    }

    cl_kernel kernel2 = clCreateKernel(program, "updateMean", &ret);
    if (ret != CL_SUCCESS)
    {
        printf("CreateKernel2 Error %d\n", ret);
        exit(1);
    }

    clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&clMem);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&xMem);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&yMem);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&cxMem);
    clSetKernelArg(kernel, 4, sizeof(cl_mem), (void *)&cyMem);
    clSetKernelArg(kernel, 5, sizeof(cl_mem), (void *)&sxMem);
    clSetKernelArg(kernel, 6, sizeof(cl_mem), (void *)&syMem);
    clSetKernelArg(kernel, 7, sizeof(cl_mem), (void *)&gcMem);
    clSetKernelArg(kernel, 8, GROUPSIZE * sizeof(int), NULL);
    clSetKernelArg(kernel, 9, GROUPSIZE * sizeof(float), NULL);
    clSetKernelArg(kernel, 10, GROUPSIZE * sizeof(float), NULL);
    clSetKernelArg(kernel, 11, sizeof(int), (void *)&n);
    clSetKernelArg(kernel, 12, sizeof(int), (void *)&k);

    cl_uint *rets = malloc(14 * sizeof(cl_uint));
    rets[0] = clSetKernelArg(kernel2, 0, sizeof(cl_mem), (void *)&cxMem);
    rets[1] = clSetKernelArg(kernel2, 1, sizeof(cl_mem), (void *)&cyMem);
    rets[2] =clSetKernelArg(kernel2, 2, sizeof(cl_mem), (void *)&sxMem);
    rets[3] = clSetKernelArg(kernel2, 3, sizeof(cl_mem), (void *)&syMem);
    rets[4] = clSetKernelArg(kernel2, 4, sizeof(cl_mem), (void *)&oxMem);
    rets[5] =clSetKernelArg(kernel2, 5, sizeof(cl_mem), (void *)&oyMem);
    rets[6] =clSetKernelArg(kernel2, 6, sizeof(cl_mem), (void *)&gcMem);
    rets[7] =clSetKernelArg(kernel2, 7, GROUPSIZE * sizeof(int), NULL);
    rets[8] =clSetKernelArg(kernel2, 8, GROUPSIZE * sizeof(float), NULL);
    rets[9] =clSetKernelArg(kernel2, 9, GROUPSIZE * sizeof(float), NULL);
    rets[10] =clSetKernelArg(kernel2, 10, k * sizeof(int), NULL);
    rets[11] =clSetKernelArg(kernel2, 11, sizeof(int), (void *)&k);
    rets[12] =clSetKernelArg(kernel2, 12, sizeof(int), (void *)&numGroups);
    rets[13] = clSetKernelArg(kernel2, 13, sizeof(cl_mem), (void *)&fMem);
    for(int i = 0; i < 14; i++)
    {
        if(rets[i] != CL_SUCCESS)
        {
            printf("Set argument %d error %d\n", i, rets[i]);
            exit(1);
        }
    }
    
    size_t glob;
    size_t loc;
    int iters = 0;
    hrtime_t start = gethrtime();
    do
    {
        iters++;
        glob = n;
        loc = GROUPSIZE;
        
        ret = clEnqueueNDRangeKernel(commandQueue, kernel, 1, 0, &glob, &loc, 0, NULL, NULL);
        if (ret != CL_SUCCESS)
        {
            printf("EnqueueKernel Error %d\n", ret);
            exit(1);
        }
        clFinish(commandQueue);

        glob = k * numGroups > GROUPSIZE ? GROUPSIZE : k * numGroups;
        loc = k * numGroups > GROUPSIZE ? GROUPSIZE : k * numGroups;
        ret = clEnqueueNDRangeKernel(commandQueue, kernel2, 1, 0, &glob, &loc, 0, NULL, NULL);
        if (ret != CL_SUCCESS)
        {
            printf("EnqueueKernel2 Error %d\n", ret);
            exit(1);
        }

        clFinish(commandQueue);

        ret = clEnqueueReadBuffer(commandQueue, fMem, CL_TRUE, 0, sizeof(int), flag, 0, NULL, NULL);
        if (ret != CL_SUCCESS)
        {
            printf("1Read Buffer Error %d\n", ret);
            exit(1);
        }
    } while(flag[0]);

    ret = clEnqueueReadBuffer(commandQueue, cyMem, CL_TRUE, 0, k * sizeof(float), CY, 0, NULL, NULL);
    if (ret != CL_SUCCESS)
    {
        printf("Read Buffer Error %d\n", ret);
        exit(1);
    }
    ret = clEnqueueReadBuffer(commandQueue, cxMem, CL_TRUE, 0, k * sizeof(float), CX, 0, NULL, NULL);
    if (ret != CL_SUCCESS)
    {
        printf("Read Buffer Error %d\n", ret);
        exit(1);
    }
    ret = clEnqueueReadBuffer(commandQueue, clMem, CL_TRUE, 0, n * sizeof(int), clusters, 0, NULL, NULL);
    if (ret != CL_SUCCESS)
    {
        printf("Read Buffer Error %d\n", ret);
        exit(1);
    }
    hrtime_t end = gethrtime();
    float t = (end - start) / 1E09;
    printf("total seconds: %f\n", t);
    
    writeFiles(clusters, CX, CY, k, n);
    return (0);
}

void readFiles(float *X, float *Y, char *x, char *y, int n)
{
    FILE *xFile = fopen(x, "r");
    FILE *yFile = fopen(y, "r");
    if (xFile == NULL || yFile == NULL)
    {
        printf("Could not read files\n");
        exit(1);
    }
    //float *buff
    for (int i = 0; i < n; i++)
    {
        fscanf(xFile, "%f", &X[i]);
        fscanf(yFile, "%f", &Y[i]);
        //printf("%f\t%f", X[i], Y[i]);
        //printf("\n");
    }

    fclose(xFile);
    fclose(yFile);
}

void writeFiles(int *C, float *CX, float *CY, int k, int n)
{
    FILE *cFile = fopen("../OCLC.txt", "w");
    for (int i = 0; i < n; i++)
    {
        fprintf(cFile, "%d", C[i]);
        if (i < n - 1)
        {
            fprintf(cFile, "\n");
        }
    }

    FILE *cxFile = fopen("../OCLCX.txt", "w");
    for (int i = 0; i < k; i++)
    {
        fprintf(cxFile, "%f", CX[i]);
        if (i < k - 1)
        {
            fprintf(cxFile, "\n");
        }
    }

    FILE *cyFile = fopen("../OCLCY.txt", "w");
    for (int i = 0; i < k; i++)
    {
        fprintf(cyFile, "%f", CY[i]);
        if (i < k - 1)
        {
            fprintf(cyFile, "\n");
        }
    }

    fclose(cFile);
    fclose(cyFile);
    fclose(cxFile);
}