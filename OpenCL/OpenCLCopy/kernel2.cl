#define GROUPSIZE 1000

float dist(float px, float py, float cx, float cy)
{
    float distance = ((px - cx) * (px - cx)) + ((py - cy) * (py - cy));
    return distance;
}

__kernel void assignCluster(__global int *label,
                            __global float *pointX,
                            __global float *pointY,
                            __global float *centerX,
                            __global float *centerY,
                            __global float *sumx,
                            __global float *sumy,
                            __global int *counts,
                            __local float *sharedX,
                            __local float *sharedY,
                            __local int *sharedC,
                            int n, int k 
                            )
{

    int gid = get_global_id(0);
    int lid = get_local_id(0);

    if(lid < k)
    {
        sharedX[lid] = centerX[lid];
        sharedY[lid] = centerY[lid];
    }
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

    float x = pointX[gid];
    float y = pointY[gid];

    float min_dist = 1E06;
    int cluster = 100;
    float distance = 0;
    for(int i = 0; i < k; i++)
    {
        distance = dist(x, y, sharedX[i], sharedY[i]);
        if(distance < min_dist)
        {
            min_dist = distance;
            cluster = i;
        }
    }
    label[gid] = cluster;

    barrier(CLK_LOCAL_MEM_FENCE| CLK_GLOBAL_MEM_FENCE);

    for(int i = 0; i < k; i++)
    {
        sharedX[lid] = cluster == i ? x : 0;
        sharedY[lid] = cluster == i ? y : 0;
        sharedC[lid] = cluster == i ? 1 : 0;

        barrier(CLK_LOCAL_MEM_FENCE| CLK_GLOBAL_MEM_FENCE);

        int size = GROUPSIZE % 2 ? GROUPSIZE + 1 : GROUPSIZE;
        int thresh = GROUPSIZE % 2 ? (GROUPSIZE - 1) / 2 : GROUPSIZE / 2;
        int stride = size / 2;
        int stop = 0;
        while(!stop)
        {
            stop = stride == 1 ? 1 : 0;
            if(lid < thresh)
            {
                sharedX[lid] += sharedX[lid + stride];
                sharedY[lid] += sharedY[lid + stride];
                sharedC[lid] += sharedC[lid + stride];
                
            }
            thresh = stride % 2 ? (stride - 1) / 2 : stride / 2;
            size = stride % 2 ? stride + 1 : stride;
            stride = size / 2;
            barrier(CLK_LOCAL_MEM_FENCE| CLK_GLOBAL_MEM_FENCE);
        }   

        if(lid == 0)
        {
            int ci = (gid/GROUPSIZE) * k + i;
            sumx[ci] = sharedX[lid];
            sumy[ci] = sharedY[lid];
            counts[ci] = sharedC[lid];
        }
        barrier(CLK_LOCAL_MEM_FENCE| CLK_GLOBAL_MEM_FENCE);
    }
}

__kernel void updateMean(__global float *cX,
                         __global float *cY,
                         __global float *sumx,
                         __global float *sumy,
                         __global float *oldx,
                         __global float *oldy,
                         __global int *counts,
                         __local float *sharedX,
                         __local float *sharedY,
                         __local int *sharedC,
                         __local int *flags,
                         int k, int numGroups, __global int *flag)
 {
    int lid = get_local_id(0);
    int locSize = get_local_size(0);
    int dim = k * numGroups / locSize;
    int size, thresh, stride;
    int stop = 0;

    if(k * numGroups > 1000)
    {
        numGroups = GROUPSIZE/k;
        for(int i = 1; i < dim; i++)
        {
            sumx[lid] += sumx[lid + (i * locSize)];
            sumy[lid] += sumy[lid + (i * locSize)];
            counts[lid] += counts[lid + (i * locSize)];
        }
    }
    
    barrier(CLK_LOCAL_MEM_FENCE| CLK_GLOBAL_MEM_FENCE);
    sharedX[lid] = sumx[lid];
    sharedY[lid] = sumy[lid];
    sharedC[lid] = counts[lid];

    barrier(CLK_LOCAL_MEM_FENCE| CLK_GLOBAL_MEM_FENCE);

    size = numGroups % 2 ? numGroups + 1 : numGroups;
    thresh = numGroups % 2 ? (numGroups - 1) / 2 : numGroups / 2;
    stride = size / 2;
    
    stop = 0;
    if(numGroups  > 1)
    {
        while(!stop)
        {
            stop = stride == 1 ? 1 : 0;
            if(lid < (thresh * k))
            {
                sharedX[lid] += sharedX[lid + (stride * k)];
                sharedY[lid] += sharedY[lid + (stride * k)];
                sharedC[lid] += sharedC[lid + (stride * k)];        
            }
            thresh = stride % 2 ? (stride - 1) / 2 : stride / 2;
            size = stride % 2 ? stride + 1 : stride;
            stride = size / 2;
            barrier(CLK_LOCAL_MEM_FENCE| CLK_GLOBAL_MEM_FENCE);
        } 
    }

    if(lid < k)
    {
        int count = sharedC[lid];
        if(count != 0 )
        {
            cX[lid] = sharedX[lid] / count;
            cY[lid] = sharedY[lid] / count;
        }
        float diffx = fabs(cX[lid] - oldx[lid]);
        float diffy = fabs(cY[lid] - oldy[lid]);
        flags[lid] = (diffx == 0 && diffy == 0) ? 1 : 0; 
        oldx[lid] = cX[lid];
        oldy[lid] = cY[lid];
        counts[lid] = 0;
        sumy[lid] = 0;
        sumx[lid] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE| CLK_GLOBAL_MEM_FENCE);

    if(lid == 0)
    {
        int ret = 0;
        for(int i = 0; i < k; i++)
        {
            if(flags[i] == 0)
            {
                ret = 1;
                break;
            }
        }
        flag[0] = ret;
    }

 }

 __kernel void testReduction(__global int *vals, const int k)
 {

     int lid = get_local_id(0);
     int size = GROUPSIZE % 2 ? GROUPSIZE + 1 : GROUPSIZE;
     int thresh = GROUPSIZE % 2 ? GROUPSIZE - 1 / 2 : GROUPSIZE / 2;
     int stride = size / 2;

     int stop = 0;
     
     while(!stop)
    {
        stop = stride == 1 ? 1 : 0;
        if(lid < thresh)
        {
            vals[lid] += vals[lid + stride];
        }
        thresh = stride % 2 ? (stride - 1) / 2 : stride / 2;
        size = stride % 2 ? stride + 1 : stride;
        stride = size / 2;        
        barrier(CLK_LOCAL_MEM_FENCE);
    }  
 }