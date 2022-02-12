/**
 * @file OpenMP.c
 * @author FallingLights (fallinglights@protonmail.com)
 * @brief
 * @version 0.1
 * @date 2022-02-12
 * 
 * Uporabil sem schedule(runtime) moreš nastaviti $ export OMP_SCHEDULE=static
 * @copyright
 * @compile gcc -Wall -o serial.out serial.c -lm -fopenm -O2
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

#define STB_IMAGE_IMPLEMENTATION
#include "include/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "include/stb_image_write.h"

#define N_THREADS 2

typedef unsigned char byte_t;

int seed;

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

void label_pixels(byte_t *data, double *centers, int *labels, double *dists, int *changes, int n_px, int n_ch, int n_clus)
{
    int px, ch, k;
    int min_k, tmp_changes = 0;
    double dist, min_dist, tmp;

    #pragma omp parallel for schedule(guided, 100) private(px, ch, k, min_k, dist, min_dist, tmp)
    for (px = 0; px < n_px; px++) {
        min_dist = DBL_MAX;

        for (k = 0; k < n_clus; k++) {
            dist = 0;

            for (ch = 0; ch < n_ch; ch++) {
                tmp = (double)(data[px * n_ch + ch] - centers[k * n_ch + ch]);
                dist += tmp * tmp;
            }

            if (dist < min_dist) {
                min_dist = dist;
                min_k = k;
            }
        }

        dists[px] = min_dist;

        if (labels[px] != min_k) {
            labels[px] = min_k;
            tmp_changes = 1;
        }
    }

    *changes = tmp_changes;
}

void update_centers(byte_t *data, double *centers, int *labels, double *dists, int n_px, int n_ch, int n_clus)
{
    int px, ch, k;
    int *counts;
    int min_k, far_px;
    double max_dist;

    counts = malloc(n_clus * sizeof(int));

    
    //Resetting arrays
    for (k = 0; k < n_clus; k++) {
        for (ch = 0; ch < n_ch; ch++) {
            centers[k * n_ch + ch] = 0;
        }

        counts[k] = 0;
    }

    //Calculating sums and updating cluster counters
    for (px = 0; px < n_px; px++) {
        min_k = labels[px];

        for (ch = 0; ch < n_ch; ch++) {
            centers[min_k * n_ch + ch] += data[px * n_ch + ch];
        }

        counts[min_k]++;
    }

    //means
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

void kmeans(byte_t *data, int width, int height, int n_channels, int n_clus, int *n_iters, int n_threads)
{
    int n_px;
    int iter, max_iters;
    int changes;
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

    //Training
    for (iter = 0; iter < max_iters; iter++) {
        label_pixels(data, centers, labels, dists, &changes, n_px, n_channels, n_clus);

        if (!changes) {
            break;
        }

        update_centers(data, centers, labels, dists, n_px, n_channels, n_clus);
        printf("Train step %d done\n", iter);
    }

    update_image(data, centers, labels, n_px, n_channels);

    *n_iters = iter;

    //Clean-Up
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
                  "   -h              : print help. \n\n";

    fprintf(stderr, usage, pgr_name, 4, 150);
}

void print_exec(int width, int height, int n_ch, int n_clus, int n_iters, off_t inSize, off_t outSize, double dt)
{
    char *details = "\nEXECUTION\n\n"
                    "  Image size              : %d x %d\n"
                    "  Color channels          : %d\n"
                    "  Number of clusters      : %d\n"
                    "  Number of iterations    : %d\n"
                    "  Input Image Size        : %ld KB\n"
                    "  Output Image Size       : %ld KB\n"
                    "  Size diffrance          : %ld KB\n"
                    "  Runtime                 : %f\n\n";

    fprintf(stdout, details, width, height, n_ch, n_clus, n_iters, inSize / 1000, outSize / 1000, (inSize - outSize) / 1000, dt);
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

    // Parsing arguments and optional parameters
    // https://www.gnu.org/software/libc/manual/html_node/Example-of-Getopt.html
    char optchar;
    while ((optchar = getopt(argc, argv, "k:i:o:s:t:h")) != -1) {
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

    //Loading image
    data = img_load(in_path, &width, &height, &n_ch);

    // Executing k-means segmentation
    double dt = omp_get_wtime();
    kmeans(data, width, height, n_ch, n_clus, &n_iters, n_threads);
    dt = omp_get_wtime() - dt;

    // Saving Image
    img_save(out_path, data, width, height, n_ch);

    //Statistics
    off_t inSize = fsize(in_path);
    off_t outSize = fsize(out_path);
    print_exec(width, height, n_ch, n_clus, n_iters, inSize, outSize, dt);


    //Cleaning up
    free(data);

    return EXIT_SUCCESS;
}
