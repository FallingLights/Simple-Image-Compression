/**
 * @file serial.cpp
 * @author AG
 * @brief
 * @version 0.1
 * @date 2022-02-10
 *
 * @copyright Copyright (c) 2022
 *
 * @compile c++ -o serial -I include/ serial.cpp
 *  c++ serial.cpp -I include/ -Wl,-rpath,./ -L./ -l:"libfreeimage.so.3" -o serial
 */
#include <algorithm>
#include <chrono>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <sys/stat.h>
#include <vector>

#include <tclap/CmdLine.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define errorexit(errcode, errstr)                \
    fprintf(stderr, "%s: &d\n", errstr, errcode); \
    exit(1);

long GetFileSize(std::string filename)
{ // https://stackoverflow.com/questions/5840148/how-can-i-get-a-files-size-in-c
    struct stat stat_buf;
    int rc = stat(filename.c_str(), &stat_buf);
    return rc == 0 ? stat_buf.st_size : -1;
}

class Pixel
{
public:
    u_char b, g, r;

    Pixel(u_char b, u_char g, u_char r)
    {
        this->b = b;
        this->g = g;
        this->r = r;
    }
};

class KMeans
{
private:
    unsigned char *image;
    int width;
    int height;
    int n_ch;
    int n_clus;
    int iterations;

    int n_px;
    int *labels;
    double *centers;
    double *distances;
    int changes;
    bool first;

public:
    // Intialise cluster centres as random pixels from the image
    KMeans(unsigned char *image, int width, int height, int n_ch, int n_clus, int iterations)
    {
        this->image = image;
        this->width = width;
        this->height = height;
        this->n_ch = n_ch;
        this->n_clus = n_clus;
        this->iterations = iterations;

        this->n_px = width * height;

        this->labels = new int[n_px];
        this->centers = new double[n_ch * n_clus];
        this->distances = new double[n_px];

        this->first = true;
        initialize_n_clus();
        train();
    }

    void train()
    {
        std::cout << "Training...\n";
        for (size_t i = 0; i < iterations; i++) {
            label_pixels();
            if (!changes) {
                break;
            }
            computeCentroids();
            std::cout << "Train step " << i << " done" << std::endl;
        }
    }

    void out(std::string outputPath)
    {
        int minimum_cluster;
        int px, ch;
        for (px = 0; px < n_px; px++) {
            minimum_cluster = labels[px];

            for (ch = 0; ch < n_ch; ch++) {
                image[n_ch * (px * + ch)] = (u_char)round(centers[minimum_cluster * n_ch + ch]);
                // image[px * n_ch + 3] = 255;
            }
        }

        std::cout << "Writing Image\n";
        stbi_write_jpg(outputPath.c_str(), width, height, n_ch, image, 100);
        clean();
    }
    void clean()
    {
        delete (labels);
        delete (centers);
        delete (distances);
        delete (image);
    }

private:
    void initialize_n_clus()
    {
        int k, ch, rnd;

        for (k = 0; k < n_clus; k++) {
            rnd = rand() % n_px;

            for (ch = 0; ch < n_ch; ch++) {
                centers[k * n_ch + ch] = image[rnd * n_ch + ch];
            }
        }
    }

    void label_pixels()
    {
        int px, ch, k;
        int changesTemp = 0;
        int centroidLabel = 0;
        double distance, tmp, minimum_distance;

        for (px = 0; px < n_px; px++) {
            minimum_distance = std::numeric_limits<double>::max(); // nastavimo na MAX
            for (k = 0; k < n_clus; k++) {
                distance = 0;
                for (ch = 0; ch < n_ch; ch++) {
                    tmp = (double)(image[px * n_ch + ch] - centers[k * n_ch + ch]);
                    distance += tmp * tmp;
                }

                if (distance < minimum_distance) {
                    minimum_distance = distance;
                    centroidLabel = k;
                }
            }

            distances[px] = minimum_distance;

            if (labels[px] != centroidLabel) {
                labels[px] = centroidLabel;
                changesTemp = 1;
            }
        }
        // changes
        changes = changesTemp;
    }

    void computeCentroids()
    { // Compute mean
        int *cluster_counter = new int[n_clus];
        int centroidLabel;
        int furthest_pixel;
        int px, ch, k;
        double max_distance;
        // Reset counter and centers
        for (k = 0; k < n_clus; k++) {
            for (ch = 0; ch < n_ch; ch++) {
                centers[k * n_ch + ch] = 0;
            }
            cluster_counter[k] = 0;
        }

        for (px = 0; px < n_px; px++) {
            centroidLabel = labels[px];
            for (ch = 0; ch < n_ch; ch++) {
                centers[centroidLabel * n_ch + ch] += image[px * n_ch + ch];
            }
            cluster_counter[centroidLabel]++;
        }

        for (k = 0; k < n_clus; k++) {
            if (cluster_counter[k]) {
                for (ch = 0; ch < n_ch; ch++) {
                    centers[k * n_ch * ch] /= cluster_counter[k];
                }
            } else { // Če je cluster empty
                max_distance = 0;
                for (px = 0; px < n_px; px++) {
                    if (distances[px] > max_distance) {
                        max_distance = distances[px];
                        furthest_pixel = px;
                    }
                }

                for (ch = 0; ch < n_ch; ch++) {
                    centers[k * n_ch * ch] = image[furthest_pixel * n_ch * ch];
                }

                distances[furthest_pixel] = 0;
            }
        }
        delete (cluster_counter);
    }

    double euclidian_distance(int x1, int y1, int z1, int x2, int y2, int z2)
    {
        return sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2) + pow(z1 - z2, 2));
    }

    int my_random(int min, int max) // range : [min, max]
    {
        if (first) {
            srand(time(NULL)); // seeding for the first time only!
            first = false;
        }
        return min + rand() % ((max + 1) - min);
    }

    long random_at_most(long max)
    {
        unsigned long
            // max <= RAND_MAX < ULONG_MAX, so this is okay.
            num_bins = (unsigned long)max + 1,
            num_rand = (unsigned long)RAND_MAX + 1,
            bin_size = num_rand / num_bins,
            defect = num_rand % num_bins;

        long x;
        do {
            x = random();
        }
        // This is carefully written not to overflow
        while (num_rand - defect <= (unsigned long)x);

        // Truncated division is intentional
        return x / bin_size;
    }
};

int main(int argc, char const *argv[])
{
    try {

        // Last line in help, delimiter, version number
        TCLAP::CmdLine cmd("Serial k-Means image compressor", ' ', "0.1");

        // Define a value argument and add it to the command line.
        TCLAP::ValueArg<std::string> inputArg("i", "input", "Path to input photo", true, "image.png", "string", cmd);
        TCLAP::ValueArg<std::string> outputArg("o", "output", "Path to output photo", true, "image_compressed.png", "string", cmd);
        TCLAP::ValueArg<int> clutstersArg("k", "n_clus", "Number of n_clus", false, 4, "int", cmd);
        TCLAP::ValueArg<int> iterationsArg("t", "iterations", "Number of iterations", false, 150, "int", cmd);
        // Parse the argv array.
        cmd.parse(argc, argv);

        // Get the value parsed by each arg.
        std::string inputPath = inputArg.getValue();
        std::string outputPath = outputArg.getValue();
        int numberOfn_clus = clutstersArg.getValue();
        int iterations = iterationsArg.getValue();

        std::cout << inputPath << std::endl;
        std::cout << outputPath << std::endl;
        std::cout << numberOfn_clus << std::endl;
        std::cout << iterations << std::endl;

        // Do what you intend.
        srand(time(NULL));
        // Load image from file
        unsigned char *image;
        int width;
        int height;
        int n_ch;
        image = stbi_load(inputPath.c_str(), &width, &height, &n_ch, 0);
        if (image == NULL) {
            fprintf(stderr, "ERROR LOADING IMAGE: << Invalid file name or format >> \n");
            exit(EXIT_FAILURE);
        }

        KMeans kmeans(image, width, height, n_ch, numberOfn_clus, iterations);
        // kmeans.train();
        kmeans.out(outputPath);

    } catch (TCLAP::ArgException &e) // catch exceptions
    {
        std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl;
    }
    return 0;
}
