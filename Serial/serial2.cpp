/**
 * @file serial.cpp
 * @author AG
 * @brief https://www.youtube.com/watch?v=tas0O586t80
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

#include <bits/stdc++.h>
#include <iomanip>
#include <math.h>

#include <tclap/CmdLine.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define errorexit(errcode, errstr)                \
    fprintf(stderr, "%s: &d\n", errstr, errcode); \
    exit(1);

#define sqr(x) ((x)*(x))

long GetFileSize(std::string filename)
{ // https://stackoverflow.com/questions/5840148/how-can-i-get-a-files-size-in-c
    struct stat stat_buf;
    int rc = stat(filename.c_str(), &stat_buf);
    return rc == 0 ? stat_buf.st_size : -1;
}

// Loads as RGBA... even if file is only RGB
// Feel free to adjust this if you so please, by changing the 4 to a 0.
bool load_image(std::vector<unsigned char>& image, const std::string& filename, int& x, int&y)
{ // https://www.cplusplus.com/forum/beginner/267364/
    int n;
    unsigned char* data = stbi_load(filename.c_str(), &x, &y, &n, 4);
    if (data != nullptr)
    {
        image = std::vector<unsigned char>(data, data + x * y * 4);
    }
    stbi_image_free(data);
    return (data != nullptr);
}

template <class T>
class MyArray2D
{
    std::vector<T> data;
    size_t sizeX, sizeY;
public:

const T& at(int x, int y) const { return data.at(y + x * sizeY); }

T& at(int x, int y) { return data.at(y + x * sizeY); }

// wrap other methods you need of std::vector here
};

class Pixel
{
public:
    u_char b, g, r;
    Pixel(u_char blue, u_char green, u_char red)
    {
        this->b = blue;
        this->g = green;
        this->r = red;
    }
};

std::default_random_engine dre(std::chrono::steady_clock::now().time_since_epoch().count()); // provide seed
class KMeans
{
private:
    unsigned char *image;
    int width;
    int height;
    int n_channels;
    int n_clusters;
    int iterations;

    int n_px;
    std::vector<std::vector<int>> labels; // https://stackoverflow.com/questions/9694838/how-to-implement-2d-vector-array
    std::vector<Pixel> clusterCentres;
public:
    KMeans(unsigned char *image, int width, int height, int n_ch, int n_clus, int iterations)
    {
        this->image = image;

        this->width = width;
        this->height = height;
        this->n_channels = n_ch;

        this->n_clusters = n_clus;
        this->iterations = iterations;

        this->n_px = width * height;
        this->labels.resize(height, std::vector<int>(width, 0));

        for (size_t i = 0; i < n_clusters; i++)
        {
            int rndRow = my_random(height - 1);
            int rndCol = my_random(width - 1);

            //unsigned bytePerPixel = n_channels;
            //unsigned char* pixelOffset = image + (rndRow + height * rndCol) * bytePerPixel; // ? Ali je rndRow zamenjam z rndCol (x,y)
            //unsigned char* pixelOffset = image + (4 * (rndCol * width + rndRow)); //(y , x) y = rndCol | pa 4 zamenji z bytePerPixel
            unsigned char* pixelOffset = get_pixel(rndRow, rndCol);
            unsigned char r = pixelOffset[0];
            unsigned char g = pixelOffset[1];
            unsigned char b = pixelOffset[2];
            printf("Row : %d, Col : %d, Red : %d, Green : %d, Blue: %d, rgb(%d, %d, %d)\n", rndRow, rndCol ,r, g, b, r,g,b);
            //unsigned char a = n_channels >= 4 ? pixelOffset[3] : 0xff;
            clusterCentres.push_back(Pixel(b, g, r));

            Pixel p = clusterCentres.at(i);
            printf("Cluster : R : %d G : %d : B : %d\n", p.r, p.g, p.b);
        }
        label_pixels();
    }

    void train()
    {
        std::cout << "Training...\n";
        for (size_t i = 0; i < iterations; i++) {
            computeCentroids();
            label_pixels();
            std::cout << "Train step " << i << " done" << std::endl;
        }
    }

    void out(std::string outputPath)
    {
        for (size_t r = 0; r < height; r++)
        {
            for (size_t c = 0; c < width; c++)
            {
                unsigned bytePerPixel = n_channels;
                Pixel p = clusterCentres.at(labels[r][c]);
                // ! Uprašanje če je to pravilno
                //labels[r][c] = 99;
                //printf("Out => Row : %d, Col : %d, Label : %d Red : %d, Green : %d, Blue: %d\n", r, c, labels[r][c], p.r, p.g, p.b);
                image[bytePerPixel * (r * width + c) + 0] = p.r; //Red n_channels * r * width + n_channels * c + 1
                image[bytePerPixel * (r * width + c) + 1] = p.g; // Green
                image[bytePerPixel * (r * width + c) + 2] = p.b; // Blue
            }
            
        }
        
        std::string ext;
        int dot = outputPath.find_last_of('.');
        ext = outputPath.substr(dot);
        std::cout << "Writing Image\n";
        if (ext.empty()) {
            fprintf(stderr, "ERROR SAVING IMAGE: << Unspecified format >> \n\n");
            return;
        }
        
        if (ext.compare(".jpeg") == 0 || ext.compare(".jpg") == 0) {
            stbi_write_jpg(outputPath.c_str(), width, height, n_channels, image, 100);
        } else if (ext.compare(".png") == 0) {
            stbi_write_png(outputPath.c_str(), width, height, n_channels, image, width * n_channels);
        } else if (ext.compare(".bmp") == 0) {
            stbi_write_bmp(outputPath.c_str(), width, height, n_channels, image);
        } else if (ext.compare(".tga") == 0) {
            stbi_write_tga(outputPath.c_str(), width, height, n_channels, image);
        } else {
            fprintf(stderr, "ERROR SAVING IMAGE: << Unsupported format >> \n\n");
        }
        //stbi_write_jpg(outputPath.c_str(), width, height, n_channels, image, 100);
        clean();
    }
    
    void clean()
    {
        for (size_t r = 0; r < height; r++)
        {
            labels[r].clear();
        }
        labels.clear();
        clusterCentres.clear();
        //delete (image);
    }

private:
    unsigned char* get_pixel(int x, int y){ // ! tuakj je lahko napaka
        unsigned bytePerPixel = n_channels;
        //unsigned char* pixelOffset = image + (x + height * y) * bytePerPixel; // ? Ali je rndRow zamenjam z rndCol (x,y)
        unsigned char* pixelOffset = image + (bytePerPixel * (x * width + y)); //(y , x) y = rndCol | pa 4 zamenji z bytePerPixel
        return pixelOffset;
    }

    /* void initialize_n_clus()
    {
        int k, ch, rnd;

        for (k = 0; k < n_clusters; k++) {
            rnd = rand() % n_px;

            for (ch = 0; ch < n_channels; ch++) {
                centers[k * n_channels + ch] = image[rnd * n_channels + ch];
            }
        }
    } */

    void label_pixels()
    {
        for (size_t r = 0; r < height; r++)
        {
            for (size_t c = 0; c < width; c++)
            { // ! TUKAJ NEKJE JE NAPAKA
                int centroidLabel = 0;
                u_char b, g, r;
                float distance, min_distance;
                u_char * rgb_pixel = get_pixel(r, c);
                r = rgb_pixel[0];
                g = rgb_pixel[1];
                b = rgb_pixel[2];
                Pixel test = Pixel(b, g, r);
                //min_distance = euclidian_distance(clusterCentres[0].r, clusterCentres[0].g, clusterCentres[0].b, r, g, b);
                min_distance= std::numeric_limits<float>::max();
                //printf("Row : %d Col : %d Dist 0 : %f\n", r,c,min_distance);
                for (size_t i = 0; i < n_clusters; i++)
                {
                    distance = 0;
                    //distance = euclidian_distance(clusterCentres[i].r, clusterCentres[i].g, clusterCentres[i].b, r, g, b);
                    distance = distance3d((float)clusterCentres[i].r, (float)clusterCentres[i].g, (float)clusterCentres[i].b, (float)r, (float)g, (float)b);
                    //printf("Row : %d Col : %d Dist %d : %f\n", r,c, i, min_distance);
                    if (min_distance > distance)
                    {
                        //std::cout << "New minimum\n";
                        min_distance = distance;
                        centroidLabel = i;
                        labels[r][c] = centroidLabel;
                        //printf("row : %d, col : %d, label : %d, writen %d\n", r, c, centroidLabel, labels[r][c]);
                    }
                }
                
            }
            
        }
        
    }
   private:
    void computeCentroids()
    { // Compute mean
        for (int i = 0; i < n_clusters; i++)
        {
            double mean_b = 0.0, mean_g = 0.0, mean_r = 0.0;
            int n = 0;
            for (int r = 0; r < height; r++)
            {
                for (int c = 0; c < width; c++)
                {
                    if (labels[r][c] == i)
                    {
                        u_char * rgb_pixel = get_pixel(r, c);
                        mean_r = rgb_pixel[0];
                        mean_g = rgb_pixel[1];
                        mean_b = rgb_pixel[2];
                        n++;
                    }
                }
            }
            mean_b /= n;
            mean_g /= n;
            mean_r /= n;
            clusterCentres.at(i) = Pixel(mean_b, mean_g, mean_r);
        }
    }

    double euclidian_distance(int x1, int y1, int z1, int x2, int y2, int z2)
    {
        return sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2) + pow(z1 - z2, 2));
    }

    float distance3d(float x1, float y1, float z1, float x2, float y2, float z2)
    { // https://www.geeksforgeeks.org/program-to-calculate-distance-between-two-points-in-3-d/
        float d = sqrt(pow(x2 - x1, 2) +
                    pow(y2 - y1, 2) +
                    pow(z2 - z1, 2) * 1.0);
        return d;
    }

    /* int randomy(int min, int max) // range : [min, max]
    {
        if (first) {
            srand(time(NULL)); // seeding for the first time only!
            first = false;
        }
        return min + rand() % ((max + 1) - min);
    } */
    
    static int my_random(int lim)
    {//https://stackify.dev/391611-c-generating-random-numbers-inside-loop
        std::uniform_int_distribution<int> uid{0, lim};                                              // help dre to generate nos from 0 to lim (lim included);
        return uid(dre);                                                                             // pass dre as an argument to uid to generate the random no
    }

    /* long random_at_most(long max)
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
    } */
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
        std::cout <<"Number of Channels: " <<n_ch << std::endl;
        if (image == NULL) {
            fprintf(stderr, "ERROR LOADING IMAGE: << Invalid file name or format >> \n");
            exit(EXIT_FAILURE);
        }
        printf("Image loaded => Number of channels : %d, width : %d, height :%d\n", n_ch, width, height);
        KMeans kmeans(image, width, height, n_ch, numberOfn_clus, iterations);
        //kmeans.train();
        kmeans.out(outputPath);

        stbi_image_free(image);
    } catch (TCLAP::ArgException &e) // catch exceptions
    {
        std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl;
    }
    return 0;
}
