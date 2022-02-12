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
#include <string>
#include <random>
#include <vector>
#include <sys/stat.h>
#include <limits>

#include <tclap/CmdLine.h>
#include "FreeImage.h"

#define errorexit(errcode, errstr) \
    fprintf(stderr, "%s: &d\n", errstr, errcode); \
    exit(1);

#define NUMOFCHANELS 4
/*class InputParser{
    // https://stackoverflow.com/questions/865668/parsing-command-line-arguments-in-c?page=1&tab=votes#tab-top
    public:
        InputParser (int &argc, char **argv){
            for (int i=1; i < argc; ++i)
                this->tokens.push_back(std::string(argv[i]));
        }
        const std::string& getCmdOption(const std::string &option) const{
            std::vector<std::string>::const_iterator itr;
            itr =  std::find(this->tokens.begin(), this->tokens.end(), option);
            if (itr != this->tokens.end() && ++itr != this->tokens.end()){
                return *itr;
            }
            static const std::string empty_string("");
            return empty_string;
        }
        bool cmdOptionExists(const std::string &option) const{
            return std::find(this->tokens.begin(), this->tokens.end(), option)
                   != this->tokens.end();
        }
    private:
        std::vector <std::string> tokens;
};*/

long GetFileSize(std::string filename)
{//https://stackoverflow.com/questions/5840148/how-can-i-get-a-files-size-in-c
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
    int clusters;
    int width;
	int height;
	int pitch;
    long numberOfPixels;
    int *labels;
    double *centers;
    double *distances;
    int iterations;
    bool changes = false;
    bool first;

  public:
    // Intialise cluster centres as random pixels from the image
    KMeans(unsigned char *image, int width, int height, int pitch, int clusters, int iterations)
    {
        this->image = image;
        this->width = width;
        this->height = height;
        this->pitch = pitch;
        this->clusters = clusters;
        this->iterations = iterations;

        this->numberOfPixels = width * height;

        this->labels = new int[numberOfPixels];
        this->centers = new double[NUMOFCHANELS * clusters];
        this->distances = new double[numberOfPixels];

        this->first = true;
        initialize_clusters();
        
    }

    void train()
    {   
        std::cout << "Training...\n";
        for (size_t i = 0; i < iterations; i++)
        {
            label_pixels();
            /*if (changes == false)
            {
                break;
            }*/
            computeCentroids();
            std::cout << "Train step " << i << " done" << std::endl;
        }
        delete(labels);
        delete(centers);
        delete(distances);
    }

    void out(std::string outputPath)
    {   
        int minimum_cluster;
        for (size_t index_pixel = 0; index_pixel < numberOfPixels; index_pixel++)
        {
            minimum_cluster = labels[index_pixel];

            for (size_t index_channel = 0; index_channel < NUMOFCHANELS-1; index_channel++)
            {
                image[index_pixel * NUMOFCHANELS + index_channel] = (u_char)round(centers[minimum_cluster * NUMOFCHANELS + index_channel]);
                image[index_pixel * NUMOFCHANELS + 3] = 255;
            }
        }
        
        std::cout << "Saving Image" << std::endl;
        //shranimo sliko
        FIBITMAP *dst = FreeImage_ConvertFromRawBits(image, width, height, pitch,
            32, FI_RGBA_RED_MASK, FI_RGBA_GREEN_MASK, FI_RGBA_BLUE_MASK, TRUE);
        FreeImage_Save(FIF_PNG, dst, outputPath.c_str(), 0);
    }

  private:
    void initialize_clusters(){
        for (size_t index_cluster = 0; index_cluster < clusters; index_cluster++)
        {
            int rnd = my_random(0, numberOfPixels);
            for (size_t index_channel = 0; index_channel < NUMOFCHANELS-1; index_channel++)
            {
                //TODO: Preveri če je pravilni vsrtni ali je RGBA ali BGRA  pa če rabimo tale A
                centers[index_cluster * NUMOFCHANELS + index_channel] = image[rnd * NUMOFCHANELS + index_channel];
            }
        }
    }

    void label_pixels(){
        double minimum_distance;
        bool changesTemp = false;
        int furthest_pixel;

        for (size_t index_pixel = 0; index_pixel < numberOfPixels; index_pixel++)
        {
            int centroidLabel = 0;
            minimum_distance = std::numeric_limits<double>::max(); //nastavimo na MAX
            for (size_t index_cluster = 0; index_cluster < clusters; index_cluster++)
            {
                double distance = euclidian_distance(centers[index_cluster * NUMOFCHANELS], centers[index_cluster * NUMOFCHANELS + 1], centers[index_cluster * NUMOFCHANELS + 2], 
                                                        image[index_pixel * NUMOFCHANELS], image[index_pixel * NUMOFCHANELS + 1], image[index_pixel * NUMOFCHANELS + 2]); // ! Možno da je napaka, preveč računanja zame
                
                if (distance < minimum_distance)
                {
                    minimum_distance = distance;
                    centroidLabel = index_cluster;
                }
                
            }

            if (labels[index_pixel] != centroidLabel)
            {
                labels[index_pixel] = centroidLabel;
                changesTemp = true;
            }
            
            distances[index_pixel] = minimum_distance;
        }
        //changes
        changes = changesTemp;
    }

    void computeCentroids()
    { //Compute mean
        int *cluster_counter = new int[clusters];
        int minimum_clusters;
        int furthest_pixel;
        //Reset counter and centers
        for (size_t index_cluster = 0; index_cluster < clusters; index_cluster++)
        {
            cluster_counter[index_cluster] = 0;
            for (size_t index_channel = 0; index_channel < NUMOFCHANELS-1; index_channel++)
            {
                centers[index_cluster * NUMOFCHANELS + index_channel] = 0;
            }  
        }

        for (size_t index_pixel = 0; index_pixel < numberOfPixels; index_pixel++)
        {
            minimum_clusters = labels[index_pixel];
            for (size_t index_channel = 0; index_channel < NUMOFCHANELS-1; index_channel++)
            {
                centers[minimum_clusters * NUMOFCHANELS + index_channel] += image[index_pixel * NUMOFCHANELS + index_channel];
            }
            cluster_counter[minimum_clusters]++;
        }

        for (size_t index_cluster = 0; index_cluster < clusters; index_cluster++)
        {
            if (cluster_counter[index_cluster] > 0)
            {
                for (size_t index_channel = 0; index_channel < NUMOFCHANELS-1; index_channel++)
                {
                    centers[index_cluster * NUMOFCHANELS * index_channel] = centers[index_cluster * NUMOFCHANELS * index_channel] / cluster_counter[index_cluster];
                }
            }else{ // Če je cluster empty
                int max_distance = 0;
                for (size_t index_pixel = 0; index_pixel < numberOfPixels; index_pixel++)
                {
                    if (distances[index_pixel] > max_distance)
                    {
                        max_distance = distances[index_pixel];
                        furthest_pixel = index_pixel;
                    }
                }
                
                for (size_t index_channel = 0; index_channel < NUMOFCHANELS-1; index_channel++)
                {
                    centers[index_cluster * NUMOFCHANELS * index_channel] = image[furthest_pixel * NUMOFCHANELS * index_channel];
                }
                
                distances[furthest_pixel] = 0;
            }
            
        }
        delete(cluster_counter);
    }

    /**
     * @brief Calculate Euclidian Distance
     * 
     * @param x1
     * @param y1 
     * @param z1 
     * @param x2 
     * @param y2 
     * @param z2 
     * @return double
     */
    double euclidian_distance(int x1, int y1, int z1, int x2, int y2, int z2)
    {
        return sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2) + pow(z1- z2, 2));
    }

    int my_random(int min, int max) //range : [min, max]
    {
        if (first) 
        {  
            srand(time(NULL)); //seeding for the first time only!
            first = false;
        }
        return min + rand() % (( max + 1 ) - min);
    }

    long random_at_most(long max) {
        unsigned long
            // max <= RAND_MAX < ULONG_MAX, so this is okay.
            num_bins = (unsigned long) max + 1,
            num_rand = (unsigned long) RAND_MAX + 1,
            bin_size = num_rand / num_bins,
            defect   = num_rand % num_bins;

        long x;
        do {
        x = random();
        }
        // This is carefully written not to overflow
        while (num_rand - defect <= (unsigned long)x);

        // Truncated division is intentional
        return x/bin_size;
    }                        
};

int main(int argc, char const *argv[])
{
	try {  

	//Last line in help, delimiter, version number
	TCLAP::CmdLine cmd("Serial k-Means image compressor", ' ', "0.1");

	// Define a value argument and add it to the command line.
	TCLAP::ValueArg<std::string> inputArg("i","input","Path to input photo",true,"image.png","string", cmd);
    TCLAP::ValueArg<std::string> outputArg("o","output","Path to output photo",true,"image_compressed.png","string", cmd);
    TCLAP::ValueArg<int> clutstersArg("k","clusters","Number of Clusters",false, 4,"int", cmd);
    TCLAP::ValueArg<int> iterationsArg("t","iterations","Number of iterations",false, 150,"int", cmd);
	// Parse the argv array.
	cmd.parse( argc, argv );

	// Get the value parsed by each arg. 
	std::string inputPath = inputArg.getValue();
    std::string outputPath = outputArg.getValue();
    int numberOfClusters = clutstersArg.getValue();
    int iterations = iterationsArg.getValue();

    //std::cout << inputPath << std::endl;
    //std::cout << outputPath << std::endl;
    //std::cout << numberOfClusters << std::endl;
    //std::cout << iterations << std::endl;

	// Do what you intend. 

    //Load image from file
	FIBITMAP *imageBitmap = FreeImage_Load(FIF_JPEG, inputPath.c_str(), 0);
    if(imageBitmap == NULL){
        printf("Error loading Image\n");
        return(99);
    }
	//Convert it to a 32-bit image
    FIBITMAP *imageBitmap32 = FreeImage_ConvertTo32Bits(imageBitmap);
	
    //Get image dimensions
    int width = FreeImage_GetWidth(imageBitmap32);
	int height = FreeImage_GetHeight(imageBitmap32);
	int pitch = FreeImage_GetPitch(imageBitmap32);
    // calculate the number of bytes per pixel
    unsigned bytespp = FreeImage_GetLine(imageBitmap32) / FreeImage_GetWidth(imageBitmap32);
    // calculate the number of samples per pixel
    unsigned samples = bytespp / sizeof(BYTE);

    //Allcoate space for image
    unsigned char *image = (unsigned char *)malloc(height * pitch * sizeof(unsigned char));

    //Kopiraj vsebino v alociran prostor
	FreeImage_ConvertToRawBits(image, imageBitmap, pitch, 32, FI_RGBA_RED_MASK, FI_RGBA_GREEN_MASK, FI_RGBA_BLUE_MASK, TRUE);

    //Slik ne potrebujemo več zato jih unloudamo
	FreeImage_Unload(imageBitmap32);
	FreeImage_Unload(imageBitmap);

    KMeans kmeans(image, width, height, pitch, numberOfClusters, iterations);
    kmeans.train();
    kmeans.out(outputPath);

	} catch (TCLAP::ArgException &e)  // catch exceptions
	{ std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl; }
    return 0;
}
