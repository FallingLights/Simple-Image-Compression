/**
 * @file serial.cpp
 * @author AG
 * @brief 
 * @version 0.1
 * @date 2022-02-10
 * 
 * @copyright Copyright (c) 2022
 * 
 * c++ -o serial -I include/ serial.cpp
 */
#include <algorithm>
#include <chrono>
#include <iostream>
#include <string>
#include <random>
#include <vector>

#include <tclap/CmdLine.h>
#include "FreeImage.h"

#define errorexit(errcode, errstr) \
    fprintf(stderr, "%s: &d\n", errstr, errcode); \
    exit(1);

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

class KMeans
{
  private:
    unsigned char *image;
    int clusters;
    int width;
	int height;
	int pitch;
    int numberOfPixels;
    int *labels;
    double *centers;
    double *distances;

  public:
    // Intialise cluster centres as random pixels from the image
    KMeans(unsigned char *image, int width, int height, int pitch, int clusters, int iterations)
    {
        this->image = image;
        this->width = width;
        this->height = height;
        this->pitch = pitch;
        this->clusters = clusters;

        this->numberOfPixels = width * height;

        this->labels = new int[numberOfPixels];
        this->centers = new double[4 * clusters];
        this->distances = new double[numberOfPixels];

        initialize_clusters();

        for (size_t i = 0; i < iterations; i++)
        {
            label_pixels();
            computeCentroids();
        }
        
    }

  private:
    void initialize_clusters(){
        for (size_t index_cluster = 0; index_cluster < clusters; index_cluster++)
        {
            int rnd = random(0, numberOfPixels);
            for (size_t index_channel = 0; index_channel < 4; index_channel++)
            {
                //TODO: Preveri če je pravilni vsrtni ali je RGBA ali BGRA  pa če rabimo tale A
                centers[index_cluster * 4 + index_channel] = image[rnd * 4 + index_channel];
            }
        }
    }
    void label_pixels(){
        double minimum_distance;
        for (size_t index_pixel = 0; index_pixel < numberOfPixels; index_pixel++)
        {
            int centroidLabel = 0;
            minimum_distance = std::numeric_limits<double>::max(); //nastavimo na MAX
            for (size_t index_cluster = 0; index_cluster < clusters; index_cluster++)
            {
                double distance = euclidian_distance(centers[index_cluster * 4 + 0], centers[index_cluster * 4 + 1], centers[index_cluster * 4 + 2], 
                                                        image[index_pixel * 4 + 0], image[index_pixel * 4 + 1], image[index_pixel * 4 + 2]); // ! Možno da je napaka, preveč računanja zame
                
                if (distance < minimum_distance)
                {
                    minimum_distance = distance;
                    centroidLabel = index_cluster;
                    labels[index_pixel] = centroidLabel;
                }
                
            }
            distances[index_pixel] = minimum_distance;
        }
        //changes
    }
    void computeCentroids()
    { //Compute mean
        for (size_t index_cluster = 0; index_cluster < clusters; index_cluster++)
        {

        }
    }
    /**
     * @brief Calculate euclidian distance
     * 
     * @param dim Number of Dimensions
     * @param p1 vector p1
     * @param p2 vector p2
     * @return double Distance
     */
    double euclidian_distance(int x1, int y1, int z1, int x2, int y2, int z2)
    {
        return sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2) + pow(z1- z2, 2));
    }

    static int random(int min, int max) //range : [min, max]
    {
        static bool first = true;
        if (first) 
        {  
            srand(time(NULL)); //seeding for the first time only!
            first = false;
        }
        return min + rand() % (( max + 1 ) - min);
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

	// Parse the argv array.
	cmd.parse( argc, argv );

	// Get the value parsed by each arg. 
	std::string inputPath = inputArg.getValue();
    std::string outputPath = outputArg.getValue();
    int numberOfClusters = clutstersArg.getValue();

	// Do what you intend. 

    //Load image from file
	FIBITMAP *imageBitmap = FreeImage_Load(FIF_BMP, inputPath.c_str(), 0);
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

    KMeans kmeans(image, width, height, pitch, numberOfClusters);




	} catch (TCLAP::ArgException &e)  // catch exceptions
	{ std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl; }
    return 0;
}
