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
#include <iostream>
#include <string>
#include <vector>
#include <tclap/CmdLine.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

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

    //Loading input image
	int width, height, channels;
    unsigned char *idata;
    if((idata = stbi_load(inputPath.c_str(), &width, &height, &channels, 0)) == NULL){
        printf("Error in loading the image\n");
        exit (99);
    }



	} catch (TCLAP::ArgException &e)  // catch exceptions
	{ std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl; }
    return 0;
}
