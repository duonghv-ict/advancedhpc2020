#pragma once

#include <include/jpegloader.h>
#include <include/timer.h>

class Labwork {

private:
    JpegLoader jpegLoader;
    JpegInfo *inputImage;
    char *outputImage;

public:
    void loadInputImage(std::string inputFileName);
    void saveOutputImage(std::string outputFileName);

    void labwork1_CPU();
    void labwork1_OpenMP();

    void labwork2_GPU();

    void labwork3_GPU(int i);

    void labwork4_GPU();

    void labwork5_CPU();
    void labwork5_GPU(bool shared);

    void labwork6_GPU_binarization(int threshold);
    void labwork6_GPU_brighness(int threshold);
    void labwork6_GPU_blending(float weight);

    void labwork7_GPU();

    void labwork8_GPU();

    void labwork9_GPU();

    void labwork10_GPU();
};
