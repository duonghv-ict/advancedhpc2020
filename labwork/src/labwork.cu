#include <stdio.h>
#include <include/labwork.h>
#include <cuda_runtime_api.h>
#include <omp.h>

#define ACTIVE_THREADS 16

int main(int argc, char **argv) {
    printf("USTH ICT Master 2018, Advanced Programming for HPC.\n");
    if (argc < 2) {
        printf("Usage: labwork <lwNum> <inputImage>\n");
        printf("   lwNum        labwork number\n");
        printf("   inputImage   the input file name, in JPEG format\n");
        return 0;
    }

    int lwNum = atoi(argv[1]);
    std::string inputFilename;

    int threshold = 10;

    // pre-initialize CUDA to avoid incorrect profiling
    printf("Warming up...\n");
    char *temp;
    cudaMalloc(&temp, 1024);

    Labwork labwork;
    if (lwNum != 2 ) {
        inputFilename = std::string(argv[2]);
        labwork.loadInputImage(inputFilename);

        threshold = atoi(argv[3]);
    }

    printf("Starting labwork %d\n", lwNum);
    Timer timer;
    timer.start();
    switch (lwNum) {
        case 1:
            labwork.labwork1_CPU();
            labwork.saveOutputImage("labwork2-cpu-out.jpg");
            printf("labwork 1 CPU ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
            timer.start();
            labwork.labwork1_OpenMP();
            printf("labwork 1 CPU omp ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
            labwork.saveOutputImage("labwork2-openmp-out.jpg");
            break;
        case 2:
            labwork.labwork2_GPU();
            break;
        case 3:
            for (int i = 1; i <= 64; ++i)
            {
                timer.start();
                labwork.labwork3_GPU(i);
                printf("labwork 3 GPU with blockSize=%d ellapsed %.1fms\n", i, lwNum, timer.getElapsedTimeInMilliSec());
            }
            
            labwork.saveOutputImage("labwork3-gpu-out.jpg");
            break;
        case 4:
            labwork.labwork4_GPU();
            labwork.saveOutputImage("labwork4-gpu-out.jpg");
            break;
        case 5:
            labwork.labwork5_CPU();
            labwork.saveOutputImage("labwork5-cpu-out.jpg");
            labwork.labwork5_GPU(FALSE);
            labwork.saveOutputImage("labwork5-gpu-out.jpg");
            break;
        case 6:
            labwork.labwork6_GPU_binarization(threshold);
            labwork.saveOutputImage("labwork6-gpu-out-binary.jpg");

            labwork.labwork6_GPU_brighness(threshold);
            labwork.saveOutputImage("labwork6-gpu-out-brightness.jpg");
            break;
        case 7:
            labwork.labwork7_GPU();
            printf("[ALGO ONLY] labwork %d ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
            labwork.saveOutputImage("labwork7-gpu-out.jpg");
            break;
        case 8:
            labwork.labwork8_GPU();
            printf("[ALGO ONLY] labwork %d ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
            labwork.saveOutputImage("labwork8-gpu-out.jpg");
            break;
        case 9:
            labwork.labwork9_GPU();
            printf("[ALGO ONLY] labwork %d ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
            labwork.saveOutputImage("labwork9-gpu-out.jpg");
            break;
        case 10:
            labwork.labwork10_GPU();
            printf("[ALGO ONLY] labwork %d ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
            labwork.saveOutputImage("labwork10-gpu-out.jpg");
            break;
    }
    printf("labwork %d ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
}

void Labwork::loadInputImage(std::string inputFileName) {
    inputImage = jpegLoader.load(inputFileName);
}

void Labwork::saveOutputImage(std::string outputFileName) {
    jpegLoader.save(outputFileName, outputImage, inputImage->width, inputImage->height, 90);
}

void Labwork::labwork1_CPU() {
    int pixelCount = inputImage->width * inputImage->height;
    outputImage = static_cast<char *>(malloc(pixelCount * 3));
    for (int j = 0; j < 100; j++) {     // let's do it 100 times, otherwise it's too fast!
        for (int i = 0; i < pixelCount; i++) {
            outputImage[i * 3] = (char) (((int) inputImage->buffer[i * 3] + (int) inputImage->buffer[i * 3 + 1] +
                                          (int) inputImage->buffer[i * 3 + 2]) / 3);
            outputImage[i * 3 + 1] = outputImage[i * 3];
            outputImage[i * 3 + 2] = outputImage[i * 3];
        }
    }
}

void Labwork::labwork1_OpenMP() {
    int pixelCount = inputImage->width * inputImage->height;
    outputImage = static_cast<char *>(malloc(pixelCount * 3));
    // do something here
    omp_set_num_threads(ACTIVE_THREADS);

    #pragma omp parallel for
    for (int j = 0; j < 100; j++) {     // let's do it 100 times, otherwise it's too fast!
        for (int i = 0; i < pixelCount; i++) {
            outputImage[i * 3] = (char) (((int) inputImage->buffer[i * 3] + (int) inputImage->buffer[i * 3 + 1] +
                                          (int) inputImage->buffer[i * 3 + 2]) / 3);
            outputImage[i * 3 + 1] = outputImage[i * 3];
            outputImage[i * 3 + 2] = outputImage[i * 3];
        }
    }
}

int getSPcores(cudaDeviceProp devProp) {
    int cores = 0;
    int mp = devProp.multiProcessorCount;
    switch (devProp.major) {
        case 2: // Fermi
            if (devProp.minor == 1) cores = mp * 48;
            else cores = mp * 32;
            break;
        case 3: // Kepler
            cores = mp * 192;
            break;
        case 5: // Maxwell
            cores = mp * 128;
            break;
        case 6: // Pascal
            if (devProp.minor == 1) cores = mp * 128;
            else if (devProp.minor == 0) cores = mp * 64;
            else printf("Unknown device type\n");
            break;
        default:
            printf("Unknown device type\n");
            break;
    }
    return cores;
}

void Labwork::labwork2_GPU() {
    int nDevices = 0;
    // get all devices
    cudaGetDeviceCount(&nDevices);
    printf("Number total of GPU : %d\n\n", nDevices);
    for (int i = 0; i < nDevices; i++){
        // get informations from individual device
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        // something more here
	printf("Device name is %s\n",prop.name);
	printf("Clock rate is %d\n",prop.clockRate);
	printf("Core count is %d\n",getSPcores(prop));
	printf("Multiprocessor count is %d\n",prop.multiProcessorCount);
	printf("warpSize is %d\n",prop.warpSize);
	printf("Memory info - clock rate is %d\n",prop.memoryClockRate);
	printf("Memory info - bus width is %d\n",prop.memoryBusWidth);
    }

}

__device__ int binarization(int input, int threshold) {
    int output;
    if (input >= threshold)
    {
        output = 255;
    }
    else
        output = 0;
    return output;
}

__global__ void grayscaleWithMapFunction(uchar3 *input, uchar3 *output, int threshold) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    // convert from rgb into grayscale
    output[tid].x = (input[tid].x + input[tid].y + input[tid].z) / 3;

    // convert to binary value
    output[tid].x = binarization(output[tid].x, threshold);

    output[tid].z = output[tid].y = output[tid].x;
}

__global__ void grayscale1D(uchar3 *input, uchar3 *output) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    output[tid].x = (input[tid].x + input[tid].y +
    input[tid].z) / 3;
    output[tid].z = output[tid].y = output[tid].x;
}

__global__ void grayscale2D(uchar3 *input, uchar3 *output, int width, int height) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if ((x >= width) || (y >= height))
        return;

    int r = blockDim.x * gridDim.x;
    x += r*y;
    

    output[x].x = (input[x].x + input[x].y + input[x].z) / 3;
    output[x].z = output[x].y = output[x].x;
}

__global__ void brightness(uchar3 *input, uchar3 *output, int threshold) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    output[tid].x = input[tid].x + threshold;
    if (output[tid].x > 255)
    {
        output[tid].x = 255;
    }

    output[tid].y = input[tid].y + threshold;
    if (output[tid].y > 255)
    {
        output[tid].y = 255;
    }

    output[tid].z = input[tid].z + threshold;
    if (output[tid].y > 255)
    {
        output[tid].y = 255;
    }
}

void Labwork::labwork3_GPU(int blockSize) {
    // Calculate number of pixels
    int pixelCount = inputImage->width * inputImage->height;
    outputImage = static_cast<char *>(malloc(pixelCount * 3));

    // Allocate CUDA memory
    uchar3 *devInput;
    uchar3 *devGray;

    cudaMalloc(&devInput, pixelCount * sizeof(uchar3));
    cudaMalloc(&devGray, pixelCount * sizeof(uchar3));    

    // Copy CUDA Memory from CPU to GPU
    char *hostInput = (char *) malloc(pixelCount * 3);
    // for (int i = 0; i < pixelCount; i++) {
    //         hostInput[i * 3] = (char) (int) inputImage->buffer[i * 3];;
    //         hostInput[i * 3 + 1] = (char) (int) inputImage->buffer[i * 3 + 1];
    //         hostInput[i * 3 + 2] = (char)  (int) inputImage->buffer[i * 3 + 2];
    //     }

    hostInput = (char *) inputImage->buffer;

    cudaMemcpy(devInput, hostInput, pixelCount * sizeof(uchar3), cudaMemcpyHostToDevice);

    // Processing
    // int blockSize = 8;
    int numBlock = pixelCount / blockSize;
    grayscale1D<<<numBlock, blockSize>>>(devInput, devGray);

    // Copy CUDA Memory from GPU to CPU
    cudaMemcpy(outputImage, devGray, pixelCount * sizeof(uchar3), cudaMemcpyDeviceToHost);

    // Cleaning
    cudaFree(devInput);
    cudaFree(devGray);
}

void Labwork::labwork4_GPU() {
    // Calculate number of pixels
    int pixelCount = inputImage->width * inputImage->height;
    outputImage = static_cast<char *>(malloc(pixelCount * 3));

    // Allocate CUDA memory
    uchar3 *devInput;
    uchar3 *devGray;

    cudaMalloc(&devInput, pixelCount * sizeof(uchar3));
    cudaMalloc(&devGray, pixelCount * sizeof(uchar3));    

    // Copy CUDA Memory from CPU to GPU
    char *hostInput = (char *) malloc(pixelCount * 3);
    // for (int i = 0; i < pixelCount; i++) {
    //         hostInput[i * 3] = (char) (int) inputImage->buffer[i * 3];;
    //         hostInput[i * 3 + 1] = (char) (int) inputImage->buffer[i * 3 + 1];
    //         hostInput[i * 3 + 2] = (char)  (int) inputImage->buffer[i * 3 + 2];
    //     }

    hostInput = (char *) inputImage->buffer;

    cudaMemcpy(devInput, hostInput, pixelCount * sizeof(uchar3), cudaMemcpyHostToDevice);

    // Processing
    dim3 blockSize = dim3(8,8);
    int grid_x;
    int grid_y;

    if(inputImage->width % blockSize.x == 0)
        grid_x = inputImage->width/blockSize.x;
    else
        grid_x = (int) (inputImage->width/blockSize.x + 1);

    if(inputImage->height % blockSize.y == 0)
        grid_y = inputImage->height/blockSize.y;
    else
        grid_y = (int) (inputImage->height/blockSize.y + 1);

    dim3 gridSize = dim3(grid_x, grid_y);
    //int gridSize = pixelCount / blockSize;
    grayscale2D<<<gridSize, blockSize>>>(devInput, devGray, inputImage->width, inputImage->height);

    // Copy CUDA Memory from GPU to CPU
    cudaMemcpy(outputImage, devGray, pixelCount * sizeof(uchar3), cudaMemcpyDeviceToHost);

    // Cleaning
    cudaFree(devInput);
    cudaFree(devGray);
}

void Labwork::labwork5_CPU() {
}

void Labwork::labwork5_GPU(bool shared) {
}

void Labwork::labwork6_GPU_binarization(int threshold) {
    // Calculate number of pixels
    int pixelCount = inputImage->width * inputImage->height;
    outputImage = static_cast<char *>(malloc(pixelCount * 3));

    // Allocate CUDA memory
    uchar3 *devInput;
    uchar3 *devGray;

    cudaMalloc(&devInput, pixelCount * sizeof(uchar3));
    cudaMalloc(&devGray, pixelCount * sizeof(uchar3));    

    // Copy CUDA Memory from CPU to GPU
    char *hostInput = (char *) malloc(pixelCount * 3);
    // for (int i = 0; i < pixelCount; i++) {
    //         hostInput[i * 3] = (char) (int) inputImage->buffer[i * 3];;
    //         hostInput[i * 3 + 1] = (char) (int) inputImage->buffer[i * 3 + 1];
    //         hostInput[i * 3 + 2] = (char)  (int) inputImage->buffer[i * 3 + 2];
    //     }

    hostInput = (char *) inputImage->buffer;

    cudaMemcpy(devInput, hostInput, pixelCount * sizeof(uchar3), cudaMemcpyHostToDevice);

    // Processing
    int blockSize = 8;
    int numBlock = pixelCount / blockSize;
    grayscaleWithMapFunction<<<numBlock, blockSize>>>(devInput, devGray, threshold);


    // Copy CUDA Memory from GPU to CPU
    cudaMemcpy(outputImage, devGray, pixelCount * sizeof(uchar3), cudaMemcpyDeviceToHost);

    // Cleaning
    cudaFree(devInput);
    cudaFree(devGray);
}

void Labwork::labwork6_GPU_brighness(int threshold) {
    // Calculate number of pixels
    int pixelCount = inputImage->width * inputImage->height;
    outputImage = static_cast<char *>(malloc(pixelCount * 3));

    // Allocate CUDA memory
    uchar3 *devInput;
    uchar3 *devGray;

    cudaMalloc(&devInput, pixelCount * sizeof(uchar3));
    cudaMalloc(&devGray, pixelCount * sizeof(uchar3));    

    // Copy CUDA Memory from CPU to GPU
    char *hostInput = (char *) malloc(pixelCount * 3);

    hostInput = (char *) inputImage->buffer;

    cudaMemcpy(devInput, hostInput, pixelCount * sizeof(uchar3), cudaMemcpyHostToDevice);

    // Processing
    int blockSize = 8;
    int numBlock = pixelCount / blockSize;
    brightness<<<numBlock, blockSize>>>(devInput, devGray, threshold);


    // Copy CUDA Memory from GPU to CPU
    cudaMemcpy(outputImage, devGray, pixelCount * sizeof(uchar3), cudaMemcpyDeviceToHost);

    // Cleaning
    cudaFree(devInput);
    cudaFree(devGray);
}

void Labwork::labwork7_GPU() {
}

void Labwork::labwork8_GPU() {
}

void Labwork::labwork9_GPU() {

}

void Labwork::labwork10_GPU(){
}


























