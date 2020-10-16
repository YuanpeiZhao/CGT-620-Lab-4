#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "ImageProcessor.cuh"

#include <stdio.h>
#include <string>
#include <iostream>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using namespace std;

__global__ void  ImgProcKernelRectangular(unsigned char* d_imgIn, unsigned char* d_imgOut, int blurSize, int imgPtr, int width, int height, int components)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col >= width || row >= height) return;

    int offset = components * (row * width + col);
    int imgSize = components * width * height;
    float weight = 1.0f / (float)blurSize;

    float r = 0.0f, g = 0.0f, b = 0.0f;

    for (int i = 0; i < blurSize; i++) {
        int externalOffset = ((imgPtr + i) % blurSize) * imgSize;

        float tmp = (float)d_imgIn[externalOffset + offset] * weight;
        r += tmp;

        tmp = (float)d_imgIn[externalOffset + offset + 1] * weight;
        g += tmp;

        tmp = (float)d_imgIn[externalOffset + offset + 2] * weight;
        b += tmp;

    }
    d_imgOut[offset] = (unsigned char)r;
    d_imgOut[offset + 1] = (unsigned char)g;
    d_imgOut[offset + 2] = (unsigned char)b;
}

__global__ void  ImgProcKernelTriangular(unsigned char* d_imgIn, unsigned char* d_imgOut, int blurSize, int imgPtr, int width, int height, int components)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col >= width || row >= height) return;

    int offset = components * (row * width + col);
    int imgSize = components * width * height;

    float total = 0.0f;
    float r = 0.0f, g = 0.0f, b = 0.0f;
    int center = blurSize / 2;
    
    for (int i = 0; i < blurSize; i++) {
        int externalOffset = ((imgPtr + i) % blurSize) * imgSize;

        float weight = 1.0f - (float)abs(center - i) / center;
        total += weight;

        float tmp = (float)d_imgIn[externalOffset + offset] * weight;
        r += tmp;

        tmp = (float)d_imgIn[externalOffset + offset + 1] * weight;
        g += tmp;

        tmp = (float)d_imgIn[externalOffset + offset + 2] * weight;
        b += tmp;

    }
    r /= total;
    g /= total;
    b /= total;

    d_imgOut[offset] = (unsigned char)r;
    d_imgOut[offset + 1] = (unsigned char)g;
    d_imgOut[offset + 2] = (unsigned char)b;
}

__global__ void  ImgProcKernelGaussian(unsigned char* d_imgIn, unsigned char* d_imgOut, int blurSize, int imgPtr, int width, int height, int components)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col >= width || row >= height) return;

    int offset = components * (row * width + col);
    int imgSize = components * width * height;

    float total = 0.0f;
    float r = 0.0f, g = 0.0f, b = 0.0f;
    float miu = (float)(blurSize / 2);
    float sig = miu * (miu + 1) * (2 * miu + 1) / (3 * blurSize);
    for (int i = 0; i < blurSize; i++) {
        int externalOffset = ((imgPtr + i) % blurSize) * imgSize;
        float weight = exp(-(i - miu) * (i - miu) / (2 * sig)) / sqrt(sig * 6.28f);
        total += weight;

        float tmp = (float)d_imgIn[externalOffset + offset] * weight;
        r += tmp;

        tmp = (float)d_imgIn[externalOffset + offset + 1] * weight;
        g += tmp;

        tmp = (float)d_imgIn[externalOffset + offset + 2] * weight;
        b += tmp;

    }
    r /= total;
    g /= total;
    b /= total;

    d_imgOut[offset] = (unsigned char)r;
    d_imgOut[offset + 1] = (unsigned char)g;
    d_imgOut[offset + 2] = (unsigned char)b;
}

void ImageProcessor::Init(int n, int t, int frameNum) {
    if (n % 2 == 0) {
        fprintf(stderr, "The blur size must be odd!\n");
        exit(-1);
    }

    blurSize = n;
    blurType = t; 
    totalFrame = frameNum;
    currFrame = 0;
    imgPtr = 0;
    requiredComponents = 3;

    LoadImage(false);
    int imgSize = imgWidth * imgHeight * components;

    // Allocate CPU imgae out buffer 
    h_imgOut = (unsigned char*)malloc(imgSize * sizeof(unsigned char));
    if (h_imgOut == NULL) {
        cout << "malloc failed" << endl;
        exit(-1);
    }

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaError_t cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!\nDo you have a CUDA-capable GPU installed?");
        exit(-1);
    }

    // Allocate GPU buffers
    cudaStatus = cudaMalloc((void**)&d_imgIn, blurSize * imgSize * sizeof(unsigned char));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        exit(-1);
    }

    cudaStatus = cudaMalloc((void**)&d_imgOut, imgSize * sizeof(unsigned char));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        exit(-1);
    }

    for (int i = 0; i < blurSize-1; i++) {
        Process(false);
    }   
}

bool ImageProcessor::Process(bool process) {

    if (currFrame >= totalFrame) return false;

    LoadImage(true);

    cudaEvent_t startT, stopT;
    float time;

    if (process) {
        cudaEventCreate(&startT);
        cudaEventCreate(&stopT);
        cudaEventRecord(startT, 0);
    }

    TransferDataToDevice();
    imgPtr = (imgPtr + 1) % blurSize;

    if (process) {

        int imgSize = imgWidth * imgHeight * components;

        // Launch a kernel on the GPU.
        const int TILE = 16;
        dim3 dimGrid(ceil((float)imgWidth / TILE), ceil((float)imgHeight / TILE));
        dim3 dimBlock(TILE, TILE, 1);
        if(blurType == 0) ImgProcKernelRectangular <<<dimGrid, dimBlock >>> (d_imgIn, d_imgOut, blurSize, imgPtr, imgWidth, imgHeight, components);
        else if (blurType == 1) ImgProcKernelTriangular <<<dimGrid, dimBlock >>> (d_imgIn, d_imgOut, blurSize, imgPtr, imgWidth, imgHeight, components);
        else if (blurType == 2) ImgProcKernelGaussian <<<dimGrid, dimBlock >>> (d_imgIn, d_imgOut, blurSize, imgPtr, imgWidth, imgHeight, components);

        // Check for any errors launching the kernel
        cudaError_t cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
            exit(-1);
        }

        // cudaDeviceSynchronize waits for the kernel to finish, and returns
        // any errors encountered during the launch.
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching kernel!\n", cudaStatus);
            exit(-1);
        }

        // Copy output image from GPU buffer to host memory.
        cudaStatus = cudaMemcpy(h_imgOut, d_imgOut, imgSize * sizeof(unsigned char), cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "D2H cudaMemcpy failed!");
            exit(-1);
        }

        cudaEventRecord(stopT, 0);
        cudaEventSynchronize(stopT);
        cudaEventElapsedTime(&time, startT, stopT);
        cudaEventDestroy(startT);
        cudaEventDestroy(stopT);

        cout << "iter " << currFrame << ":\t" << time << endl;
        totalTime += time / (totalFrame - blurSize + 1);

        SaveImage();
    }
    return true;
}

void ImageProcessor::LoadImage(bool addFrame) {

    string fileNum = to_string(currFrame);
    while (fileNum.size() < 4) fileNum = "0" + fileNum;
    string fileName = "images/source" + fileNum +".png";

    h_imgIn = stbi_load(fileName.c_str(), &(imgWidth), &(imgHeight), &components, requiredComponents);
    if (!h_imgIn) {
        fprintf(stderr, "Cannot read input image, invalid path?\n");
        exit(-1);
    }
    if (addFrame) currFrame++;
}

void ImageProcessor::SaveImage() {
    //save the output image
    string fileNum = to_string(currFrame - 1);
    while (fileNum.size() < 4) fileNum = '0' + fileNum;
    string fileName = "results_" + to_string(blurSize) + "/" + fileNum +".png";
    
    int result = stbi_write_png(fileName.c_str(), imgWidth, imgHeight, components, h_imgOut, 0);
    if (!result) {
        cout << "Something went wrong during writing. Invalid path?" << endl;
        exit(-1);
    }
}

void ImageProcessor::TransferDataToDevice() {

    int imgSize = imgWidth * imgHeight * components;

    // Copy input image from host memory to GPU buffers.
    cudaError_t cudaStatus = cudaMemcpy(d_imgIn + imgSize * imgPtr, h_imgIn, imgSize * sizeof(unsigned char), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "H2D cudaMemcpy failed!");
        exit(1);
    }
    free(h_imgIn);
}

float ImageProcessor::getAvgTime() {
    return totalTime;
}