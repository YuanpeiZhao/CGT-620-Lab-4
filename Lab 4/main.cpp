#include "ImageProcessor.cuh"
#include <iostream>

#define BLUR_SIZE 71
#define TOTAL_FRAME 1000

#define BLUR_TYPE 2 // 0 for rectangular, 1 for triangular, 2 for gaussian

using namespace std;

int main() {

	ImageProcessor processor;
	processor.Init(BLUR_SIZE, BLUR_TYPE, TOTAL_FRAME);
	while(processor.Process(true)){}

	cout << BLUR_SIZE << ":\t" << processor.getAvgTime() << endl;

	return 0;
}