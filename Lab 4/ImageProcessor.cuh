#ifndef IMAGE_PROCESSOR_CUH
#define IMAGE_PROCESSOR_CUH

class ImageProcessor {

private:
	unsigned char *h_imgIn, *h_imgOut;
	unsigned char *d_imgIn, *d_imgOut;
	int imgWidth, imgHeight;
	int components = 0;
	int requiredComponents = 3;
	int imgPtr;
	int blurSize;
	int blurType;
	int currFrame;
	int totalFrame;
	float totalTime = 0.0f;

	void LoadImage(bool addFrame);
	void SaveImage();
	void TransferDataToDevice();

public:
	void Init(int n, int blurType, int frameNum);
	bool Process(bool process);
	float getAvgTime();
};

#endif
