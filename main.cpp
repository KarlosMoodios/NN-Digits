/* Neural network - digits recognition
 Dr Andrew Watson and Mr Karl Moody 2020.
 http://yann.lecun.com/exdb/mnist/
 http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf
 http://www.neuro.nigmatec.ru/materials/themeid_17/riedmiller93direct.pdf */

#include <opencv2/core.hpp>
#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/ml.hpp"
#include <opencv2/dnn.hpp>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <conio.h>
#include <vector>
#include <windows.h>
using namespace cv;
using namespace ml;
using namespace dnn;
using namespace std;
//const int SZ = 20;		// Size of each image is SZ x SZ pixels greyscale (1 channel)		digits.png tile size
const int SZ = 28;		// Size of each image is SZ x SZ pixels greyscale (1 channel)		fashion.png tile size
const int CLASS_N = 10;	// Number of output classes, in this case '0','1','2',...,'8','9'

// Break image into individual tiles
static void mosaic(const int width, const vector<Mat>& images, Mat& grid)
{
	int mat_width = SZ * width;
	int mat_height = SZ * (int)ceil((double)images.size() / width);
	if (!images.empty())
	{
		grid = Mat(Size(mat_width, mat_height), images[0].type());
		for (size_t i = 0; i < images.size(); i++)
		{
			Mat location_on_grid = grid(Rect(SZ * ((int)i % width), SZ * ((int)i / width), SZ, SZ));
			images[i].copyTo(location_on_grid);
		}
	}
}

// Shuffle individual tiles
static void shuffle(vector<Mat>& digits, vector<int>& labels)
{
	vector<int> shuffled_indexes(digits.size());
	for (size_t i = 0; i < digits.size(); i++) shuffled_indexes[i] = (int)i;
	randShuffle(shuffled_indexes);//shuffle the test images
	vector<Mat> shuffled_digits(digits.size());
	vector<int> shuffled_labels(labels.size());
	for (size_t i = 0; i < shuffled_indexes.size(); i++)
	{
		shuffled_digits[shuffled_indexes[i]] = digits[i];
		shuffled_labels[shuffled_indexes[i]] = labels[i];
	}
	digits = shuffled_digits;
	labels = shuffled_labels;
}

// Deskew digits
static void deskew(const Mat& img, Mat& deskewed_img)
{
	Moments m = moments(img);

	if (abs(m.mu02) < 0.0001) { deskewed_img = img.clone(); return; }
	float skew = (float)(m.mu11 / m.mu02);
	float M_vals[2][3] = { {1, skew, -0.5f * SZ * skew}, {0, 1, 0} };
	Mat M(Size(3, 2), CV_32F);
	for (int i = 0; i < M.rows; i++)
	{
		for (int j = 0; j < M.cols; j++)
		{
			M.at<float>(i, j) = M_vals[i][j];
		}
	}
	warpAffine(img, deskewed_img, M, Size(SZ, SZ), WARP_INVERSE_MAP | INTER_LINEAR);
}

int main(void)
{
	Mat blob, raw;
	vector<Mat> digits, digitsraw;
	vector<int> labels;
	LARGE_INTEGER fr, t1, t2;

	printf("Loading training/test image mosaic...\n");
	//raw = imread("digits.png", IMREAD_GRAYSCALE);		// Handwritten digits image
	raw = imread("fashion.png", IMREAD_GRAYSCALE);		// Fashion image
	//imshow("Raw Image Mosaic",raw);
	
	// Split the mosaic into individual images and set labels
	int height = raw.rows;	// Get height of mosaic image
	int width = raw.cols;	// Get width of mosaic image
	int sx = SZ;			// Set width of individual tile
	int sy = SZ;			// Set height of individual tile
	digitsraw.clear();
	for (int i = 0; i < height; i += sy) {
		for (int j = 0; j < width; j += sx) 
			digitsraw.push_back(raw(Rect(j, i, sx, sy)));
	}

	// Store labels for each
	for (int i = 0; i < CLASS_N; i++) for (size_t j = 0; j < digitsraw.size() / CLASS_N; j++) labels.push_back(i);
	printf("Raw image %d x %d pixels\npreprocessing...\n", width, height);
	
	// Shuffle the tiles
	shuffle(digitsraw, labels);

	// Deskew the digits
	for (size_t i = 0; i < digitsraw.size(); i++)
	  {
      Mat deskewed_digit;
      deskew(digitsraw[i],deskewed_digit);
      digits.push_back(deskewed_digit);
	  }

	// Set the training and testing ratios
	int train_n = (int)(0.9 * digits.size());
	int test_n = (int)(digits.size() - train_n);
	cout << "Total images    =" << digits.size() << endl;
	cout << "Training images =" << train_n << endl;
	cout << "Testing images  =" << test_n << endl;
	
	// Image for training images
	Mat train_set;
	vector<Mat> digits_train(digits.begin(), digits.begin() + train_n);
	mosaic(50, digits_train, train_set);
	//imshow("train set", train_set);
	vector<int> labels_train(labels.begin(), labels.begin() + train_n);
	
	// Image for test images
	Mat test_set;
	vector<Mat>digits_test(digits.begin() + train_n, digits.end());
	mosaic(25, digits_test, test_set);
	//imshow("test set", test_set);
	vector<int> labels_test(labels.begin() + train_n, labels.end());

	// Convert training array of images to 32 bit floating point
	Mat temp;
	Mat inputtrainingdata;
	vector <Mat> digits_trainf;
	for (int i = 0; i < train_n; i++)
	{
		digits_train[i].convertTo(temp, CV_32F, 1.0 / 255.0);
		digits_trainf.push_back(temp);
		Mat DataInOneRow = digits_trainf[i].reshape(0, 1);
		inputtrainingdata.push_back(DataInOneRow);
	}
	cout << "First training image is a " << labels_train[0] << endl;
	
	// Convert testing array of images to 32 bit floating point
	Mat inputtestdata;
	vector <Mat> digits_testf;
	for (int i = 0; i < test_n; i++)
	{
		digits_test[i].convertTo(temp, CV_32F, 1.0 / 255.0);
		digits_testf.push_back(temp);
		Mat DataInOneRow = digits_testf[i].reshape(0, 1);
		inputtestdata.push_back(DataInOneRow);
	}
	cout << "First test image is a " << labels_test[0]<<endl;

	// Convert output results to floating point with ouput=1.0 for correct digit
	Mat outputtrainingdata(0, CLASS_N, CV_32FC1);
	for (int i = 0; i < train_n; i++)
	{
		vector<float> outputTraningVector(CLASS_N);
		fill(outputTraningVector.begin(), outputTraningVector.end(), 0.0);
		outputTraningVector[labels_train[i]] = 1.0;
		Mat tempMatrix(outputTraningVector, false);
		outputtrainingdata.push_back(tempMatrix.reshape(0, 1));
	}
	
	// Set up the NN
	int epochs = 10;	// Runtime
	const int h1 = 15;	// Hidden layer 1
	const int h2 = 30;	// Hidden layer 2
	const int h3 = 15;	// Hidden layer 3

	Ptr<ANN_MLP>nndigits = ANN_MLP::create();
	Mat layersSize = Mat(4, 1, CV_32F);
	layersSize.row(0) = Scalar(inputtrainingdata.cols);		// Input layer
	layersSize.row(1) = Scalar(h1);
	layersSize.row(2) = Scalar(h2);
	layersSize.row(3) = Scalar(h3);
	layersSize.row(3) = Scalar(outputtrainingdata.cols);	// Output layer

	// Create the NN
	nndigits->setLayerSizes(layersSize);
	nndigits->setActivationFunction(ANN_MLP::ActivationFunctions::SIGMOID_SYM); // Activation functions, SIGMOID_SYM, GAUSSIAN, RELU, LEAKYRELU
	TermCriteria termCrit = TermCriteria(TermCriteria::Type::COUNT + TermCriteria::Type::EPS, 1, 0.001);
	nndigits->setTermCriteria(termCrit);
	nndigits->setTrainMethod(ANN_MLP::TrainingMethods::BACKPROP, 0.01, 0.01);
	//nndigits->setTrainMethod(ANN_MLP::TrainFlags::UPDATE_WEIGHTS);
	Ptr<TrainData> trainingData = TrainData::create(inputtrainingdata, SampleTypes::ROW_SAMPLE, outputtrainingdata);
	
	// Start training
	printf("Training...\n");
	printf("Epoch		Time/secs	Error %  %\n");
	double lowErr = 100;
	for (int i = 0; i < epochs; i++)
	{
		QueryPerformanceCounter(&t1);
		nndigits->train(trainingData);
		QueryPerformanceCounter(&t2);
		QueryPerformanceFrequency(&fr);
		double t = (t2.QuadPart - t1.QuadPart) / (double)fr.QuadPart;
		
		// Test the NN
		cout << "\t" << i << "\t" << t << "";
		Mat result;
		nndigits->predict(inputtestdata, result);

		//Test the trained network
		float best, error;
		int bint, correct = 0;
		for (int i = 0; i < result.size().height; i++)
		{
			best = result.at<float>(i, 0);
			bint = 0;
			for (int j = 1; j < result.size().width; j++)
			{
				if (result.at<float>(i, j) > best) { 
					best = result.at<float>(i, j); bint = j;
				}
			}
			if (bint == labels_test[i]) correct++;
		}
		// Check error of each epoch
		error = (100.0 - ((float)correct / result.size().height) * 100.0);
		printf("\t%.2f\n", error);

		// Save lowest error achieved
		if (error < lowErr) {
			lowErr = error;
		}
		
	}// End of epoch loop

	// Print the lowest error
	printf("\nThe lowest error rate achieved through %i Epochs\n", epochs);
	printf("Lowest error: \t%.2f %\n", lowErr);

	// Save the results to file
	//cout << "Saving NN as \"digits.xml\"..." << endl;
	//nndigits->save("digits.xml");
	//cout << "Saving NN as \"fashion.xml\"..." << endl;
	//nndigits->save("fashion.xml");

	return 0;
}