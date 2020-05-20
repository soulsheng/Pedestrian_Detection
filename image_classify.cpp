#include <iostream>
#include <fstream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/ml/ml.hpp>
#include "dataset.h" // 定义一些数据
#include "my_svm.h" // MySVM继承自CvSVM的类
#include "common.h"
#include "time.h"

using namespace std;
using namespace cv;

#define	TEST_POS	1
#if TEST_POS
#define TEST_FILE_LIST	"../dataset/fileNamePosT2.txt"
#define TEST_FILE_PATH	"../dataset/posT2/"
#else
#define TEST_FILE_LIST	"../dataset/fileNameNegT2.txt"
#define TEST_FILE_PATH	"../dataset/negT2/"
#endif

typedef std::vector<std::string> NameVec;

int main(int argc, char const *argv[])
{
	MySVM svm;//SVM分类器
	svm.load(SVM_FILE);//从XML文件读取训练好的SVM模型
	int DescriptorDim = svm.get_var_count();//特征向量的维数，即HOG描述子的维数

	string ImgName;//图片名
	ifstream ifList(TEST_FILE_LIST);//测试样本图片的文件名列表

	std::vector<std::string> fileList;
	while (getline(ifList, ImgName))
		fileList.push_back(ImgName);

	Mat testImg;
	vector<float> descriptor;
	HOGDescriptor hog(Size(HOG_WIDTH, HOG_HEIGHT), Size(24, 24), Size(8, 8), Size(8, 8), 9);//HOG检测器，用来计算HOG描述子的
	Mat testFeatureMat = Mat::zeros(1, DescriptorDim, CV_32FC1);//测试样本的特征向量矩阵

	NameVec	resultFileList[2]; // 0-neg, 1-pos
	long timeStep[4] = { 0 }; // 0-read, 1-resize, 2-hog, 3-pred

	long tBegAll = clock();
	for (int num = 0; num < fileList.size(); num++)
	{
		//cout << " 处理： " << fileList[num] << endl;
		//ImgName = "D:\\DataSet\\PersonFromVOC2012\\" + ImgName;//加上正样本的路径名
		ImgName = TEST_FILE_PATH + fileList[num];//加上正样本的路径名

		/******************读入单个测试图并对其HOG描述子进行分类*********************/
		////读取测试图片，并计算其HOG描述子
		long tBegFrameAll = clock();
		long tBeg = clock();
		testImg = imread(ImgName);
		timeStep[0] += clock() - tBeg;

		tBeg = clock();
		resize(testImg, testImg, Size(HOG_WIDTH, HOG_HEIGHT));
		timeStep[1] += clock() - tBeg;

		tBeg = clock();
		hog.compute(testImg, descriptor, Size(8, 8));//计算HOG描述子，检测窗口移动步长(8,8)
		timeStep[2] += clock() - tBeg;
		
		tBeg = clock();
		//将计算好的HOG描述子复制到testFeatureMat矩阵中
		for (int i = 0; i < descriptor.size(); i++)
			testFeatureMat.at<float>(0, i) = descriptor[i];

		//用训练好的SVM分类器对测试图片的特征向量进行分类
		int result = svm.predict(testFeatureMat);//返回类标
		timeStep[3] += clock() - tBeg;

		cout << "1 frame ms time = " << clock() - tBegFrameAll << endl;

		//cout << "分类结果：" << result << endl;
		cout << ".";

		if (1 == result)
			resultFileList[1].push_back(fileList[num]);
		else
			resultFileList[0].push_back(fileList[num]);

	}
	cout << "all frame ms time = " << clock() - tBegAll << endl;

	// predict error list to output 
	NameVec* pErrVec = &resultFileList[0];
	if (resultFileList[0].size() > resultFileList[1].size() )
		pErrVec = &resultFileList[1];;

	for (NameVec::iterator itr = pErrVec->begin(); itr != pErrVec->end(); itr++)
		cout << *itr << "\t";


	cout << endl << endl
		<< "neg list size = " << resultFileList[0].size()
		<< ", pos list size =  " << resultFileList[1].size()
		<< ", neg ratio = " << resultFileList[0].size() * 1.0f / fileList.size()
		<< ", pos ratio =  " << resultFileList[1].size() * 1.0f / fileList.size()
		<< endl << endl;

	for (int i = 0; i<4; i++)
		cout << "time step " << i << " = " << timeStep[i]*1.0 / fileList.size() << endl;

	return 0;
}