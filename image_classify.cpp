#include <iostream>
#include <fstream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/ml/ml.hpp>
#include "dataset.h" // 定义一些数据
#include <time.h>

using namespace std;
using namespace cv;

typedef std::vector<std::string> NameVec;

//#define TEST_GROUP	"V6" // Tr1, T2, V1, V6


int main(int argc, char const *argv[])
{
	CvSVM svm;//SVM分类器
	svm.load(SVM_FILE);//从XML文件读取训练好的SVM模型
	int DescriptorDim = svm.get_var_count();//特征向量的维数，即HOG描述子的维数

	string ImgName;//图片名
	NameVec fileLists[2];

	string TRAIN_GROUP("3m");
	if (argc >1)
		TRAIN_GROUP = argv[1];

	string TEST_GROUP("T2");
	if (argc >2)
		TEST_GROUP = argv[2];

	string fileListName[2];
	fileListName[0] = string("../dataset/fileNameNeg") + TEST_GROUP + ".txt";
	ifstream ifList(fileListName[0]);//测试样本图片的文件名列表
	while (getline(ifList, ImgName))
		fileLists[0].push_back(ImgName);
	ifList.close();

	fileListName[1] = string("../dataset/fileNamePos") + TEST_GROUP + ".txt";
	ifList.open(fileListName[1]);//测试样本图片的文件名列表
	while (getline(ifList, ImgName))
		fileLists[1].push_back(ImgName);

	if (fileLists[0].empty() || fileLists[1].empty())
	{
		cout << "file list is empty!" << fileListName[0] << "," << fileListName[0] << endl;
		return -1;
	}
	Mat testImg;
	vector<float> descriptor;
	HOGDescriptor hog(Size(HOG_WIDTH, HOG_HEIGHT), Size(24, 24), Size(8, 8), Size(8, 8), 9);//HOG检测器，用来计算HOG描述子的
	Mat testFeatureMat = Mat::zeros(1, DescriptorDim, CV_32FC1);//测试样本的特征向量矩阵

	long tBegAll = clock();
	for (int ii = 0; ii < 2; ii++)
	{
		NameVec& fileList = fileLists[ii];

		NameVec	resultFileList[2]; // 0-neg, 1-pos
		long timeStep[4] = { 0 }; // 0-read, 1-resize, 2-hog, 3-pred

		cout << endl << endl << "begin to predict image files in " << fileListName[ii] << endl;

		for (int num = 0; num < fileList.size(); num++)
		{
			//cout << " 处理： " << fileList[num] << endl;
			//ImgName = "D:\\DataSet\\PersonFromVOC2012\\" + ImgName;//加上正样本的路径名
			string pathImg = string("../dataset/neg") + TEST_GROUP + "/";
			if (ii == 1)
				pathImg = string("../dataset/pos") + TEST_GROUP + "/";

			ImgName = pathImg + fileList[num];//加上正样本的路径名

			/******************读入单个测试图并对其HOG描述子进行分类*********************/
			////读取测试图片，并计算其HOG描述子
			long tBegFrameAll = clock();
			long tBeg = clock();
			testImg = imread(ImgName, COLOR_GRAY);
			timeStep[0] += clock() - tBeg;

			if (testImg.empty())
			{
				cout << "image not found - " << ImgName << endl;
				continue;
			}
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

			//cout << "1 frame ms time = " << clock() - tBegFrameAll << endl;

			//cout << "分类结果：" << result << endl;
			cout << ".";

			string pathImgOut;
			if (1 == result)
			{
				resultFileList[1].push_back(fileList[num]);
				pathImgOut = pathImg + TRAIN_GROUP + "/pos/";

			}
			else
			{
				resultFileList[0].push_back(fileList[num]);
				pathImgOut = pathImg + TRAIN_GROUP + "/neg/";
			}
			imwrite(pathImgOut + fileList[num], testImg);
		}

		cout << endl << "succeed to predict " << fileList.size() << " frames, total ms time = " << clock() - tBegAll << endl;

		// predict error list to output 
		cout << " error list: " << endl;

		NameVec* pErrVec = &resultFileList[0];
		if (resultFileList[0].size() > resultFileList[1].size())
			pErrVec = &resultFileList[1];;

		for (NameVec::iterator itr = pErrVec->begin(); itr != pErrVec->end(); itr++)
			cout << *itr << endl;


		cout << endl << endl
			<< "neg list size = " << resultFileList[0].size()
			<< ", pos list size =  " << resultFileList[1].size()
			<< ", neg ratio = " << resultFileList[0].size() * 1.0f / fileList.size()
			<< ", pos ratio =  " << resultFileList[1].size() * 1.0f / fileList.size()
			<< endl << endl;

		for (int i = 0; i < 4; i++)
			cout << "time step " << i << " = " << timeStep[i] * 1.0 / fileList.size() << endl;
	}
	return 0;
}