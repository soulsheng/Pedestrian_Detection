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

#define TRAIN_FILE_LIST_NEG	"../dataset/fileNameNegTr1.txt"
#define TRAIN_FILE_PATH_NEG	"../dataset/negTr1/"
#define TRAIN_FILE_LIST_POS	"../dataset/fileNamePosTr1.txt"
#define TRAIN_FILE_PATH_POS	"../dataset/posTr1/"

using namespace std;
using namespace cv;

int main(int argc, char const *argv[])
{
  //检测窗口(64,128),块尺寸(16,16),块步长(8,8),cell尺寸(8,8),直方图bin个数9
	HOGDescriptor hog(Size(HOG_WIDTH, HOG_HEIGHT), Size(24, 24), Size(8, 8), Size(8, 8), 9);//HOG检测器，用来计算HOG描述子的
	int DescriptorDim;//HOG描述子的维数，由图片大小、检测窗口大小、块大小、细胞单元中直方图bin个数决定
	MySVM svm;//SVM分类器

	string ImgName;//图片名(绝对路径)
	ifstream finPos(TRAIN_FILE_LIST_POS);//正样本图片的文件名列表
	//ifstream finPos("PersonFromVOC2012List.txt");//正样本图片的文件名列表
	ifstream finNeg(TRAIN_FILE_LIST_NEG);//负样本图片的文件名列表
	ifstream finHardExample(HardExampleListFile);//HardExample负样本的文件名列表


	NameVec fileList[3];
	while (getline(finNeg, ImgName))
		fileList[0].push_back(ImgName);

	while (getline(finPos, ImgName))
		fileList[1].push_back(ImgName);

	while (getline(finHardExample, ImgName))
		fileList[2].push_back(ImgName);

	if (fileList[0].size() == 0 || fileList[1].size() == 0)
	{
		cout << "sample num is 0" << endl;
		return -1;
	}

	int nCountSamples = fileList[0].size() + fileList[1].size() + fileList[2].size() ;

	Mat sampleFeatureMat;//所有训练样本的特征向量组成的矩阵，行数等于所有样本的个数，列数等于HOG描述子维数
	Mat sampleLabelMat;//训练样本的类别向量，行数等于所有样本的个数，列数等于1；1表示有人，-1表示无人

	Mat src = imread(string(TRAIN_FILE_PATH_NEG) + fileList[0][0]);//读取图片
	if (src.empty())
	{
		cout << "sample image is empty" << endl;
		return -1;
	}

	resize(src, src, Size(HOG_WIDTH, HOG_HEIGHT));
	vector<float> descriptors;//HOG描述子向量
	hog.compute(src, descriptors, Size(8, 8));//计算HOG描述子，检测窗口移动步长(8,8)

	DescriptorDim = descriptors.size();//HOG描述子的维数
	//初始化所有训练样本的特征向量组成的矩阵，行数等于所有样本的个数，列数等于HOG描述子维数sampleFeatureMat
	sampleFeatureMat = Mat::zeros(nCountSamples, DescriptorDim, CV_32FC1);
	//初始化训练样本的类别向量，行数等于所有样本的个数，列数等于1；1表示有人，0表示无人
	sampleLabelMat = Mat::zeros(nCountSamples, 1, CV_32FC1);

	long tBeg = clock();
	int nOffset = 0;
	for (int i = 0; i < 3; i++)
	{
		
		//依次读取正/负样本图片，生成HOG描述子
		for (int num = 0; num<fileList[i].size(); num++)
		{
			ImgName = fileList[i][num];
			// cout<<" 处理： "<<ImgName<<endl;
			string pathImg = TRAIN_FILE_PATH_NEG;

			float valLabel = -1;
			if (i == 1)
			{
				pathImg = TRAIN_FILE_PATH_POS;
				valLabel = 1;
			}

			ImgName = pathImg + ImgName;//加上正样本的路径名
			Mat src = imread(ImgName);//读取图片

			if (src.empty())
			{
				cout << "sample image is empty, " << ImgName << endl;
				continue;
			}

			if(CENTRAL_CROP)
			if(src.cols >= 96 && src.rows >= 160)
				    src = src(Rect(16,16,64,128));//将96*160的INRIA正样本图片剪裁为64*128，即剪去上下左右各16个像素
			
			resize(src, src, Size(HOG_WIDTH, HOG_HEIGHT));

			hog.compute(src,descriptors,Size(8,8));//计算HOG描述子，检测窗口移动步长(8,8)
			//cout<<"描述子维数："<<descriptors.size()<<endl;

			//将计算好的HOG描述子复制到样本特征矩阵sampleFeatureMat
			for(int i=0; i<DescriptorDim; i++)
				sampleFeatureMat.at<float>(num+nOffset,i) = descriptors[i];//第num个样本的特征向量中的第i个元素
			
			sampleLabelMat.at<float>(num+nOffset,0) = valLabel;//正样本类别为1，负样本类别为-1

			cout << ".";

		}
		nOffset += fileList[i].size();
	}

		cout << "hog ms time = " << clock() - tBeg << endl;

		tBeg = clock();

		//训练SVM分类器
		//迭代终止条件，当迭代满1000次或误差小于FLT_EPSILON时停止迭代
		CvTermCriteria criteria = cvTermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, TermCriteriaCount, FLT_EPSILON);
		//SVM参数：SVM类型为C_SVC；线性核函数；松弛因子C=0.01
		CvSVMParams param(CvSVM::C_SVC, CvSVM::LINEAR, 0, 1, 0, 0.01, 0, 0, 0, criteria);
		cout<<" 开始训练SVM分类器 "<<endl;
		svm.train(sampleFeatureMat,sampleLabelMat, Mat(), Mat(), param);/* 训练分类器 */

		cout << "train ms time = " << clock() - tBeg << endl;

		cout<<" 训练完成 "<<endl;
		svm.save(SVM_FILE);//将训练好的SVM模型保存为xml文件

  return 0;
}
