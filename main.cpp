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
	string TRAIN_GROUP("1");
	if (argc >1)
		TRAIN_GROUP = argv[1];

	string SVM_FILE = string("../xml/SVM_HOG") + TRAIN_GROUP + ".xml";
  //检测窗口(64,128),块尺寸(16,16),块步长(8,8),cell尺寸(8,8),直方图bin个数9
	HOGDescriptor hog(Size(HOG_WIDTH, HOG_HEIGHT), Size(24, 24), Size(8, 8), Size(8, 8), 9);//HOG检测器，用来计算HOG描述子的
	int DescriptorDim;//HOG描述子的维数，由图片大小、检测窗口大小、块大小、细胞单元中直方图bin个数决定
	MySVM svm;//SVM分类器

	string fileListName[3];
	fileListName[0] = string("../dataset/fileNameNegTr") + TRAIN_GROUP + ".txt";
	fileListName[1] = string("../dataset/fileNamePosTr") + TRAIN_GROUP + ".txt";

	string ImgName;//图片名(绝对路径)
	ifstream finPos(fileListName[1]);//正样本图片的文件名列表
	//ifstream finPos("PersonFromVOC2012List.txt");//正样本图片的文件名列表
	ifstream finNeg(fileListName[0]);//负样本图片的文件名列表
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

	Mat src(HOG_WIDTH, HOG_HEIGHT, CV_8UC3);
	vector<float> descriptors;//HOG描述子向量
	hog.compute(src, descriptors, Size(8, 8));//计算HOG描述子，检测窗口移动步长(8,8)
	DescriptorDim = descriptors.size();//HOG描述子的维数
	cout << " 描述子维数： " << DescriptorDim << endl; // 8100 for img size(96,96)

	//初始化所有训练样本的特征向量组成的矩阵，行数等于所有样本的个数，列数等于HOG描述子维数sampleFeatureMat
	sampleFeatureMat = Mat::zeros(nCountSamples, DescriptorDim, CV_32FC1);
	//初始化训练样本的类别向量，行数等于所有样本的个数，列数等于1；1表示有人，0表示无人
	sampleLabelMat = Mat::zeros(nCountSamples, 1, CV_32FC1);

	long tBeg = clock();
	int nOffset = 0;
	for (int i = 0; i < 3; i++)
	{
		string pathImg = string("../dataset/negTr") + TRAIN_GROUP + "/";

		float valLabel = -1;
		if (i == 1)
		{
			pathImg = string("../dataset/posTr") + TRAIN_GROUP + "/";;
			valLabel = 1;
		}

		cout << "path image is: " << pathImg << endl;

		//依次读取正/负样本图片，生成HOG描述子
		for (int num = 0; num<fileList[i].size(); num++)
		{
			ImgName = fileList[i][num];
			// cout<<" 处理： "<<ImgName<<endl;

			ImgName = pathImg + ImgName;//加上正样本的路径名
			Mat src = imread(ImgName, COLOR_GRAY);//读取图片，转为灰度图

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
			for (int j = 0; j<DescriptorDim; j++)
				sampleFeatureMat.at<float>(num + nOffset, j) = descriptors[j];//第num个样本的特征向量中的第j个元素
			
			sampleLabelMat.at<float>(num+nOffset,0) = valLabel;//正样本类别为1，负样本类别为-1

			cout << ".";

		}
		nOffset += fileList[i].size();
		cout << fileListName[i] << " file count = " << fileList[i].size() << endl;
	}

		cout << "hog ms time = " << clock() - tBeg << endl;

		tBeg = clock();

		//训练SVM分类器
		//迭代终止条件，当迭代满1000次或误差小于FLT_EPSILON时停止迭代
		CvTermCriteria criteria = cvTermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, TermCriteriaCount, TermCriteriaEps);
		//SVM参数：SVM类型为C_SVC；线性核函数；松弛因子C=0.01
		CvSVMParams param(CvSVM::C_SVC, CvSVM::LINEAR, 0, 1, 0, 0.01, 0, 0, 0, criteria);
		cout<<" 开始训练SVM分类器 "<< SVM_FILE << endl;

#if AUTO_TRAIN
		svm.train_auto(sampleFeatureMat, sampleLabelMat, Mat(), Mat(), param,
			5,
			CvSVM::get_default_grid(CvSVM::C),
			CvSVM::get_default_grid(CvSVM::GAMMA),
			CvSVM::get_default_grid(CvSVM::P),
			CvSVM::get_default_grid(CvSVM::NU),
			CvSVM::get_default_grid(CvSVM::COEF),
			CvSVM::get_default_grid(CvSVM::DEGREE),
			true);
#else
		svm.train(sampleFeatureMat,sampleLabelMat, Mat(), Mat(), param);/* 训练分类器 */
#endif

		cout << "train ms time = " << clock() - tBeg << endl;

		cout<<" 训练完成 "<< endl;
		svm.save(SVM_FILE.c_str());//将训练好的SVM模型保存为xml文件

		CvSVMParams params_re = svm.get_params();
		float C = params_re.C;
		float P = params_re.p;
		float gamma = params_re.gamma;
		printf("\nParms: C = %f, P = %f,gamma = %f \n", C, P, gamma);

  return 0;
}
