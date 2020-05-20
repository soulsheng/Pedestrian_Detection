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

using namespace std;
using namespace cv;

int main(int argc, char const *argv[])
{
  //检测窗口(64,128),块尺寸(16,16),块步长(8,8),cell尺寸(8,8),直方图bin个数9
	HOGDescriptor hog(Size(HOG_WIDTH, HOG_HEIGHT), Size(24, 24), Size(8, 8), Size(8, 8), 9);//HOG检测器，用来计算HOG描述子的
	int DescriptorDim;//HOG描述子的维数，由图片大小、检测窗口大小、块大小、细胞单元中直方图bin个数决定
	MySVM svm;//SVM分类器

  //若TRAIN为true，重新训练分类器
	if(TRAIN)
	{
		string ImgName;//图片名(绝对路径)
		ifstream finPos(PosSamListFile);//正样本图片的文件名列表
		//ifstream finPos("PersonFromVOC2012List.txt");//正样本图片的文件名列表
		ifstream finNeg(NegSamListFile);//负样本图片的文件名列表

		Mat sampleFeatureMat;//所有训练样本的特征向量组成的矩阵，行数等于所有样本的个数，列数等于HOG描述子维数
		Mat sampleLabelMat;//训练样本的类别向量，行数等于所有样本的个数，列数等于1；1表示有人，-1表示无人


		//依次读取正样本图片，生成HOG描述子
		for(int num=0; num<PosSamNO && getline(finPos,ImgName); num++)
		{
			cout<<" 处理： "<<ImgName<<endl;
			//ImgName = "D:\\DataSet\\PersonFromVOC2012\\" + ImgName;//加上正样本的路径名
			ImgName = "../dataset/pos/" + ImgName;//加上正样本的路径名
			Mat src = imread(ImgName);//读取图片
			if(CENTRAL_CROP)
			if(src.cols >= 96 && src.rows >= 160)
				    src = src(Rect(16,16,64,128));//将96*160的INRIA正样本图片剪裁为64*128，即剪去上下左右各16个像素
			
			resize(src, src, Size(HOG_WIDTH, HOG_HEIGHT));

			vector<float> descriptors;//HOG描述子向量
			hog.compute(src,descriptors,Size(8,8));//计算HOG描述子，检测窗口移动步长(8,8)
			//cout<<"描述子维数："<<descriptors.size()<<endl;

			//处理第一个样本时初始化特征向量矩阵和类别矩阵，因为只有知道了特征向量的维数才能初始化特征向量矩阵
			if( 0 == num )
			{
				DescriptorDim = descriptors.size();//HOG描述子的维数
				//初始化所有训练样本的特征向量组成的矩阵，行数等于所有样本的个数，列数等于HOG描述子维数sampleFeatureMat
				sampleFeatureMat = Mat::zeros(PosSamNO+NegSamNO+HardExampleNO, DescriptorDim, CV_32FC1);
				//初始化训练样本的类别向量，行数等于所有样本的个数，列数等于1；1表示有人，0表示无人
				sampleLabelMat = Mat::zeros(PosSamNO+NegSamNO+HardExampleNO, 1, CV_32FC1);
			}

			//将计算好的HOG描述子复制到样本特征矩阵sampleFeatureMat
			for(int i=0; i<DescriptorDim; i++)
				sampleFeatureMat.at<float>(num,i) = descriptors[i];//第num个样本的特征向量中的第i个元素
			sampleLabelMat.at<float>(num,0) = 1;//正样本类别为1，有人
		}

		//依次读取负样本图片，生成HOG描述子
		for(int num=0; num<NegSamNO && getline(finNeg,ImgName); num++)
		{
			cout<<" 处理： "<<ImgName<<endl;
			//ImgName = "E:\\运动目标检测\\INRIAPerson\\negphoto\\" + ImgName;//加上负样本的路径名
			ImgName = "../dataset/neg/" + ImgName;//加上负样本的路径名
			Mat src = imread(ImgName);//读取图片
			resize(src, src, Size(HOG_WIDTH, HOG_HEIGHT));

			vector<float> descriptors;//HOG描述子向量
			hog.compute(src,descriptors,Size(8,8));//计算HOG描述子，检测窗口移动步长(8,8)
			//cout<<"描述子维数："<<descriptors.size()<<endl;

			//将计算好的HOG描述子复制到样本特征矩阵sampleFeatureMat
			for(int i=0; i<DescriptorDim; i++)
				sampleFeatureMat.at<float>(num+PosSamNO,i) = descriptors[i];//第PosSamNO+num个样本的特征向量中的第i个元素
			sampleLabelMat.at<float>(num+PosSamNO,0) = -1;//负样本类别为-1，无人

		}

		//处理HardExample负样本
		if(HardExampleNO > 0)
		{
			ifstream finHardExample(HardExampleListFile);//HardExample负样本的文件名列表
			//依次读取HardExample负样本图片，生成HOG描述子
			for(int num=0; num<HardExampleNO && getline(finHardExample,ImgName); num++)
			{
				cout<<" 处理： "<<ImgName<<endl;
				//ImgName = "D:\\DataSet\\HardExample_2400PosINRIA_12000Neg\\" + ImgName;//加上HardExample负样本的路径名
        ImgName = "dataset/HardExample/" + ImgName;//加上HardExample负样本的路径名
				Mat src = imread(ImgName);//读取图片
				//resize(src,src,Size(64,128));

				vector<float> descriptors;//HOG描述子向量
				hog.compute(src,descriptors,Size(8,8));//计算HOG描述子，检测窗口移动步长(8,8)
				//cout<<"描述子维数："<<descriptors.size()<<endl;

				//将计算好的HOG描述子复制到样本特征矩阵sampleFeatureMat
				for(int i=0; i<DescriptorDim; i++)
					sampleFeatureMat.at<float>(num+PosSamNO+NegSamNO,i) = descriptors[i];//第PosSamNO+num个样本的特征向量中的第i个元素
				sampleLabelMat.at<float>(num+PosSamNO+NegSamNO,0) = -1;//负样本类别为-1，无人
			}
		}

		//输出样本的HOG特征向量矩阵到文件
	/*	ofstream fout("SampleFeatureMat.txt");
		for(int i=0; i<PosSamNO+NegSamNO; i++)
		{
			fout<<i<<endl;
			for(int j=0; j<DescriptorDim; j++)
			{	fout<<sampleFeatureMat.at<float>(i,j)<<"  ";

			}
			fout<<endl;
		}*/

		//训练SVM分类器
		//迭代终止条件，当迭代满1000次或误差小于FLT_EPSILON时停止迭代
		CvTermCriteria criteria = cvTermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, TermCriteriaCount, FLT_EPSILON);
		//SVM参数：SVM类型为C_SVC；线性核函数；松弛因子C=0.01
		CvSVMParams param(CvSVM::C_SVC, CvSVM::LINEAR, 0, 1, 0, 0.01, 0, 0, 0, criteria);
		cout<<" 开始训练SVM分类器 "<<endl;
		svm.train(sampleFeatureMat,sampleLabelMat, Mat(), Mat(), param);/* 训练分类器 */
		cout<<" 训练完成 "<<endl;
		svm.save(SVM_FILE);//将训练好的SVM模型保存为xml文件

	}
	else //若TRAIN为false，从XML文件读取训练好的分类器
	{
		svm.load(SVM_FILE);//从XML文件读取训练好的SVM模型
	}

	//从svm xml中获取HOG检测子参数
	vector<float> myDetector;
	parseSvmXML(svm, myDetector);

	//设置HOGDescriptor的检测子
	HOGDescriptor myHOG(Size(HOG_WIDTH, HOG_HEIGHT), Size(24, 24), Size(8, 8), Size(8, 8), 9);
	myHOG.setSVMDetector(myDetector);
	//myHOG.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());

	//保存检测子参数到文件
	ofstream fout("HOGDetectorForOpenCV.txt");
	for(int i=0; i<myDetector.size(); i++)
	{
		fout<<myDetector[i]<<endl;
	}

  /**************读入图片进行HOG行人检测******************/
	Mat src = imread(TestImageFileName);
	vector<Rect> found, found_filtered;//矩形框数组
	cout<<" 进行多尺度HOG人体检测 "<<endl;
	myHOG.detectMultiScale(src, found, 0, Size(8,8), Size(32,32), 1.05, 2);//对图片进行多尺度行人检测
	 //src为输入待检测的图片；found为检测到目标区域列表；参数3为程序内部计算为行人目标的阈值，也就是检测到的特征到SVM分类超平面的距离;
   //参数4为滑动窗口每次移动的距离。它必须是块移动的整数倍；参数5为图像扩充的大小；参数6为比例系数，即测试图片每次尺寸缩放增加的比例；
    //参数7为组阈值，即校正系数，当一个目标被多个窗口检测出来时，该参数此时就起了调节作用，为0时表示不起调节作用。

	//找出所有没有嵌套的矩形框r,并放入found_filtered中,如果有嵌套的话,则取外面最大的那个矩形框放入found_filtered中
	for(int i=0; i < found.size(); i++)
	{
		Rect r = found[i];
		int j=0;
		for(; j < found.size(); j++)
			if(j != i && (r & found[j]) == r)
				break;
		if( j == found.size())
			found_filtered.push_back(r);
	}
  cout<<" 找到的矩形框个数： "<<found_filtered.size()<<endl;

	//画矩形框，因为hog检测出的矩形框比实际人体框要稍微大些,所以这里需要做一些调整
	for(int i=0; i<found_filtered.size(); i++)
	{
		Rect r = found_filtered[i];
		r.x += cvRound(r.width*0.1);
		r.width = cvRound(r.width*0.8);
		r.y += cvRound(r.height*0.07);
		r.height = cvRound(r.height*0.8);
		rectangle(src, r.tl(), r.br(), Scalar(0,255,0), 3);
	}

	imwrite("ImgProcessed.jpg",src);
	namedWindow("src",0);
	imshow("src",src);
	waitKey();//注意：imshow之后必须加waitKey，否则无法显示图像


	/******************读入单个64*128的测试图并对其HOG描述子进行分类*********************/
	////读取测试图片(64*128大小)，并计算其HOG描述子
	//Mat testImg = imread("person014142.jpg");
	//Mat testImg = imread("noperson000026.jpg");
	//vector<float> descriptor;
	//hog.compute(testImg,descriptor,Size(8,8));//计算HOG描述子，检测窗口移动步长(8,8)
	//Mat testFeatureMat = Mat::zeros(1,3780,CV_32FC1);//测试样本的特征向量矩阵
	//将计算好的HOG描述子复制到testFeatureMat矩阵中
	//for(int i=0; i<descriptor.size(); i++)
	//	testFeatureMat.at<float>(0,i) = descriptor[i];

	//用训练好的SVM分类器对测试图片的特征向量进行分类
	//int result = svm.predict(testFeatureMat);//返回类标
	//cout<<"分类结果："<<result<<endl;

  return 0;
}
