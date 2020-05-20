#include <iostream>
#include <fstream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/ml/ml.hpp>
#include "dataset.h" // ����һЩ����
#include "my_svm.h" // MySVM�̳���CvSVM����
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
	MySVM svm;//SVM������
	svm.load(SVM_FILE);//��XML�ļ���ȡѵ���õ�SVMģ��
	int DescriptorDim = svm.get_var_count();//����������ά������HOG�����ӵ�ά��

	string ImgName;//ͼƬ��
	ifstream ifList(TEST_FILE_LIST);//��������ͼƬ���ļ����б�

	std::vector<std::string> fileList;
	while (getline(ifList, ImgName))
		fileList.push_back(ImgName);

	Mat testImg;
	vector<float> descriptor;
	HOGDescriptor hog(Size(HOG_WIDTH, HOG_HEIGHT), Size(24, 24), Size(8, 8), Size(8, 8), 9);//HOG���������������HOG�����ӵ�
	Mat testFeatureMat = Mat::zeros(1, DescriptorDim, CV_32FC1);//����������������������

	NameVec	resultFileList[2]; // 0-neg, 1-pos
	long timeStep[4] = { 0 }; // 0-read, 1-resize, 2-hog, 3-pred

	long tBegAll = clock();
	for (int num = 0; num < fileList.size(); num++)
	{
		//cout << " ���� " << fileList[num] << endl;
		//ImgName = "D:\\DataSet\\PersonFromVOC2012\\" + ImgName;//������������·����
		ImgName = TEST_FILE_PATH + fileList[num];//������������·����

		/******************���뵥������ͼ������HOG�����ӽ��з���*********************/
		////��ȡ����ͼƬ����������HOG������
		long tBegFrameAll = clock();
		long tBeg = clock();
		testImg = imread(ImgName);
		timeStep[0] += clock() - tBeg;

		tBeg = clock();
		resize(testImg, testImg, Size(HOG_WIDTH, HOG_HEIGHT));
		timeStep[1] += clock() - tBeg;

		tBeg = clock();
		hog.compute(testImg, descriptor, Size(8, 8));//����HOG�����ӣ���ⴰ���ƶ�����(8,8)
		timeStep[2] += clock() - tBeg;
		
		tBeg = clock();
		//������õ�HOG�����Ӹ��Ƶ�testFeatureMat������
		for (int i = 0; i < descriptor.size(); i++)
			testFeatureMat.at<float>(0, i) = descriptor[i];

		//��ѵ���õ�SVM�������Բ���ͼƬ�������������з���
		int result = svm.predict(testFeatureMat);//�������
		timeStep[3] += clock() - tBeg;

		cout << "1 frame ms time = " << clock() - tBegFrameAll << endl;

		//cout << "��������" << result << endl;
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