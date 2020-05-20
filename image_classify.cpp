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

using namespace std;
using namespace cv;

#define	TEST_POS	1
#if TEST_POS
#define TEST_FILE_LIST	"../dataset/fileNamePosTest.txt"
#define TEST_FILE_PATH	"../dataset/posTestAIZOO/"
#else
#define TEST_FILE_LIST	"../dataset/fileNameNegTest.txt"
#define TEST_FILE_PATH	"../dataset/NegTestAIZOO/"
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
	for (int num = 0; num < fileList.size(); num++)
	{
		//cout << " ���� " << fileList[num] << endl;
		//ImgName = "D:\\DataSet\\PersonFromVOC2012\\" + ImgName;//������������·����
		ImgName = TEST_FILE_PATH + fileList[num];//������������·����

		/******************���뵥������ͼ������HOG�����ӽ��з���*********************/
		////��ȡ����ͼƬ����������HOG������
		testImg = imread(ImgName);
		resize(testImg, testImg, Size(HOG_WIDTH, HOG_HEIGHT));

		hog.compute(testImg, descriptor, Size(8, 8));//����HOG�����ӣ���ⴰ���ƶ�����(8,8)
		//������õ�HOG�����Ӹ��Ƶ�testFeatureMat������
		for (int i = 0; i < descriptor.size(); i++)
			testFeatureMat.at<float>(0, i) = descriptor[i];

		//��ѵ���õ�SVM�������Բ���ͼƬ�������������з���
		int result = svm.predict(testFeatureMat);//�������
		//cout << "��������" << result << endl;
		cout << ".";

		if (1 == result)
			resultFileList[1].push_back(fileList[num]);
		else
			resultFileList[0].push_back(fileList[num]);

	}

	for (int i = 0; i < 2; i++)
	{
		cout << endl << endl << "file list " << i << " (0-neg,1-pos), length = " << resultFileList[i].size() << endl << endl;

		for (NameVec::iterator itr = resultFileList[i].begin(); itr != resultFileList[i].end(); itr++)
			cout << *itr << "\t";

	}

	cout << endl << endl
		<< "neg list size = " << resultFileList[0].size()
		<< ", pos list size =  " << resultFileList[1].size()
		<< ", neg ratio = " << resultFileList[0].size() * 1.0f / fileList.size()
		<< ", pos ratio =  " << resultFileList[1].size() * 1.0f / fileList.size()
		<< endl << endl;

	return 0;
}