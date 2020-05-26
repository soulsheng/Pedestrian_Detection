#include <iostream>
#include <fstream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/ml/ml.hpp>
#include "dataset.h" // ����һЩ����
#include <time.h>

using namespace std;
using namespace cv;

typedef std::vector<std::string> NameVec;

//#define TEST_GROUP	"V6" // Tr1, T2, V1, V6


int main(int argc, char const *argv[])
{
	CvSVM svm;//SVM������
	svm.load(SVM_FILE);//��XML�ļ���ȡѵ���õ�SVMģ��
	int DescriptorDim = svm.get_var_count();//����������ά������HOG�����ӵ�ά��

	string ImgName;//ͼƬ��
	NameVec fileLists[2];

	string TRAIN_GROUP("3m");
	if (argc >1)
		TRAIN_GROUP = argv[1];

	string TEST_GROUP("T2");
	if (argc >2)
		TEST_GROUP = argv[2];

	string fileListName[2];
	fileListName[0] = string("../dataset/fileNameNeg") + TEST_GROUP + ".txt";
	ifstream ifList(fileListName[0]);//��������ͼƬ���ļ����б�
	while (getline(ifList, ImgName))
		fileLists[0].push_back(ImgName);
	ifList.close();

	fileListName[1] = string("../dataset/fileNamePos") + TEST_GROUP + ".txt";
	ifList.open(fileListName[1]);//��������ͼƬ���ļ����б�
	while (getline(ifList, ImgName))
		fileLists[1].push_back(ImgName);

	if (fileLists[0].empty() || fileLists[1].empty())
	{
		cout << "file list is empty!" << fileListName[0] << "," << fileListName[0] << endl;
		return -1;
	}
	Mat testImg;
	vector<float> descriptor;
	HOGDescriptor hog(Size(HOG_WIDTH, HOG_HEIGHT), Size(24, 24), Size(8, 8), Size(8, 8), 9);//HOG���������������HOG�����ӵ�
	Mat testFeatureMat = Mat::zeros(1, DescriptorDim, CV_32FC1);//����������������������

	long tBegAll = clock();
	for (int ii = 0; ii < 2; ii++)
	{
		NameVec& fileList = fileLists[ii];

		NameVec	resultFileList[2]; // 0-neg, 1-pos
		long timeStep[4] = { 0 }; // 0-read, 1-resize, 2-hog, 3-pred

		cout << endl << endl << "begin to predict image files in " << fileListName[ii] << endl;

		for (int num = 0; num < fileList.size(); num++)
		{
			//cout << " ���� " << fileList[num] << endl;
			//ImgName = "D:\\DataSet\\PersonFromVOC2012\\" + ImgName;//������������·����
			string pathImg = string("../dataset/neg") + TEST_GROUP + "/";
			if (ii == 1)
				pathImg = string("../dataset/pos") + TEST_GROUP + "/";

			ImgName = pathImg + fileList[num];//������������·����

			/******************���뵥������ͼ������HOG�����ӽ��з���*********************/
			////��ȡ����ͼƬ����������HOG������
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
			hog.compute(testImg, descriptor, Size(8, 8));//����HOG�����ӣ���ⴰ���ƶ�����(8,8)
			timeStep[2] += clock() - tBeg;

			tBeg = clock();
			//������õ�HOG�����Ӹ��Ƶ�testFeatureMat������
			for (int i = 0; i < descriptor.size(); i++)
				testFeatureMat.at<float>(0, i) = descriptor[i];

			//��ѵ���õ�SVM�������Բ���ͼƬ�������������з���
			int result = svm.predict(testFeatureMat);//�������
			timeStep[3] += clock() - tBeg;

			//cout << "1 frame ms time = " << clock() - tBegFrameAll << endl;

			//cout << "��������" << result << endl;
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