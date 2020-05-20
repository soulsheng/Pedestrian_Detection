#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <stdio.h>
#include "my_svm.h"
#include "common.h"

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
  VideoCapture capture;
  if( argc == 1 )
  {
    capture.open("video.avi");
    if(!capture.isOpened()){
      printf("Usage: %s (<image_filename> | <video_filename>)\n",argv[0]);
      return 0;
    }
  } else {
    capture.open(argv[1]);
    if(!capture.isOpened()){
      printf("Usage: %s <video_filename>\n",argv[0]);
      return 0;
    }
  }

  //检测窗口(64,128),块尺寸(16,16),块步长(8,8),cell尺寸(8,8),直方图bin个数9
  //HOGDescriptor hog(Size(64,128),Size(16,16),Size(8,8),Size(8,8),9);//HOG检测器，用来计算HOG描述子的
  int DescriptorDim;//HOG描述子的维数，由图片大小、检测窗口大小、块大小、细胞单元中直方图bin个数决定
  MySVM svm;//SVM分类器
  svm.load("SVM_HOG.xml");

  
	//从svm xml中获取HOG检测子参数
	vector<float> myDetector;
	parseSvmXML(svm, myDetector);

  //设置HOGDescriptor的检测子
  HOGDescriptor myHOG;
  myHOG.setSVMDetector(myDetector);
  //myHOG.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());

  //	//保存检测子参数到文件
  //	ofstream fout("HOGDetectorForOpenCV.txt");
  //	for(int i=0; i<myDetector.size(); i++)
  //	{
  //		fout<<myDetector[i]<<endl;
  //	}

  //VideoCapture capture(argv[1]);
  //if(!capture.isOpened())
  //  return 1;
  double rate=capture.get(CV_CAP_PROP_FPS);
  bool stop(false);
  Mat frame;

  namedWindow("Video");
  int delay = 1000/rate;

  while(!stop)
  {
    if(!capture.read(frame))
      break;
    Mat src=frame;

    vector<Rect> found, found_filtered;//矩形框数组
    myHOG.detectMultiScale(src, found, 0, Size(8,8), Size(32,32), 1.05, 2);//对图片进行多尺度行人检测

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

    imshow("Video",src);

    if(waitKey(delay)>=0)
      stop=true;
  }
  capture.release();
}
