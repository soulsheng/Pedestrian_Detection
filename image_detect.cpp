#include <iostream>
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include "my_svm.h"
#include "common.h"

using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
  Mat img;
  FILE* f = 0;
  char _filename[1024];

  if( argc == 1 )
  {
      printf("Usage: %s (<image_filename> | <image_list>.txt)\n",argv[0]);
      return 0;
  }

  img = imread(argv[1]);

  if( img.data )
  {
      strcpy(_filename, argv[1]);
  }
  else
  {
      f = fopen(argv[1], "rt");
      if(!f)
      {
          fprintf( stderr, "ERROR: the specified file could not be loaded\n");
          return -1;
      }
  }

  //检测窗口(64,128),块尺寸(16,16),块步长(8,8),cell尺寸(8,8),直方图bin个数9
	//HOGDescriptor hog(Size(64,128),Size(16,16),Size(8,8),Size(8,8),9);//HOG检测器，用来计算HOG描述子的
	int DescriptorDim;//HOG描述子的维数，由图片大小、检测窗口大小、块大小、细胞单元中直方图bin个数决定
	MySVM svm;//SVM分类器
  svm.load("SVM_HOG.xml");


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

  namedWindow("people detector", 1);

  for(;;)
  {
      char* filename = _filename;
      if(f)
      {
          if(!fgets(filename, (int)sizeof(_filename)-2, f))
              break;
          //while(*filename && isspace(*filename))
          //  ++filename;
          if(filename[0] == '#')
              continue;
          int l = (int)strlen(filename);
          while(l > 0 && isspace(filename[l-1]))
              --l;
          filename[l] = '\0';
          img = imread(filename);
      }
      printf("%s:\n", filename);
      if(!img.data)
          continue;

      fflush(stdout);
      vector<Rect> found, found_filtered;
      double t = (double)getTickCount();
      // run the detector with default parameters. to get a higher hit-rate
      // (and more false alarms, respectively), decrease the hitThreshold and
      // groupThreshold (set groupThreshold to 0 to turn off the grouping completely).
      myHOG.detectMultiScale(img, found, 0, Size(8,8), Size(32,32), 1.05, 2);
      t = (double)getTickCount() - t;
      printf("tdetection time = %gms\n", t*1000./cv::getTickFrequency());
      size_t i, j;
      for( i = 0; i < found.size(); i++ )
      {
          Rect r = found[i];
          for( j = 0; j < found.size(); j++ )
              if( j != i && (r & found[j]) == r)
                  break;
          if( j == found.size() )
              found_filtered.push_back(r);
      }
      for( i = 0; i < found_filtered.size(); i++ )
      {
          Rect r = found_filtered[i];
          // the HOG detector returns slightly larger rectangles than the real objects.
          // so we slightly shrink the rectangles to get a nicer output.
          r.x += cvRound(r.width*0.1);
          r.width = cvRound(r.width*0.8);
          r.y += cvRound(r.height*0.07);
          r.height = cvRound(r.height*0.8);
          rectangle(img, r.tl(), r.br(), cv::Scalar(0,255,0), 3);
      }
      imshow("people detector", img);
      int c = waitKey(0) & 255;
      if( c == 'q' || c == 'Q' || !f)
          break;
  }
  if(f)
      fclose(f);
  return 0;
}
