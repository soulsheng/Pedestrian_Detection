
#ifndef __COMMON_H__
#define __COMMON_H__
#include <vector>
#include <iostream>
#include <opencv2/core/core.hpp>

#include "my_svm.h" // MySVM¼Ì³Ð×ÔCvSVMµÄÀà

using namespace cv;
using namespace std;

int parseSvmXML(MySVM& svm, std::vector<float>& myDetector);


#endif //__COMMON_H__s