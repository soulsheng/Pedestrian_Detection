#ifndef DATASET_H
#define DATASET_H


#define CENTRAL_CROP false   //true:训练时，对正样本图片剪裁

#define HardExampleListFile "HardExample_FromINRIA_NegList.txt"
//HardExample：负样本个数。如果HardExampleNO大于0，表示处理完初始负样本集后，继续处理HardExample负样本集。
//不使用HardExample时必须设置为0，因为特征向量矩阵和特征类别矩阵的维数初始化时用到这个值
#define HardExampleNO 0

#define TermCriteriaCount 10000  //迭代终止条件，当迭代满50000次或误差小于FLT_EPSILON时停止迭代
#define TermCriteriaEps		1e-3

#define SVM_FILE	"../xml/SVM_HOG6b.xml"
#define	HOG_WIDTH	96
#define HOG_HEIGHT	96

#define COLOR_GRAY		0	// 1-COLOR, 0-GRAY
#define AUTO_TRAIN		1

#endif
