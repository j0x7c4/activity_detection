#ifndef HOG_H
#define HOG_H
#include <opencv2\opencv.hpp>

#include "HOGFeaturesOfBlock.h"
#include "Point2DAbhishek.h"

#define min(x,y) ((x)<=(y)?(x):(y))
#define max(x,y) ((x)<=(y)?(y):(x))

using namespace cv;

class HOG
{
static double const uu[9];
static double const vv[9];
// unit vectors used to compute gradient orientation
  double *feat;
  int numBlocksOutX;
  int numBlocksOutY;
  


public :
  HOG()
  {
    feat=NULL;
  }
  ~HOG()
  {
    free( feat);
  }

  void getFeatVec(int blockY, int blockX, HOGFeaturesOfBlock & featsB);
  //void getFeatValForPixels(const std::vector<Point2DAbhishek> & interestPointsInImage, HOGFeaturesOfBlock & hogFeats);
  void pixel2BlockOut(const Point2DAbhishek & p,Point2DAbhishek  & b );
    
  static  int const sbin=8;

  void computeHOG(const Mat& image, int width, int height);
  size_t getOffsetInMatlabImage(size_t y, size_t x, size_t channel,size_t dimY, size_t dimX);
  void process(const Mat& image, const int *dims);


int 
getNumFeatsPerBlock () const
{
  return HOGFeaturesOfBlock::numFeats;
}

int
getNumBlocksY () const
{
  return numBlocksOutY;
}

int
getNumBlocksX () const
{
  return numBlocksOutX;
}

};


#endif
