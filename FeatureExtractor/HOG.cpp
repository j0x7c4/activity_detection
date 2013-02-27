#include <math.h>
#include <iostream>
//#include "mex.h"
#include <stdlib.h>
#include "HOGFeaturesOfBlock.h"
#include "HOG.h"
#include "constants.h"
#include <assert.h>

// small value, used to avoid division by zero
#define eps 0.0001

// originally obtained from http://people.cs.uchicago.edu/~pff/
// modified by Abhishek Anand (abhishek.anand.iitg@gmail.com)


void HOG::getFeatVec(int blockY, int blockX, HOGFeaturesOfBlock & featsB)
{
  for(int featIndex=0;featIndex<HOGFeaturesOfBlock::numFeats;featIndex++)
    featsB.feats[featIndex]=*(feat + featIndex*numBlocksOutX*numBlocksOutY + blockX*numBlocksOutY+ blockY);
}

//void HOG::getFeatValForPixels(const std::vector<Point2DAbhishek> & interestPointsInImage, HOGFeaturesOfBlock & hogFeats)
//{
//  // bin pixels into blocksOuts()
//  cout<<interestPointsInImage.size ()<<endl;
//  int numPointsInBlock[numBlocksOutY][numBlocksOutX];
//  for(int y=0;y<numBlocksOutY;y++)
//    for(int x=0;x<numBlocksOutX;x++)
//      numPointsInBlock[y][x]=0;
//  
//  Point2DAbhishek outBlock;
//  
//  for(int i=0;i<interestPointsInImage.size();i++)
//    {
//      //     cout<<"in loop"<<endl;
//      pixel2BlockOut (interestPointsInImage[i],outBlock);
//      if(outBlock.x!=-1) // not out of range ... boundary blocks are out of range
//	numPointsInBlock[outBlock.y][outBlock.x]++;
//      //   cout<<"out loop"<<endl;
//    }
//  
//  int max=-1;
//  //find the block(s) where most pixels lie 
//  for(int y=0;y<numBlocksOutY;y++)
//    for(int x=0;x<numBlocksOutX;x++)
//      {
//	if(max<numPointsInBlock[y][x])
//	  max=numPointsInBlock[y][x];
//      }
//  HOGFeaturesOfBlock temp;
//  std::vector<HOGFeaturesOfBlock> maxBlockFeats;
//  cout<<"max= "<<max<<endl;
//  for(int y=0;y<numBlocksOutY;y++)
//    for(int x=0;x<numBlocksOutX;x++)
//      {
//	if(max==numPointsInBlock[y][x])
//	  {
//	    cout <<"out block selected" <<x <<","<<y<<endl;
//	    getFeatVec (y,x,temp);
//	    maxBlockFeats.push_back(temp);
//	  }
//      }
//  HOGFeaturesOfBlock::aggregateFeatsOfBlocks (maxBlockFeats,hogFeats);
//  //push all out blocks with max to a vector and aggregate
//}



  // Assume IMAGE is width x height x 4
void HOG::computeHOG(const Mat& image, int width, int height)
{
  int ndims[3] = {height, width, 3};

  process (image,ndims);
}

size_t HOG::getOffsetInMatlabImage(size_t y, size_t x, size_t channel,size_t dimY, size_t dimX)
{
  assert(channel*dimX*dimY+x*dimY+y<dimX*dimY*3);
  return channel*dimX*dimY+x*dimY+y;
}

// main function:
// takes a double color image and a bin size 
// returns HOG features
void HOG::process(const Mat& image, const int *dims) {
  
  // memory for caching orientation histograms & their norms
  int numBlocksInX;
  int numBlocksInY;
  int blocks[2];
  blocks[0] = (int)((double)dims[0]/(double)sbin+0.5); 
  numBlocksInY=blocks[0];
  blocks[1] = (int)((double)dims[1]/(double)sbin+0.5);
  numBlocksInX=blocks[1];
  
  //double *hist = (double *)calloc(blocks[0]*blocks[1]*18, sizeof(double)); // stores histogram of gradients along each direction in a block
  vector<double> hist(blocks[0]*blocks[1]*18,0);
	//double *norm = (double *)calloc(blocks[0]*blocks[1], sizeof(double)); // stores for the norm for each block
	vector<double> norm(blocks[0]*blocks[1],0);
  // memory for HOG features
  int out[3];
  out[0] = max(blocks[0]-2, 0); // ignore the boundary blocks ?
  numBlocksOutY=out[0];
  
  out[1] = max(blocks[1]-2, 0);
  numBlocksOutX=out[1];
  //cout <<"HOG block dims"<<numBlocksOutX<<","<<numBlocksOutY<<endl;
  out[2] = HOGFeaturesOfBlock::numFeats ;//32 dimensional feature for each block
  //  mxArray *mxfeat = mxCreateNumericArray(3, out, mxDOUBLE_CLASS, mxREAL);
  //  mxArray *mxfeat = mxCreateNumericArray(3, out, mxDOUBLE_CLASS, mxREAL);
  feat = (double *)calloc(out[0]*out[1]*out[2],sizeof(double));
  
  int visible[2];
  visible[0] = blocks[0]*sbin;
  visible[1] = blocks[1]*sbin;
  
	for ( int y = 1;  y < visible[0]-1 ; y++ ) {
		for ( int x = 1 ; x< visible[1]-1 ; x++ ) {
	    //printf("%d %d\n",y,x);
			// first color channel
      double dy = *(image.ptr(y+1) + 3*x) - *(image.ptr(y-1) + 3*x);
      double dx = *(image.ptr(y) + 3*(x+1) ) - *(image.ptr(y) + 3*(x-1));
      double v = dx*dx + dy*dy;

      // second color channel
      double dy2 = *(image.ptr(y+1) + 3*x+1) - *(image.ptr(y-1) + 3*x+1);
      double dx2 = *(image.ptr(y) + 3*(x+1)+1) - *(image.ptr(y) + 3*(x-1)+1);
      double v2 = dx2*dx2 + dy2*dy2;

      // third color channel
      double dy3 = *(image.ptr(y+1) + 3*x+2) - *(image.ptr(y-1) + 3*x+2);
      double dx3 = *(image.ptr(y) + 3*(x+1)+2) - *(image.ptr(y) + 3*(x-1)+2);
      double v3 = dx3*dx3 + dy3*dy3;

      // pick channel with strongest gradient
      if (v2 > v) {
				v = v2;
				dx = dx2;
				dy = dy2;
      } 
      if (v3 > v) {
				v = v3;
				dx = dx3;
				dy = dy3;
      }

      // snap to one of 18 orientations
      double best_dot = 0;
      int best_o = 0;
      for (int o = 0; o < 9; o++) {
				double dot = uu[o]*dx + vv[o]*dy;
				if (dot > best_dot) {
					best_dot = dot;
					best_o = o;
				} else if (-dot > best_dot) {
					best_dot = -dot;
					best_o = o+9;
				}
      }
      
      // add to 4 histograms(blocks) around pixel using linear interpolation
      double xp = ((double)x+0.5)/(double)sbin - 0.5;
      double yp = ((double)y+0.5)/(double)sbin - 0.5;
      int ixp = (int)floor(xp);
      int iyp = (int)floor(yp);
      double vx0 = xp-ixp;
      double vy0 = yp-iyp;
      double vx1 = 1.0-vx0;
      double vy1 = 1.0-vy0;
      v = sqrt(v);

      if (ixp >= 0 && iyp >= 0) {
					hist[ixp*blocks[0] + iyp + best_o*blocks[0]*blocks[1]] += 
					vx1*vy1*v;
      }

      if (ixp+1 < blocks[1] && iyp >= 0) {
					hist[(ixp+1)*blocks[0] + iyp + best_o*blocks[0]*blocks[1]] += 
					vx0*vy1*v;
      }

      if (ixp >= 0 && iyp+1 < blocks[0]) {
				hist[ixp*blocks[0] + (iyp+1) + best_o*blocks[0]*blocks[1]] += 
					vx1*vy0*v;
      }

      if (ixp+1 < blocks[1] && iyp+1 < blocks[0]) {
				hist[(ixp+1)*blocks[0] + (iyp+1) + best_o*blocks[0]*blocks[1]] += 
					vx0*vy0*v;
      }
    }
  }

  // compute energy in each block by summing over orientations
  for (int o = 0; o < 9; o++) {
    double *src1 = &hist[0] + o*blocks[0]*blocks[1];
    double *src2 = &hist[0] + (o+9)*blocks[0]*blocks[1];
    double *dst = &norm[0];
    double *end = &norm[0] + blocks[1]*blocks[0];
    while (dst < end) {
      *(dst++) += (*src1 + *src2) * (*src1 + *src2);
      src1++;
      src2++;
    }
  }

  // compute features
  for (int x = 0; x < out[1]; x++) {
    for (int y = 0; y < out[0]; y++) {
      double *dst = feat + x*out[0] + y;      
      double *src, *p, n1, n2, n3, n4;

      p = &norm[0] + (x+1)*blocks[0] + y+1;
      n1 = 1.0 / sqrt(*p + *(p+1) + *(p+blocks[0]) + *(p+blocks[0]+1) + eps);
      p = &norm[0] + (x+1)*blocks[0] + y;
      n2 = 1.0 / sqrt(*p + *(p+1) + *(p+blocks[0]) + *(p+blocks[0]+1) + eps);
      p = &norm[0] + x*blocks[0] + y+1;
      n3 = 1.0 / sqrt(*p + *(p+1) + *(p+blocks[0]) + *(p+blocks[0]+1) + eps);
      p = &norm[0] + x*blocks[0] + y;      
      n4 = 1.0 / sqrt(*p + *(p+1) + *(p+blocks[0]) + *(p+blocks[0]+1) + eps);

      double t1 = 0;
      double t2 = 0;
      double t3 = 0;
      double t4 = 0;

      // contrast-sensitive features
      src = &hist[0] + (x+1)*blocks[0] + (y+1);
      for (int o = 0; o < 18; o++) {
				double h1 = min(*src * n1, 0.2);
				double h2 = min(*src * n2, 0.2);
				double h3 = min(*src * n3, 0.2);
				double h4 = min(*src * n4, 0.2);
				*dst = 0.5 * (h1 + h2 + h3 + h4);
				t1 += h1;
				t2 += h2;
				t3 += h3;
				t4 += h4;
				dst += out[0]*out[1];
				src += blocks[0]*blocks[1];
      }

      // contrast-insensitive features
      src = &hist[0] + (x+1)*blocks[0] + (y+1);
      for (int o = 0; o < 9; o++) {
        double sum = *src + *(src + 9*blocks[0]*blocks[1]);
        double h1 = min(sum * n1, 0.2);
        double h2 = min(sum * n2, 0.2);
        double h3 = min(sum * n3, 0.2);
        double h4 = min(sum * n4, 0.2);
        *dst = 0.5 * (h1 + h2 + h3 + h4);
        dst += out[0]*out[1];
        src += blocks[0]*blocks[1];
      }

      // texture features
      *dst = 0.2357 * t1;
      dst += out[0]*out[1];
      *dst = 0.2357 * t2;
      dst += out[0]*out[1];
      *dst = 0.2357 * t3;
      dst += out[0]*out[1];
      *dst = 0.2357 * t4;

      // truncation feature
      dst += out[0]*out[1];
      *dst = 0;
    }
  }

  //std::free(hist);
  //std::free(norm);
//  return feat;
}

double const HOG::uu[9] = {1.0000, 
		0.9397, 
		0.7660, 
		0.500, 
		0.1736, 
		-0.1736, 
		-0.5000, 
		-0.7660, 
		-0.9397};
double const HOG::vv[9] = {0.0000, 
		0.3420, 
		0.6428, 
		0.8660, 
		0.9848, 
		0.9848, 
		0.8660, 
		0.6428, 
		0.3420};
