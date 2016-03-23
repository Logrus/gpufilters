#include <cmath>
#include <iostream>
#include <helpers/CMatrix.h>
//#include "timer.h"
#include <nlm_texture/nlmf.h>
#include <omp.h>
#include <helpers/timer_c.h>
#include <helpers/load_sequence.h>
#include <nlm_lkop/lkopgpu.h>
#include <nlm_clustertree/clustertree.h>
#include <nlm_shared/nlmsh.h>
#ifndef __linux__
#include <iso646.h>
#include <helpers/ImageDisplay.h>
#endif
#include <fstream>

// computes the weighted euclidean distance between the patch at (x1,y1) and
// (x2,y2)
inline float patch_distance( const CMatrix<float>& img,
                             int x1, int y1,
                             int x2, int y2,
                             int patch_radius,
                             float* gauss_lut_center )
{
  const int x_size = img.xSize();
  const int y_size = img.ySize();

  float ssd = 0;
  for( int ty = -patch_radius; ty <= patch_radius; ++ty )
  for( int tx = -patch_radius; tx <= patch_radius; ++tx )
  {
    // clamp coordinates
    int p1x = std::min(x_size-1,std::max(0,x1+tx));
    int p1y = std::min(y_size-1,std::max(0,y1+ty));
    int p2x = std::min(x_size-1,std::max(0,x2+tx));
    int p2y = std::min(y_size-1,std::max(0,y2+ty));
    float tmp = img(p1x,p1y)-img(p2x,p2y);
    float gauss_w = *(gauss_lut_center+tx);
    gauss_w *= *(gauss_lut_center+ty);
    ssd += tmp*tmp*gauss_w;
  }

  return ssd;
}



//
// Naive implementation
//
CMatrix<float> denoise_naive(
  const CMatrix<float>& img, int window_radius, int patch_radius, float sqr_sigma )
{
  const float inv_sqr_sigma = 1/sqr_sigma;
  const int x_size = img.xSize();
  const int y_size = img.ySize();
  CMatrix<float> result(x_size,y_size);

  // create a gauss lut for the function patch_distance()
  float* gauss_lut = new float[2*patch_radius+1];
  float* gauss_lut_center = gauss_lut+patch_radius;
  for( int x = -patch_radius; x <= patch_radius; ++x )
    *(gauss_lut_center+x) = std::exp(-0.5*x*x/(patch_radius*patch_radius));

#pragma omp parallel for
  for( int y = 0; y < y_size; ++y )
  for( int x = 0; x < x_size; ++x )
  {
    // window
    const int x1 = std::max(0,x-window_radius);
    const int y1 = std::max(0,y-window_radius);
    const int x2 = std::min(x_size-1,x+window_radius);
    const int y2 = std::min(y_size-1,y+window_radius);

    float sum = 0;
    float new_value = 0;
    for( int ny = y1; ny <= y2; ++ny )
    for( int nx = x1; nx <= x2; ++nx )
    {
      float dsqr = patch_distance(img,x,y,nx,ny,patch_radius,gauss_lut_center);
      float w = std::exp(-dsqr*inv_sqr_sigma);
      new_value += w*img(nx,ny);
      sum += w;
    }
    result(x,y) = new_value/sum;
  }

  delete[] gauss_lut;
  return result;
}



struct Value
{
  float value, sum;
};


//
// Implementation exploiting symmetries
//
CMatrix<float> denoise_symmetries(
  const CMatrix<float>& img, int window_radius, int patch_radius, float sqr_sigma )
{
  const float inv_sqr_sigma = 1/sqr_sigma;
  const int x_size = img.xSize();
  const int y_size = img.ySize();

  // accumulator array for storing the weighted averages of the intensities
  // and the sum of all weights
  CMatrix<Value> acc(x_size,y_size);
  Value zero = {0.f, 0.f};
  acc = zero;


  // create a gauss lut for the function patch_distance()
  float* gauss_lut = new float[2*patch_radius+1];
  float* gauss_lut_center = gauss_lut+patch_radius;
  for( int x = -patch_radius; x <= patch_radius; ++x )
    *(gauss_lut_center+x) = std::exp(-0.5*x*x/(patch_radius*patch_radius));

  // different threads must be 'window_radius+1' rows apart to avoid writing to
  // the same memory
  for( int y_start = 0; y_start <= window_radius; ++y_start)
  {
#pragma omp parallel for
    for( int y = y_start; y < y_size; y+=window_radius+1 )
    for( int x = 0; x < x_size; ++x )
    {
      const int linear_index_reference = y*x_size + x;
      // window
      const int x1 = std::max(0,x-window_radius);
      const int y1 = std::max(0,y); // start in the same row as the current pix
      const int x2 = std::min(x_size-1,x+window_radius);
      const int y2 = std::min(y_size-1,y+window_radius);

      for( int ny = y1; ny <= y2; ++ny )
      for( int nx = x1; nx <= x2; ++nx )
      {
        const int linear_index_neighbour = ny*x_size + nx;
        // skip if the neigbour lies before the reference pixel in memory
        if( linear_index_neighbour < linear_index_reference )
          continue;

        float dsqr = patch_distance(img,x,y,nx,ny,patch_radius,gauss_lut_center);
        float w = std::exp(-dsqr*inv_sqr_sigma);

        acc(x,y).value += w*img(nx,ny);
        acc(x,y).sum += w;
        // avoid weighting the reference patch twice
        if( linear_index_reference != linear_index_neighbour )
        {
          acc(nx,ny).value += w*img(x,y);
          acc(nx,ny).sum += w;
        }
      }
    }
  } // for y_start

  CMatrix<float> result(x_size,y_size);
#pragma omp parallel for
  for( int y = 0; y < y_size; ++y )
  for( int x = 0; x < x_size; ++x )
  {
    result(x,y) = acc(x,y).value/acc(x,y).sum;
  }

  delete[] gauss_lut;
  return result;
}


// computes the mean squared error
float mse( const CMatrix<float>& imgA, const CMatrix<float>& imgB )
{
  float sqr_err = 0;
  for( int i = 0; i < imgA.size(); ++i )
  {
    float diff = imgA.data()[i] - imgB.data()[i];
    sqr_err += diff*diff;
  }
  return sqr_err/imgA.size();
}


// computes the peak signal to noise ratio for 8 bit images
float psnr_8bit( const CMatrix<float>& noise_free, const CMatrix<float>& noisy )
{
  float max_signal = 255;
  float psnr = 10.f*std::log10(max_signal*max_signal/mse(noise_free,noisy));
  return psnr;
}



int main(int argc, char *argv[]){


#ifndef __linux__
  ImageDisplay::ImageDisplay Display;
 #endif
  CVector<CMatrix<float> > seq, ground;
  seq = loadSequence("resources/gstennisg15/t.txt");
  ground = loadSequence("resources/gstennis/t.txt");
  CMatrix<float> noisy_img;

  // read noise free image for comparison
  CMatrix<float> noise_free_img;
  CMatrix<float> denoised_img;
  
  using namespace std;

  CTensor<unsigned short> t(160, 160, 6700);

  //streampos size;
  //char * buffer;
  //ifstream file("resources/cells.raw", ios::in | ios::binary | ios::ate);

  //if (file.is_open())
  //{
	 // size = file.tellg();
	 // buffer = new char[size];
	 // file.seekg(0, ios::beg);
	 // file.read(buffer, size);
	 // file.close();

	 // memcpy(t.data(), buffer, 160*160*6700*sizeof(unsigned short));

	 // delete[] buffer;
  //}

  //
  //CTensor<float> o(160, 160, 6700);
  ////Display.Display("im", t);

  //for (int z = 0; z < 6700; z++){
	 // for (int i = 0; i < t.ySize(); i++)
		//  for (int j = 0; j < t.xSize(); j++){
		//	  unsigned short val = (t(i, j, z) << 8) | (t(i, j, z) >> 8);
		//	  o(i, j, z) = val;
		//	  //std::cout << val << "\t" << t(i, j, 0) << std::endl;
		//  }
	 // Display.Display("im2", o.getMatrix(z));
	 // //cin.get();
  //}

  //o.normalize(0, 255);
  //Display.Display("im2", o);
 // t.writeToPGM("res.pgm");

  float sqr_sigma(16000);
  float sum(0);
  for (int i = 0; i < seq.size(); ++i){

	  noisy_img = seq(i);
	  noise_free_img = ground(i);

	  // compute the PSNR baseline
	  std::cout << "-----------------------------------------------------------------\n";
	  std::cout << "PSNR noisy   : " << psnr_8bit(noise_free_img, noisy_img) << " dB\n";


	  //timer::start("naive");
	  //denoised_img = denoise_naive(noisy_img, 10, 3, sqr_sigma);
	  //timer::stop("naive"); timer::printToScreen(); timer::reset();
	  //std::cout << "PSNR denoised: " << psnr_8bit(noise_free_img, denoised_img) << " dB\n";
	  //
	  //
	  //timer::start("symmetries");
	  //denoised_img = denoise_symmetries(noisy_img, 10, 3, sqr_sigma);
	  //timer::stop("symmetries"); timer::printToScreen(); timer::reset();
	  //std::cout << "PSNR denoised: " << psnr_8bit(noise_free_img, denoised_img) << " dB\n";

	  timer::start("gpu_naive");
	  denoised_img = nonLocalMeanCudaNaive(noisy_img, 10, 3, sqr_sigma);
	  timer::stop("gpu_naive"); timer::printToScreen(); timer::reset();
	  std::cout << "PSNR denoised: " << psnr_8bit(noise_free_img, denoised_img) << " dB\n";

	  timer::start("gpu_textures");
	  denoised_img = nonLocalMeanCudaNaiveTexture(noisy_img, 10, 3, sqr_sigma);
	  timer::stop("gpu_textures"); timer::printToScreen(); timer::reset();
	  std::cout << "PSNR denoised: " << psnr_8bit(noise_free_img, denoised_img) << " dB\n";

	  
	  // Cluster Tree part
	  timer::start("tree_initialization");
	  ClusterTree k(noisy_img, 3, 16000);
	  timer::stop("tree_initialization"); timer::printToScreen(); timer::reset();
	  k.setW(0); // Step to the neighbouring pixel
	  k.setMaximumLevel(8); // Max lvl of traversal
	  //k.setMaximumOfPointsInTheLeafs(100); 
	  timer::start("tree_building");
	  k.buildTree(k.getHead(), 0);
	  timer::stop("tree_building"); timer::printToScreen(); timer::reset();
	  timer::start("tree_nlm");
	  denoised_img = k.nlm();
	  timer::stop("tree_nlm"); timer::printToScreen(); timer::reset();
	  std::cout << "PSNR denoised: " << psnr_8bit(noise_free_img, denoised_img) << " dB\n";

	  // If we want to write output
	  //std::string s;
	  //s = "resources/out/";
	  //std::string zeros = "00";
	  //if (i >= 10) zeros = "0";
	  //if (i >= 100) zeros = "";
	  //std::stringstream ss;
	  //ss << i;
	  //s += "image" + zeros + ss.str() + ".pgm";
	  //denoised_img.writeToPGM(s.c_str());

#ifndef __linux__
	  Display.Display(noisy_img, "noisy");
	  Display.Display(denoised_img, "denoised");
#endif
  }

//  std::cout << (sum / seq.size()) << std::endl;
//// Optical flow object takes the whole sequence and works by itself
//#define OPTICALFLOW
//#ifdef OPTICALFLOW
//  float sqr_sigma_of(2000);
//  NLMLK op(seq, 3, sqr_sigma_of);
//  op.ground_truth = ground;
//  op.computeOF();
//  denoised_img = op.output;
//#endif

  return 0;
}

