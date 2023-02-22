/**
 * File: FIR.cpp
 * Date: November 2011
 * Author: Dorian Galvez-Lopez
 * Description: functions for Surf64 descriptors
 * License: see the LICENSE.txt file
 *
 */

#include <vector>
#include <string>
#include <sstream>

#include "FClass.h"
#include "FIR.h"

using namespace std;

namespace DBoW2
{

  // --------------------------------------------------------------------------

  void FIR::meanValue(const std::vector<FIR::pDescriptor> &descriptors,
                          FIR::TDescriptor &mean)
  {
    mean.zeros(1, FIR::L, CV_32F);

    float s = descriptors.size();

    vector<FIR::pDescriptor>::const_iterator it;
    for (it = descriptors.begin(); it != descriptors.end(); ++it)
    {
      const FIR::TDescriptor &desc = **it;
      mean += desc/s;
    }
  }

  // --------------------------------------------------------------------------

  double FIR::distance(const FIR::TDescriptor &a, const FIR::TDescriptor &b)
  {
    double sqd = cv::norm(a, b, cv::NORM_L2);
    return sqd;
  }

  // --------------------------------------------------------------------------

  std::string FIR::toString(const FIR::TDescriptor &a)
  {
    stringstream ss;
    for (int i = 0; i < FIR::L; ++i)
    {
      ss << to_string(a.at<float>(0,i)) << " ";
    }
    return ss.str();
  }

  // --------------------------------------------------------------------------

  void FIR::fromString(FIR::TDescriptor &a, const std::string &s)
  {
    a.create(1, FIR::L, CV_32F);

    stringstream ss(s);
    for (int i = 0; i < FIR::L; ++i)
    {
      ss >> a.at<float>(0,i);
    }
  }

  // --------------------------------------------------------------------------

  void FIR::toMat32F(const std::vector<TDescriptor> &descriptors,
                         cv::Mat &mat)
  {
    if (descriptors.empty())
    {
      mat.release();
      return;
    }

    const int N = descriptors.size();
    const int L = FIR::L;

    mat.create(N, L, CV_32F);

    for (int i = 0; i < N; ++i)
    {
      const TDescriptor &desc = descriptors[i];
      desc.copyTo(mat.row(i));
    }
  }

  // --------------------------------------------------------------------------

} // namespace DBoW2