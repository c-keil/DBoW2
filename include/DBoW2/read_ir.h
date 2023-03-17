#ifndef READ_NPY_IR_
#define READ_NPY_IR_

#include <string>
// #include <vector>
#include <opencv2/core/types.hpp>

void readDescNPY(std::string const &data_fname, cv::Mat &out);
void readDescNPY(std::string const &data_fname, std::vector<std::vector<float>> &out);
// void readKpNPY(std::string const &data_fname, std::vector<cv::KeyPoint> &keypoints);
#endif