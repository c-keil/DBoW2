#include <string>
#include <opencv2/opencv.hpp>

#include "read_ir.h"
#include "cnpy.h"

void readDescNPY(std::string const &data_fname, cv::Mat &out)
{
    // Load the data from file
    cnpy::NpyArray npy_data = cnpy::npy_load(data_fname);

    // Get pointer to data
    float *ptr = npy_data.data<float>();
    int num_rows = npy_data.shape[0];
    int num_cols = npy_data.shape[1];
    out = cv::Mat(num_rows, num_cols, CV_32F, ptr).clone();
}

// void readKpNPY(std::string const &data_fname, std::vector<cv::KeyPoint> &keypoints)
// {
//     cnpy::NpyArray npy_data = cnpy::npy_load(data_fname);
//     keypoints.clear();
//     std::vector<float> data;
//     float *ptr = npy_data.data<float>();

//     for (std::size_t i = 0; i < npy_data.shape[0]; i++)
//     {
//         std::size_t i2 = i * 2;
//         keypoints.push_back(cv::KeyPoint(ptr[i2], ptr[i2 + 1], 1.0));
//     }
// }