#include <iostream>
#include <vector>
#include <string>

// DBoW2
#include "DBoW2.h" // defines IRFeatures
#include "read_ir.h"

// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>


using namespace DBoW2;
using namespace std;

void createVocabulary(const vector<vector<vector<float>>> &features, const string vocabName, const int k, const int L);
// void loadFeatures(const vector<string> &fnames, vector<vector<cv::Mat>> &features);
// void changeStructure(const cv::Mat &plain, vector<cv::Mat> &out);
// void testVocCreation(const vector<vector<cv::Mat>> &features);
// void testDatabase(const vector<vector<cv::Mat>> &features);
// void getDescFileNames(const string strPathsFile, vector<string> &vstrDesc);
// void createVocabulary(const vector<vector<cv::Mat>> &features, const string vocabName);

// const int k = 10;
// const int L = 6;
const WeightingType weight = TF_IDF;
const ScoringType scoring = L2_NORM;

int main(int argc, char **argv)
{
    string file_name = "/home/colin/Research/ir/DBoW2/vocs/crunch_vocs/gluestick_all_matched_descriptors_downsamp2_L5_k10.yml.gz";
    string save_path = "/home/colin/Research/ir/DBoW2/vocs/crunch_vocs/gluestick_all_matched_descriptors_downsamp2_L5_k10.txt";
    
    cout << "reading voc" << endl;
    IRVocabulary2 voc(file_name);
    
    cout << endl << "Saving vocabulary..." << endl;
    voc.saveToTextFile(save_path);


    return 0;
}

