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

void loadFeatures(const vector<string> &fnames, vector<vector<vector<float>>> &features);
void changeStructure(const cv::Mat &plain, vector<vector<float>> &out);
void testVocCreation(const vector<vector<vector<float> > > &features);
void testDatabase(const vector<vector<vector<float> > > &features);
void getDescFileNames(const string strPathsFile, vector<string> &vstrDesc);
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

    if(argc != 5)
    {
        cerr << endl << "Usage: build_voc <path_file> <output_file> <k=10> <L=10>" << endl;
        return 1;
    }
    string desc_index_file = string(argv[1]);
    string save_path = string(argv[2]);
    vector<string> fileNames;
    int k = std::stoi(argv[3]);
    int L = std::stoi(argv[4]);
    getDescFileNames(desc_index_file, fileNames);

    vector<vector<vector<float>>> features;
    loadFeatures(fileNames, features);

    createVocabulary(features, save_path, k, L);

    return 0;
}

void loadFeatures(const vector<string>& fnames, vector<vector<vector<float>>> &features)
{
    uint lim = 500;
    for (string fname : fnames)
    {   
        cout << "processing file " << fname << endl;
        vector<vector<float>> desc;
        readDescNPY(fname, desc);
        // if (desc.size() > lim)
        // {
        //     desc.resize(lim);
        // }
        features.push_back(desc);
    }
}

// void changeStructure(const cv::Mat &plain, vector<cv::Mat> &out)
// {
//   out.resize(plain.rows);

//   for(int i = 0; i < plain.rows; ++i)
//   {
//     out[i] = plain.row(i);
//   }
// }

void getDescFileNames(const string strPathsFile, vector<string> &vstrDescFiles)
{
    cout << "Reading file names from: '" << strPathsFile << "'" << endl;
    ifstream fTimes;
    fTimes.open(strPathsFile.c_str());
    vector<string> vTimeStamps; 
    vTimeStamps.reserve(5000);
    string desc_file_type = ".npy";
    while(!fTimes.eof())
    {
        string s;
        getline(fTimes,s);
        if(!s.empty())
        {
            stringstream ss;
            ss << s;
            vstrDescFiles.push_back(ss.str());
        }
    }
}

void createVocabulary(const vector<vector<vector<float>>> &features, const string vocabName, const int k, const int L)
{   
    cout << "creating vocabulary" << endl;
    IRVocabulary2 voc(k, L, weight, scoring);
    voc.create(features);
    cout << voc << endl;
    cout << endl << "Saving vocabulary..." << endl;
    voc.save(vocabName + ".yml.gz");

    // cout << "testing vocabulary " << endl;
    // BowVector v1, v2;
    // for (uint i = 0; i < features.size(); i++)
    // {
    //     voc.transform(features[i], v1);
    //     voc.transform(features[i], v2);
    //     double score = voc.score(v1, v2);
    //     if (score < 0.0001)
    //     {
    //         // cout << "vector: " << v1 << " vector 2: " << v2 << endl;
    //         cout << "Image " << i << " vs Image " << i << ": " << score << endl;
    //     }

    //     // for (uint j = 0; j < features.size(); j++)
    //     // {
    //     //     voc.transform(features[j], v2);

    //     //     double score = voc.score(v1, v2);
    //     //     if (score > 0.0)
    //     //     {
    //     //         // cout << "SOCRE" << endl;
    //     //         cout << "Image " << i << " vs Image " << j << ": " << score << endl;
    //     //     }
    //     // }
    // }

    // cout << endl << "Saving vocabulary..." << endl;
    // voc.save(vocabName + ".yml.gz");
    // // cout << "Done" << endl;
}