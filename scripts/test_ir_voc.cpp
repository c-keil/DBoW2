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

void loadFeatures(const vector<string> &fnames, vector<vector<cv::Mat>> &features);
void changeStructure(const cv::Mat &plain, vector<cv::Mat> &out);
void testVocCreation(const vector<vector<cv::Mat > > &features);
void testDatabase(const vector<vector<cv::Mat > > &features);
void getDescFileNames(const string strPathsFile, vector<string> &vstrDesc);
void testVocabulary(const vector<vector<cv::Mat>> &features, const string vocab_file);

// const string desc_index_file = "";
// k: 10
// L: 6
// scoringType: 1
// weightingType: 0
// branching factor and depth levels
const int k = 10;
const int L = 6;
// const WeightingType weight = IDF;
const WeightingType weight = TF_IDF;
// const ScoringType scoring = L1_NORM;
const ScoringType scoring = L2_NORM;

int main(int argc, char **argv)
{

    if(argc != 3)
    {
        cerr << endl << "Usage: build_voc <path_file> <output_file>" << endl;
        return 1;
    }
    string desc_index_file = string(argv[1]);
    string voc_file = string(argv[2]);
    vector<string> fileNames;
    getDescFileNames(desc_index_file, fileNames);

    vector<vector<cv::Mat>> features;
    loadFeatures(fileNames, features);

    testVocabulary(features, voc_file);
    // createVocabulary(features, save_path);

    return 0;
}

void loadFeatures(const vector<string>& fnames, vector<vector<cv::Mat>> &features)
{
    for (string fname : fnames)
    {   
        cout << "processing file " << fname << endl;
        cv::Mat desc;
        features.push_back(vector<cv::Mat>());
        readDescNPY(fname, desc);
        // cv::Mat desc_reduced = desc(cv::Range(0, 200), cv::Range::all()); //TODO remove this
        changeStructure(desc, features.back());
    }
}

void changeStructure(const cv::Mat &plain, vector<cv::Mat> &out)
{
  out.resize(plain.rows);

  for(int i = 0; i < plain.rows; ++i)
  {
    out[i] = plain.row(i);
  }
}

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

void testVocabulary(const vector<vector<cv::Mat>> &features, const string voc_file)
{   
    cout << "reading vocabulary" << endl;
    IRVocabulary voc(voc_file);
    cout << voc << endl;
    // voc.create(features);

    // cout << "testing vocabulary " << endl;
    // BowVector v1, v2;
    // for (uint i = 0; i < features.size(); i++)
    // {
    //     voc.transform(features[i], v1);
    //     voc.transform(features[i], v2);
    //     cout << "vector: " << v1 << endl << v2 << endl;
    //     double score = voc.score(v1, v2);
    //     cout << "Image " << i << " vs Image " << i << ": " << score << endl;

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
    // cout << "Done" << endl;
}