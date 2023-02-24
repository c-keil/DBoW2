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
void createVocabulary(const vector<vector<cv::Mat>> &features, const string vocabName);
void testVoc(vector<vector<cv::Mat>> &features1, vector<vector<cv::Mat>> &features2, const string voc_file);
// void testVoc(vector<vector<cv::Mat>> &features, const string voc_file);
// const string desc_index_file = "";
// k: 10
// L: 6
// scoringType: 1
// weightingType: 0
// branching factor and depth levels
const int k = 10;
const int L = 6;
const WeightingType weight = TF_IDF;
const ScoringType scoring = L2_NORM;

int main(int argc, char **argv)
{

    if(argc != 4)
    {
        cerr << endl
             << "Usage: test_ir_voc <paths_file1> <paths_file2> <vocab_file>" << endl;
        return 1;
    }
    string desc_index_file1 = string(argv[1]);
    string desc_index_file2 = string(argv[2]);
    string voc_file = string(argv[3]);
    vector<string> fileNames1, fileNames2;
    getDescFileNames(desc_index_file1, fileNames1);
    getDescFileNames(desc_index_file2, fileNames2);

    vector<vector<cv::Mat>> features1, features2;
    loadFeatures(fileNames1, features1);
    loadFeatures(fileNames2, features2);

    //test vacabulary
    testVoc(features1, features2, voc_file);

    return 0;
}

void loadFeatures(const vector<string>& fnames, vector<vector<cv::Mat>> &features)
{
    for (string fname : fnames)
    {   
        // cout << "processing file " << fname << endl;
        cv::Mat desc;
        features.push_back(vector<cv::Mat>());
        readDescNPY(fname, desc);
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

void createVocabulary(const vector<vector<cv::Mat>> &features, const string vocabName)
{   
    cout << "creating vocabulary" << endl;
    IRVocabulary voc(k, L, weight, scoring);
    voc.create(features);
    cout << endl << "Saving vocabulary..." << endl;
    voc.save(vocabName + ".yml.gz");
    cout << "Done" << endl;
}

void testVoc(vector<vector<cv::Mat>> &features1, vector<vector<cv::Mat>> &features2, const string voc_file)
{

    // load vocabulary
    cout << "reading in voc file : " << voc_file << endl;
    IRVocabulary voc(voc_file);
    IRDatabase db(voc, false, 0);
    //add features to db
    for (uint i=0; i<features1.size(); i++)
    {
        db.add(features1[i]);
    }
    // cout << "Database information: " << endl << db << endl;
    cout << "Testing feature set 2 against feature set 1" << endl;
    QueryResults res;
    for (uint i = 0; i < features2.size(); i++)
    {
        db.query(features2[i], res, 2);
        cout << "Searching for Image " << i << ". " << res << endl;
    }

    // BowVector v1, v2;
    // for (uint i = 0; i < features.size(); i++)
    // {
    //     voc.transform(features[i], v1);
    //     // voc.transform(features[i], v2);
    //     // double score = voc.score(v1, v2);
    //     // cout << "Image " << i << " vs Image " << i << ": " << score << endl;
    //     for (uint j = 0; j < features.size(); j++)
    //     {
    //         voc.transform(features[j], v2);
    //         double score = voc.score(v1, v2);
    //         if (score>0.8)
    //         {
    //             // cout << "SOCRE" << endl;
    //             cout << "Image " << i << " vs Image " << j << ": " << score << endl;
    //         }           
    //     }
    // }


}