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

// void loadFeatures(const vector<string> &fnames, vector<vector<cv::Mat>> &features);
// void changeStructure(const cv::Mat &plain, vector<cv::Mat> &out);
// void testVocCreation(const vector<vector<cv::Mat > > &features);
// void testDatabase(const vector<vector<cv::Mat > > &features);
// void getDescFileNames(const string strPathsFile, vector<string> &vstrDesc);
// void createVocabulary(const vector<vector<cv::Mat>> &features, const string vocabName);
// void testVoc(vector<vector<cv::Mat>> &features1, vector<vector<cv::Mat>> &features2, 
// const string voc_file, const string out_file, const string db_file, const string test_file);
void loadFeatures(const vector<string> &fnames, vector<vector<vector<float>>> &features);
void testVocCreation(const vector<vector<vector<float>>> &features);
void testDatabase(const vector<vector<vector<float>>> &features);
void getDescFileNames(const string strPathsFile, vector<string> &vstrDesc);
void createVocabulary(const vector<vector<vector<float>>> &features, const string vocabName);
void testVoc(vector<vector<vector<float>>> &features1, vector<vector<vector<float>>> &features2,
             const string voc_file, const string out_file, const string db_file, const string test_file);

// const int k = 10;
// const int L = 6;
// const WeightingType weight = TF_IDF;
// const ScoringType scoring = L2_NORM;

int main(int argc, char **argv)
{

    if(argc != 5)
    {
        cerr << endl
             << "Usage: test_ir_voc <paths_file1> <paths_file2> <vocab_file> <output_file>" << endl;
        return 1;
    }
    string desc_index_file1 = string(argv[1]);
    string desc_index_file2 = string(argv[2]);
    string voc_file = string(argv[3]);
    string out_file = string(argv[4]);
    vector<string> fileNames1, fileNames2;
    getDescFileNames(desc_index_file1, fileNames1);
    getDescFileNames(desc_index_file2, fileNames2);

    vector<vector<vector<float>>> features1, features2;
    loadFeatures(fileNames1, features1);
    loadFeatures(fileNames2, features2);

    //test vacabulary
    testVoc(features1, features2, voc_file, out_file, desc_index_file1, desc_index_file2);

    //write results to file
    //visualize results
    //
    return 0;
}

void loadFeatures(const vector<string> &fnames, vector<vector<vector<float>>> &features)
{
    uint lim = 300;
    for (string fname : fnames)
    {
        // cout << "processing file " << fname << endl;
        vector<vector<float>> desc;
        readDescNPY(fname, desc);
        if (desc.size() > lim)
        {
            desc.resize(lim);
        }
        features.push_back(desc);
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

// void createVocabulary(const vector<vector<vector<float>>> &features, const string vocabName)
// {   
//     cout << "creating vocabulary" << endl;
//     IRVocabulary2 voc(k, L, weight, scoring);
//     voc.create(features);
//     cout << endl << "Saving vocabulary..." << endl;
//     voc.save(vocabName + ".yml.gz");
//     cout << "Done" << endl;
// }

void testVoc(vector<vector<vector<float>>> &features1, vector<vector<vector<float>>> &features2, 
const string voc_file, const string out_file, const string db_file, const string test_file)
{

    // load vocabulary
    cout << "reading in voc file : " << voc_file << endl;
    IRVocabulary2 voc(voc_file);
    cout << "voc : " << voc << endl;
    IRDatabase2 db(voc, false, 0);
    //add features to db
    BowVector v1;
    for (uint i = 0; i < features1.size(); i++)
    {
        // cout << "loading features " << i << endl;
        // voc.transform(features1[i], v1);
        // cout << "Vector : " << v1 << endl;
        db.add(features1[i]);
    }
    // cout << "Database information: " << endl << db << endl;
    cout << "Testing feature set 2 against feature set 1" << endl;

    ofstream file;
    file.open(out_file);
    file << db_file << "\n" << test_file <<  "\n"; 

    QueryResults res;
    for (uint i = 0; i < features2.size(); i++)
    {
        // voc.transform(features2[i], v1);
        // cout << "Vector : " << v1 << endl;
        db.query(features2[i], res, 2);
        cout << "Searching for Image " << i << ". " << res << endl;
        file << "Test Im: " << i << "\n";
        file << res << "\n";

    }
    file.close();

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