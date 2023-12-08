#include <iostream>
#include <vector>
#include <string>
#include <boost/timer.hpp>

// DBoW2
#include "DBoW2.h" // defines IRFeatures
#include "read_ir.h"

// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>


using namespace DBoW2;
using namespace std;

void loadFeaturesSP(const vector<string> &fnames, vector<vector<vector<float>>> &features);
void loadFeaturesORB(const vector<string> &fnames, vector<vector<cv::Mat >> &features);
void time_experiment(vector<vector<vector<float>>> &features_sp, vector<vector<cv::Mat >> &features_orb);
void changeStructure(const cv::Mat &plain, vector<cv::Mat> &out);
void testVocCreation(const vector<vector<vector<float> > > &features);
void testDatabase(const vector<vector<vector<float> > > &features);
void getDescFileNames(const string strPathsFile, vector<string> &vstrDesc);
void createVocabulary(const vector<vector<vector<float>>> &features, const string vocabName, const int k, const int L);

const WeightingType weight = TF_IDF;
const ScoringType scoring = L2_NORM;

int main(int argc, char **argv)
{

    if(argc != 3)
    {
        cerr << endl << "Usage: build_voc <desc_path_file> <orb_path_image>" << endl;
        return 1;
    }
    string desc_index_file = string(argv[1]);
    string orb_images = string(argv[2]);
    vector<string> fileNamesSP, fileNamesORB;
    // int k = std::stoi(argv[3]);
    // int L = std::stoi(argv[4]);
    getDescFileNames(desc_index_file, fileNamesSP);
    getDescFileNames(orb_images, fileNamesORB);

    vector<vector<vector<float>>> features_sp;
    vector<vector<cv::Mat>> features_orb;
    loadFeaturesSP(fileNamesSP, features_sp);
    loadFeaturesORB(fileNamesORB, features_orb);

    // createVocabulary(features, save_path, k, L);
    time_experiment(features_sp, features_orb);

    return 0;
}

void time_experiment(vector<vector<vector<float>>> &features_sp, vector<vector<cv::Mat >> &features_orb)
{
  auto desc1 = features_sp[0][0]; 
  auto desc2 = features_sp[0][1];
  int n_samples = 10000000; 
  double dist;
  cout << "Timing " << n_samples << " sp distance calculations..." << endl;
  boost::timer myTimer;
  for (int i=0; i<n_samples; i++)
  {
    dist = FIR2::distance(desc1,desc2);
  }
  auto duration = myTimer.elapsed();
  std::cout << duration << " seconds elapsed." << endl;
  cout << duration/n_samples << " seconds per distance calculation" << endl;

  auto orb_desc1 = features_orb[0][0];
  auto orb_desc2 = features_orb[0][1];
  int orb_dist;
  cout << "Timing " << n_samples << " orb distance calculations..." << endl;
  boost::timer myTimer2;
  for (int i=0; i<n_samples; i++)
  {
    orb_dist = FORB::distance(orb_desc1,orb_desc2);
  }
  auto orb_duration = myTimer2.elapsed();
  std::cout << orb_duration << " seconds elapsed." << endl;
  cout << orb_duration/n_samples << " seconds per distance calculation" << endl;
  
  cout << endl << "orb is ~" << duration/orb_duration << " times faster than sp" << endl;

}

void loadFeaturesSP(const vector<string>& fnames, vector<vector<vector<float>>> &features)
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

void loadFeaturesORB(const vector<string>& fnames, vector<vector<cv::Mat >> &features)
{ 
  features.clear();
  features.reserve(fnames.size());

  cv::Ptr<cv::ORB> orb = cv::ORB::create();

  for(int fi=0; fi < fnames.size(); ++fi)
  { 
    string fname = fnames[fi];
    cout << "processing file " << fname << endl;
    cv::Mat image = cv::imread(fname, 0);
    cv::Mat mask;
    vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;

    orb->detectAndCompute(image, mask, keypoints, descriptors);

    features.push_back(vector<cv::Mat >());
    changeStructure(descriptors, features.back());
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
    // string desc_file_type = ".npy";
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