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

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
void loadFeatures(vector<vector<vector<float> > > &features);
// void changeStructure(const cv::Mat &plain, vector<cv::Mat> &out);
void testVocCreation(const vector<vector<vector<float>>> &features);
void testDatabase(const vector<vector<vector<float>>> &features);

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

// number of training images
const int NIMAGES = 4;
const string dir = "/media/colin/box_data/ir_data/nuance_data/cater_isec_day_night/carter_isec_alley_night/cam_3/matlab_clahe2/gluestick_skip2/matched_descriptors/";
const string desc1 = "1689389085960999966.npy";
const string desc2 = "1689389091493999958.npy";
const string desc3 = "1689389097026999950.npy";
const string desc4 = "1689389102561000109.npy";


// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

void wait()
{
  cout << endl << "Press enter to continue" << endl;
  getchar();
}

// ----------------------------------------------------------------------------

int main()
{
  vector<vector<vector<float> > > features;
  loadFeatures(features);

  testVocCreation(features);

  // wait();

  // testDatabase(features);

  return 0;
}

// ----------------------------------------------------------------------------

void loadFeatures(vector<vector<vector<float>>> &features)
{
  features.clear();
  features.reserve(NIMAGES);

  // cv::Ptr<cv::ORB> orb = cv::ORB::create();

  cout << "Extracting IR features..." << endl;
  cv::Mat desc_m;
  vector<vector<float>> desc;

  // readDescNPY(dir + desc1, desc_m);
  readDescNPY(dir + desc1, desc);
  features.push_back(desc);
  readDescNPY(dir + desc2, desc);
  features.push_back(desc);
  readDescNPY(dir + desc3, desc);
  features.push_back(desc);
  readDescNPY(dir + desc4, desc);
  features.push_back(desc);

  // cout << desc.size() << endl;
  // cout << desc[0].size() << endl;
  // for (uint i = 0; i < 256; i++)
  // {
  //   cout << desc[0][i] ;
  // }


  // changeStructure(desc, features.back());
  // cout << features.back().size() << endl;
  // cout << features[0][0] << endl;

  // features.push_back(vector<vector<float>>());
  // readDescNPY(dir + desc2, desc);
  // // changeStructure(desc, features.back());
  // // cout << features.back().size() << endl;
  // // cout << features[0][0] - features[1][0] << endl;

  // features.push_back(vector<vector<float>>());
  // readDescNPY(dir + desc3, desc);
  // // changeStructure(desc, features.back());
  // // cout << features.back().size() << endl;
  // // cout << desc.row(0) << endl;

  // features.push_back(vector<vector<float>>());
  // readDescNPY(dir + desc4, desc);
  // // changeStructure(desc, features.back());
  // // cout << features.back().size() << endl;
  // // cout << desc.row(0) << endl;

}

// ----------------------------------------------------------------------------

void changeStructure(const cv::Mat &plain, vector<cv::Mat> &out)
{
  out.resize(plain.rows);

  for(int i = 0; i < plain.rows; ++i)
  {
    out[i] = plain.row(i);
  }
}

// ----------------------------------------------------------------------------

void testVocCreation(const vector<vector<vector<float>>> &features)
{
  // branching factor and depth levels 
  const int k = 9;
  const int L = 3;
  const WeightingType weight = TF_IDF;
  const ScoringType scoring = L2_NORM;

  string voc_name = "/media/colin/box_data/ir_data/nuance_data/kri_day_2/cam_3/matlab_clahe2/gluestick/gluestick_day_night_voc_L5_k10.yml.gz";
  IRVocabulary2 voc(voc_name);

  // cout << "Creating a small " << k << "^" << L << " vocabulary..." << endl;
  // voc.create(features);
  // cout << "... done!" << endl;

  // cout << "Vocabulary information: " << endl
  // << voc << endl << endl;

  // lets do something with this vocabulary
  cout << "Matching images against themselves (0 low, 1 high): " << endl;
  BowVector v1, v2;
  FeatureVector featureVec; 

  voc.transform(features[0], v1);
  voc.transform(features[1], v2);
  cout << endl << "words1: " << v1 << endl; 
  cout << endl << "words2: " << v2 << endl; 
  double score = voc.score(v1, v2);
  cout << "Image 0 vs Image 1 : " << score << endl;
  cout << v1.size() << endl;
  cout << v2.size() << endl;

  // for (int i = 0; i < v1.size(); i++)
  // {
  //   for (int j = 0; j < v2.size(); j++)
  //   { 
  //     // cout << i << ", " << j << endl;
  //     cout << v1[i] << ", " << v2[j] << endl;
  //     // if (v1[i] == v2[j])
  //     // {
  //     //   cout << "Word in common : " << v1[i] << v2[j] << endl;
  //     // }  
  //   }
  // }
  uint64 word_count = 0;
  double score2 = 0;
  for (auto i : v1)
  { 
    // cout << i.first << ", " << i.second << endl;
    for (auto j : v2)
    { 
      // cout << i << ", " << j << endl;
      // cout << v1[i] << ", " << v2[j] << endl;
      if (i.first == j.first)
      { 
        word_count++;
        score2 += i.second;
        cout << "Word in common : " << i.first << ", " << i.second << endl;
      }  
    }
  }
  cout << "score : " << score2/v1.size() << endl;
  cout << "score : " << score2/v2.size() << endl;
  cout << "score : " << score2/word_count << endl;

  cout << "Found " << word_count << " words in common out of " << v1.size() << " and " << v2.size() << " words" << endl; 


  // for(int i = 0; i < NIMAGES; i++)
  // {
  //   voc.transform(features[i], v1);
  //   cout << "feature vec: " << featureVec << endl;
  //   cout << endl << "words: " << v1 << endl; 
  //   for(int j = 0; j < NIMAGES; j++)
  //   {
  //     voc.transform(features[j], v2, featureVec, 1);
  //     double score = voc.score(v1, v2);
  //     // cout << "Vec1:" << v1 << " Vec2:" << v2 << endl;
  //     cout << "Image " << i << " vs Image " << j << ": " << score << endl;
  //   }
  // }

  // // save the vocabulary to disk
  // cout << endl << "Saving vocabulary..." << endl;
  // voc.save("small_voc2.yml.gz");
  // cout << "Done" << endl;

  // voc.load("small_voc2.yml.gz");
  // // lets do something with this vocabulary
  // cout << "Matching images against themselves (0 low, 1 high): " << endl;
  // // BowVector v1, v2;
  // for (int i = 0; i < NIMAGES; i++)
  // {
  //   voc.transform(features[i], v1);
  //   for (int j = 0; j < NIMAGES; j++)
  //   {
  //     voc.transform(features[j], v2);

  //     double score = voc.score(v1, v2);
  //     cout << "Vec1:" << v1 << " Vec2:" << v2 << endl;
  //     cout << "Image " << i << " vs Image " << j << ": " << score << endl;
  //   }
  // }
}

// ----------------------------------------------------------------------------

void testDatabase(const vector<vector<vector<float>>> &features)
{
  cout << "Creating a small database..." << endl;

  // load the vocabulary from disk
  string voc_name = "/media/colin/box_data/ir_data/nuance_data/kri_day_2/cam_3/matlab_clahe2/gluestick/gluestick_day_night_voc_L5_k10.yml.gz";
  IRVocabulary2 voc(voc_name);
  
  IRDatabase2 db(voc, true, 1); // false = do not use direct index
  // (so ignore the last param)
  // The direct index is useful if we want to retrieve the features that 
  // belong to some vocabulary node.
  // db creates a copy of the vocabulary, we may get rid of "voc" now

  // add images to the database
  for(int i = 0; i < NIMAGES; i++)
  {
    db.add(features[i]);
  }

  cout << "... done!" << endl;

  cout << "Database information: " << endl << db << endl;

  // and query the database
  cout << "Querying the database: " << endl;

  QueryResults ret;
  for(int i = 0; i < NIMAGES; i++)
  {
    db.query(features[i], ret, 4);

    // ret[0] is always the same image in this case, because we added it to the 
    // database. ret[1] is the second best match.
    cout << ret[0].Id << endl;
    cout << ret[0].nWords << endl;
    cout << "Searching for Image " << i << ". " << ret << endl;
  }

  cout << endl;

  // // we can save the database. The created file includes the vocabulary
  // // and the entries added
  // cout << "Saving database..." << endl;
  // db.save("small_db2.yml.gz");
  // cout << "... done!" << endl;
  
  // // once saved, we can load it again  
  // cout << "Retrieving database once again..." << endl;
  // IRDatabase db2("small_db2.yml.gz");
  // cout << "... done! This is: " << endl << db2 << endl;
}

// ----------------------------------------------------------------------------


