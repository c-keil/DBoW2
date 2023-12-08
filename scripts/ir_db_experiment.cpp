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

void loadFeatures(const vector<string> &fnames, vector<vector<vector<float>>> &features);
void readFeatures(const string &fname, vector<vector<float>> &features, const uint lim = 300);
void testVocCreation(const vector<vector<vector<float>>> &features);
void testDatabase(const vector<vector<vector<float>>> &features);
void getDescFileNames(const string strPathsFile, vector<string> &vstrDesc);
void createVocabulary(const vector<vector<vector<float>>> &features, const string vocabName);
// void testVoc(vector<vector<float>> &features, vector<vector<vector<float>>> &features2,
//              const string voc_file, const string out_file, const string db_file, const string test_file);
void testVoc(vector<vector<vector<float>>> &features1, vector<vector<vector<float>>> &features2,
             const string voc_file, const string out_file, const uint n_results = 1);

// const int k = 10;
// const int L = 6;
// const WeightingType weight = TF_IDF;
// const ScoringType scoring = L2_NORM;

int main(int argc, char **argv)
{

    if(argc != 6)
    {
        cerr << endl
             << "Usage: ir_db_experiment <querry_desc_paths_file> <paths_file> <vocab_file> <output_file> <n-results>" << endl;
        return 1;
    }
    string querry_index_file = string(argv[1]);
    string desc_index_file = string(argv[2]);
    string voc_file = string(argv[3]);
    string out_file = string(argv[4]);
    uint n_results = stoi(argv[5]);
    vector<string> querry_file_names;
    vector<string> descriptor_file_names;
    // getDescFileNames(desc_index_file1, fileNames1);
    cout << "reading files from:" << endl << desc_index_file << endl;
    getDescFileNames(desc_index_file, descriptor_file_names);
    cout << "reading files from:" << endl << querry_index_file << endl;
    getDescFileNames(querry_index_file, querry_file_names);

    vector<vector<vector<float>>> features;
    vector<vector<vector<float>>> features2;

    // loadFeatures(fileNames1, features1);

    boost::timer timer;
    cout << "loading feature set 1..." << endl;
    loadFeatures(querry_file_names, features);
    cout << "features loaded in " << timer.elapsed() << " seconds" << endl;

    cout << "loading feature set 2..." << endl;
    timer.restart();
    loadFeatures(descriptor_file_names, features2);
    cout << "features loaded in " << timer.elapsed() << " seconds" << endl;
    // readFeatures(desc_file_name, features, 300);

    //test vacabulary
    cout << "running voc test..." << endl;
    timer.restart();
    testVoc(features, features2, voc_file, out_file, n_results);
    cout << "total db experiment took " << timer.elapsed() << " seconds" << endl;
    // testVoc(features, features2, voc_file, out_file, desc_index_file);

    //write results to file
    //visualize results
    //
    return 0;
}

void readFeatures(const string &fname, vector<vector<float>> &features, const uint lim)
{
    /*reads 1 descriptor file*/
    readDescNPY(fname, features);
    if (features.size() > lim)
    {
        features.resize(lim);
    }
}

void loadFeatures(const vector<string> &fnames, vector<vector<vector<float>>> &features)
{
    uint lim = 1000;
    for (string fname : fnames)
    {
        // cout << "processing file " << fname << endl;
        vector<vector<float>> descriptors;
        readFeatures(fname, descriptors, lim); 
        features.push_back(descriptors);
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
    cout << "Found " << vstrDescFiles.size() << " descriptor files" << endl;
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

// void testVoc(vector<vector<float>> &features, vector<vector<vector<float>>> &features2, 
// const string voc_file, const string out_file, const string db_file, const string test_file)
void testVoc(vector<vector<vector<float>>> &features, vector<vector<vector<float>>> &features2, 
    const string voc_file, const string out_file, const uint n_results)
{

    // load vocabulary
    cout << "reading in voc file : " << voc_file << endl;
    boost::timer timer;
    IRVocabulary2 voc(voc_file);
    cout << "voc read from disk in " << timer.elapsed() << " seconds" << endl;
    cout << "voc : " << voc << endl;
    
    // IRDatabase2 db(voc, false, 0);
    IRDatabase2 db(voc, true, 1);
    //add features to db
    BowVector v1, v2;
    // voc.transform(features2[2], v1);
    // voc.transform(features, v2);
    // cout << "v2" << endl;
    // cout << v2 << endl;

    cout << "Adding images features to database" << endl;
    timer.restart();
    for (uint i = 0; i < features2.size(); i++)
    {
        // cout << "loading features " << i << endl;
        // voc.transform(features2[i], v1);
        // cout << "Compare :" << voc.score(v1, v2) << endl;
        // cout << "Vector : " << v1 << endl;
        db.add(features2[i]);
    }
    cout << "Building database with " << features2.size() << " took " << timer.elapsed() << " seconds" << endl;
    // cout << "Database information: " << endl << db << endl;
    // cout << "Testing feature set 2 against feature set 1" << endl;

    ofstream file;
    // file << db_file << "\n" << test_file <<  "\n"; 

    // cout << "Searching for image " << endl;
    // cout << res << endl;
    QueryResults res;
    // db.query(features, res, n_results);
    cout << "querrying db.." << endl;
    timer.restart();
    for (uint i = 0; i < features.size(); i++)
    {
        db.query(features[i], res, n_results);
        // voc.transform(features2[i], v1);
        // cout << "Vector : " << v1 << endl;
        // cout << "Searching for Image " << i << ". " << res << endl;
        // file << "Test Im: " << i << "\n";
        // file << res << "\n";
        // res[0].sumCommonVi
        file.open(out_file + to_string(i));
        for (uint j=0; j < res.size(); j++)
        {
            file << res[j].Id << ", " << res[j].Score << '\n';
        }
        file.close();

    }
    cout << "searching for " << features.size() << " images in db with " << features2.size() << " images took " << timer.elapsed() << " seconds." << endl;

    // file << res << "\n";

    // voc.transform(features2[2], v1);
    // voc.transform(features, v2);
    // // cout << "v2" << endl;
    // // cout << v2 << endl;
    // cout << "Self score :" << voc.score(v1, v2) << endl;

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