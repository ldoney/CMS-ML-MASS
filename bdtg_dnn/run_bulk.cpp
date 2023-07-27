#include "TFile.h"
#include "TBufferJSON.h"
#include "TTree.h"
#include "TGraph.h"
#include "TTimeStamp.h"
#include "TStopwatch.h"
#include "TApplication.h"
#include "TMVA/DataLoader.h"
#include "TMVA/Factory.h"
#include "TMVA/Reader.h"
#include "TMVA/TMVAGui.h"
#include <iostream>
#include <string>
#include "run_properties.cpp"

#define SIGNAL_FILE "signal_data.root"
#define BACKGROUND_FILE "background_data.root"
#define OUTPUT_DIR "mass_output_dir/"

#ifndef __MAIN
#define __MAIN

typedef std::vector<std::string> vec_s;
typedef std::vector<int> vec_i;
typedef std::vector<vec_s> vecvec_s;
typedef std::vector<vec_i> vecvec_i;

int computeFullSize(std::pair<vecvec_s, vecvec_i> p) {
  int total_s_size = 1;
  for(int i = 0; i < p.first.size(); i++) {
    total_s_size *= p.first[i].size();
  }
  int total_i_size = 1;
  for(int i = 0; i < p.second.size(); i++) {
    total_i_size *= p.second[i].size();
  }
  return total_s_size*total_i_size;
}

std::pair<vec_s, vec_i> pickBasedOnIndex(vecvec_s options_s, vecvec_i options_i, int index) {
  vec_s res_s = {};
  vec_i res_i = {};

  int total_s_size = 1;
  for(int i = 0; i < options_s.size(); i++) {
    total_s_size *= options_s[i].size();
  }

  int total_divide = 1;
  for(int j = 0; j < options_s.size(); j++) {
    vec_s toTry = options_s[j];
    res_s.push_back(toTry[(index/total_divide)%options_s[j].size()]);
    total_divide *= toTry.size();
  }
  total_divide = 1;
  for(int j = 0; j < options_i.size(); j++) {
    vec_i toTry = options_i[j];
    res_i.push_back(toTry[((index/total_s_size)/total_divide)%options_i[j].size()]);
    total_divide *= toTry.size();
  }
  return std::make_pair(res_s, res_i);
}

void run_bulk() {
  // Enable multithreading, gives us a speed boost
  ROOT::EnableImplicitMT();
  TTimeStamp timestamp;

  // open file and retrieve trees
  auto siginputfile = TFile::Open(SIGNAL_FILE);
  auto bginputfile = TFile::Open(BACKGROUND_FILE);
  auto sigdirectory = siginputfile->Get<TDirectoryFile>("dimuons");
  auto bgdirectory = bginputfile->Get<TDirectoryFile>("dimuons");
  auto unsliced_backgroundtree = bgdirectory->Get<TTree>("tree");
  auto unsliced_signaltree = sigdirectory->Get<TTree>("tree");

  int nBackground = unsliced_backgroundtree->GetEntries();
  int nSignal = unsliced_signaltree->GetEntries();
  
  // All tuneable parameters for runs. These are all runs which will be completed,
  // adding more to the list means more events will be run. WARNING: It runs one run
  // for every possible combination here, so the number of runs grows exponentially 
  // in size and thus can take a very long time.
  int divider = 1;
  variable_preset todo = ALL;
  std::vector<ml_method> toRunMethods = {ALL_METHODS};
  std::vector<std::string> cutOptions = {""};
  std::vector<int> toTrySignalNumTrain =     {1000000/divider};
  std::vector<int> toTryBackgroundNumTrain = {1000000/divider};
  std::vector<int> toTrySignalNumTest =      {4000000/divider};
  std::vector<int> toTryBackgroundNumTest =  {4000000/divider};

  // BDTG settings
  std::vector<int> toTryNumTrees = {100};
  std::vector<int> toTryMaxDepth =    {3};

  // DNN settings
  std::vector<int> numLayers = {5};
  std::vector<int> convergenceSteps = {30};
  std::vector<std::string> layerString = {"DENSE|100|RELU"};
  std::vector<std::string> learningRate = {"1e-3"};

  // Only take some number of events to actually process. divier = 1 means that every
  // event will be used
  int toTake = nBackground/divider;
  TTree *backgroundtree = NULL;
  TTree *signaltree = NULL;
  std::cout << "Only using " << toTake << " events!" << std::endl;
  if(toTake == nBackground) {
    backgroundtree = unsliced_backgroundtree;
    signaltree = unsliced_signaltree;
  } else {
    backgroundtree = unsliced_backgroundtree->CloneTree(toTake);
    signaltree = unsliced_signaltree->CloneTree(toTake);
  }
  
  // Choose name for output directory, I decided to use the timestamp to differentiate them
  // by default.
  std::string outputDir(OUTPUT_DIR);
  Int_t secOffset = 0;
  UInt_t hour = 0, min = 0, sec = 0;
  timestamp.GetTime(kTRUE, secOffset, &hour, &min, &sec);
  std::string timestampString = std::to_string(timestamp.GetDayOfYear()) + "-" + std::to_string(hour) + "-" + std::to_string(min) + "-";
  std::string output_dir_prefix = outputDir + timestampString + "BULK/";

  // Generate output directory
  gSystem->Exec(("mkdir " + output_dir_prefix).c_str());

  RunProperties originalProperties(todo, 1000, 10000, "", {DNN});

  // This is all of the meta data about each run, so we can analyze them later
  std::string metaFileName = output_dir_prefix + "metadata.root";
  TFile *metaFile = TFile::Open(metaFileName.c_str(), "RECREATE");

  std::vector<RunProperties> propertiesToRun = {};

  // This big loop is a bit complicated. It's just to generate all possible
  // combinations of runs given the input parameters from earlier
  // and return them in a big list
  for(int j = 0; j < toRunMethods.size(); j++) { 
    ml_method method = toRunMethods[j];
    
    if(method == ALL_METHODS) {
      // All of the options which are string-valued
      vecvec_s optS = {
        cutOptions,
        layerString,
        learningRate
      };
      // All of the options which are integer-valued
      vecvec_i optI = {
        toTryNumTrees,
        toTryMaxDepth, 
        toTrySignalNumTrain, 
        toTryBackgroundNumTrain, 
        toTrySignalNumTest,
        toTryBackgroundNumTest,
        numLayers,
        convergenceSteps
      };
      // Find the full number of combinations here
      int fullSize = computeFullSize(std::make_pair(optS, optI));

      // For all combinations, put them into a RunProperties object and save it to the list
      for (int i = 0; i < fullSize; i++) {
        RunProperties properties = originalProperties.clone();

        std::pair<vec_s, vec_i> p = pickBasedOnIndex(optS, optI, i);
        properties.cut = TString(p.first[0]);
        properties.layerString = TString(p.first[1]);
        properties.learningRate = TString(p.first[2]);
        properties.numTrees = p.second[0];
        properties.maxDepth = p.second[1];
        properties.numSignalTrain = p.second[2];
        properties.numBackgroundTrain = p.second[3];
        properties.numSignalTest = p.second[4];
        properties.numBackgroundTest = p.second[5];
        properties.numLayers = p.second[6];
        properties.convergenceSteps = p.second[7];

        properties.methods = {method};
        propertiesToRun.insert(propertiesToRun.end(), properties);
      }
    } else if(method == BDTG) {
      // All of the options which are string-valued
      vecvec_s optS = {
        cutOptions,
      };
      // All of the options which are integer-valued
      vecvec_i optI = {
        toTryNumTrees,
        toTryMaxDepth, 
        toTrySignalNumTrain, 
        toTryBackgroundNumTrain, 
        toTrySignalNumTest,
        toTryBackgroundNumTest
      };

      // Find the full number of combinations here
      int fullSize = computeFullSize(std::make_pair(optS, optI));

      // For every combination, put them into a RunProperties object and save it to the big list
      for (int i = 0; i < fullSize; i++) {
        RunProperties properties = originalProperties.clone();
        
        std::pair<vec_s, vec_i> p = pickBasedOnIndex(optS, optI, i);
        properties.cut = TString(p.first[0]);
        properties.numTrees = p.second[0];
        properties.maxDepth = p.second[1];
        properties.numSignalTrain = p.second[2];
        properties.numBackgroundTrain = p.second[3];
        properties.numSignalTest = p.second[4];
        properties.numBackgroundTest = p.second[5];

        properties.methods = {method};
        propertiesToRun.insert(propertiesToRun.end(), properties);
      }
    } else if(method == DNN) {
      // All of the options which are string-valued
      vecvec_s optS = {
        cutOptions,
        layerString,
        learningRate
      };
      // All of the options which are integer-valued
      vecvec_i optI = {
        toTrySignalNumTrain, 
        toTryBackgroundNumTrain, 
        toTrySignalNumTest,
        toTryBackgroundNumTest,
        numLayers,
        convergenceSteps
      };
      //std::pair<vecvec_s, vecvec_i> optPair = std::make_pair(optS, optI);

      // Find the full number of combinations here
      int fullSize = computeFullSize(std::make_pair(optS, optI));
      
      // For every combination, put them into a RunProperties object and save it to the big list
      for (int i = 0; i < fullSize; i++) {
        RunProperties properties = originalProperties.clone();
        
        std::pair<vec_s, vec_i> p = pickBasedOnIndex(optS, optI, i);
        properties.cut = TString(p.first[0]);
        properties.layerString = TString(p.first[1]);
        properties.learningRate = TString(p.first[2]);
        properties.numSignalTrain = p.second[0];
        properties.numBackgroundTrain = p.second[1];
        properties.numSignalTest = p.second[2];
        properties.numBackgroundTest = p.second[3];
        properties.numLayers = p.second[4];
        properties.convergenceSteps = p.second[5];

        properties.methods = {method};
        propertiesToRun.insert(propertiesToRun.end(), properties);
      }
    }
  }

  // Turn off histogram visualization
  gROOT->SetBatch(true);

  // Normalize all of the data
  for(int i = 0; i < propertiesToRun.size(); i++) {
    RunProperties p = propertiesToRun[i];

    // So apparently ROOT loves histograms.. the way I find the mean and stdev is to
    // generate histograms for every variable, then compute the mean/stdev there.
    for(int j = 0; j < p.variables.size(); j++) {
      variable_tuple var = p.variables[j];
      std::string varname = std::get<0>(var);
      signaltree->Draw(varname.c_str());
      auto histo = (TH1D*)gPad->GetPrimitive("htemp");
      auto sdev = histo->GetStdDev()/2;
      auto mean = histo->GetMean();
      std::get<0>(var) = "(" + varname + " - (" + mean + "))/(" + sdev + ")";
      std::get<1>(var) = "Norm_" + varname;
      propertiesToRun[i].variables[j] = var;
    }
  }

  // Turn on histogram visualization
  gROOT->SetBatch(false);

  // Save the original directory, as we're going to be moving around a bit
  TDirectory* originalDir = gDirectory;
  std::string *originalPhysDir = new std::string(gSystem->pwd());

  for(int i = 0; i < propertiesToRun.size(); i++) {
    // Move into the directory of this particular run
    gDirectory->cd();
    gSystem->cd(originalPhysDir->c_str());

    // Find and print the RunProperties object associated with this run
    RunProperties properties = propertiesToRun[i];
    std::cout << "Running iteration " << i << "/" << propertiesToRun.size() << ") ";
    properties.Print();

    // Make the directory for this particular run
    TString *runDir = new TString(output_dir_prefix + "Run-" + std::to_string(i));
    std::string runDirAsString = std::string(runDir->Data());
    gSystem->Exec(("mkdir " + runDirAsString + "/").c_str());

    // Move into the directory of the new run
    gSystem->cd(*runDir);

    // Create objects for run
    TMVA::Factory *factory;
    TMVA::DataLoader *dataloader;
    try {
      // Save outputs of ML run
      TString *outfileName = new TString("TMVA.root");
  
      std::cout << "Writing to " << *outfileName << "!" << std::endl;
      TFile *outputFile = TFile::Open( *outfileName, "RECREATE" );
      
      // Create the factory which TMVA uses to allocate runs
      factory = new TMVA::Factory( "TMVAClassification", outputFile,
         "!Silent:Color:DrawProgressBar:Transformations=I;D;P;G,D:AnalysisType=Classification" );
  
      dataloader=NULL;

      TString *path = new TString("dataset");
      std::string path_as_str(path->Data());
  
      dataloader = properties.generateDataLoader(path_as_str);
  
      Double_t signalWeight     = 1.0;
      Double_t backgroundWeight = 1.0;
  
      dataloader->AddSignalTree    ( signaltree,     signalWeight );
      dataloader->AddBackgroundTree( backgroundtree, backgroundWeight );
  
      dataloader->SetBackgroundWeightExpression( "PU_wgt" );
  
      properties.fillDataLoaderForTree(dataloader);
  
      properties.fillFactory(factory, dataloader);
  
      factory->TrainAllMethods();
      factory->TestAllMethods();
      factory->EvaluateAllMethods();
      outputFile->Close();
      
      properties.isSuccess = true;

      std::cout << "==> Wrote root file: " << outputFile->GetName() << std::endl;
      std::cout << "==> TMVAClassification is done!" << std::endl;
    } catch (...) {
      std::cout << "This run failed! " << std::endl;
    }
    
    std::map<std::string, std::string> properties_map = properties.to_map();
    metaFile->WriteObject(&properties_map, std::to_string(i).c_str());
    delete factory;
    delete dataloader;
  }

  gDirectory->cd();
  gSystem->cd(originalPhysDir->c_str());

  std::cout << "Completed run! Directory: " << output_dir_prefix << std::endl;

  //if(propertiesToRun.size() == 1) {
  //  TString *runDir = new TString(output_dir_prefix + "Run-0");
  //  TMVA::TMVAGui(*runDir + "/TMVA.root");
  //}
}

int main(int argc, char ** argv) {
    TApplication app("MyApp", &argc, argv);
    run_bulk();
    return 0;
}
#endif
