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
#include "running_properties.cpp"

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

void run_mass_no_precut() {
   ROOT::EnableImplicitMT();
   TTimeStamp timestamp;

   // open file and retrieve trees
   
   auto siginputfile = TFile::Open(SIGNAL_FILE);
   auto bginputfile = TFile::Open(BACKGROUND_FILE);
   auto sigdirectory = siginputfile->Get<TDirectoryFile>("dimuons");
   auto bgdirectory = bginputfile->Get<TDirectoryFile>("dimuons");
   auto unsliced_backgroundtree = bgdirectory->Get<TTree>("tree");
   auto unsliced_signaltree = sigdirectory->Get<TTree>("tree");

   //auto unsliced_signaltree = TFile::Open("tree_output_dir/signal_data.root")->Get<TTree>("tree");
   //auto unsliced_backgroundtree = TFile::Open("tree_output_dir/background_data.root")->Get<TTree>("tree");

   int nBackground = unsliced_backgroundtree->GetEntries();
   int nSignal = unsliced_signaltree->GetEntries();
   
   int divider = 1;
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
   variable_set todo = ALL;
   
   std::string outputDir(OUTPUT_DIR);
   Int_t secOffset = 0;
   UInt_t hour = 0, min = 0, sec = 0;
   timestamp.GetTime(kTRUE, secOffset, &hour, &min, &sec);
   std::string timestampString = std::to_string(timestamp.GetDayOfYear()) + "-" + std::to_string(hour) + "-" + std::to_string(min) + "-";
   std::string preAsString = outputDir + timestampString + "MASS-DNN/";
   TString *pre = new TString(preAsString);

   gSystem->Exec(("mkdir " + preAsString).c_str());

   RunningProperties originalProperties(todo, 1000, 10000, "", {DNN});

   std::string metaFileName = preAsString + "metadata.root";
   TFile *metaFile = TFile::Open(metaFileName.c_str(), "RECREATE");

   std::vector<RunningProperties> propertiesToRun = {};
   std::vector<method_for_factory> toRunMethods = {ALL_METHODS};
   std::vector<int> toTrySignalNumTrain =     {1000000/divider};
   std::vector<int> toTryBackgroundNumTrain = {1000000/divider};
   std::vector<int> toTrySignalNumTest =      {4000000/divider};
   std::vector<int> toTryBackgroundNumTest =  {4000000/divider};

   std::vector<int> toTryNumTrees = {100};
   std::vector<int> toTryMaxDepth =    {3};

   std::vector<int> numLayers = {5};
   std::vector<int> convergenceSteps = {30};
   
   std::vector<std::string> cutOptions = {""};
   std::vector<std::string> layerString = {"DENSE|100|RELU"};
   std::vector<std::string> learningRate = {"1e-3"};

   for(int j = 0; j < toRunMethods.size(); j++) { 
     method_for_factory method = toRunMethods[j];
     
     if(method == ALL_METHODS) {
       vecvec_s optS = {
         cutOptions,
         layerString,
         learningRate
       };
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
       std::pair<vecvec_s, vecvec_i> optPair = std::make_pair(optS, optI);
       int fullSize = computeFullSize(optPair);

       for (int i = 0; i < fullSize; i++) {
         RunningProperties properties = originalProperties.clone();

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
       vecvec_s optS = {
         cutOptions,
       };
       vecvec_i optI = {
         toTryNumTrees,
         toTryMaxDepth, 
         toTrySignalNumTrain, 
         toTryBackgroundNumTrain, 
         toTrySignalNumTest,
         toTryBackgroundNumTest
       };
       std::pair<vecvec_s, vecvec_i> optPair = std::make_pair(optS, optI);
       int fullSize = computeFullSize(optPair);

       for (int i = 0; i < fullSize; i++) {
         RunningProperties properties = originalProperties.clone();
         
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
       vecvec_s optS = {
         cutOptions,
         layerString,
         learningRate
       };
       vecvec_i optI = {
         toTrySignalNumTrain, 
         toTryBackgroundNumTrain, 
         toTrySignalNumTest,
         toTryBackgroundNumTest,
         numLayers,
         convergenceSteps
       };
       std::pair<vecvec_s, vecvec_i> optPair = std::make_pair(optS, optI);

       int fullSize = computeFullSize(optPair);
       for (int i = 0; i < fullSize; i++) {
         RunningProperties properties = originalProperties.clone();
         
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

   gROOT->SetBatch(true);
   for(int i = 0; i < propertiesToRun.size(); i++) {
     RunningProperties p = propertiesToRun[i];
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
   gROOT->SetBatch(false);

   TDirectory* originalDir = gDirectory;
   std::string *originalPhysDir = new std::string(gSystem->pwd());

   for(int i = 0; i < propertiesToRun.size(); i++) {
     gDirectory->cd();
     gSystem->cd(originalPhysDir->c_str());

     RunningProperties properties = propertiesToRun[i];
     std::cout << "Running iteration " << i << "/" << propertiesToRun.size() << ") ";
     properties.Print();

     TString *runDir = new TString(preAsString + "Run-" + std::to_string(i));
     std::string runDirAsString = std::string(runDir->Data());
     gSystem->Exec(("mkdir " + runDirAsString + "/").c_str());

     gSystem->cd(*runDir);

     TMVA::Factory *factory;
     TMVA::DataLoader *dataloader;
     try {

       TString *outfileName = new TString("TMVA.root");
  
       std::cout << "Writing to " << *outfileName << "!" << std::endl;
       TFile *outputFile = TFile::Open( *outfileName, "RECREATE" );
       
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

   std::cout << "Completed run! Directory: " << preAsString << std::endl;

   //if(propertiesToRun.size() == 1) {
   //  TString *runDir = new TString(preAsString + "Run-0");
   //  TMVA::TMVAGui(*runDir + "/TMVA.root");
   //}
}

int main(int argc, char ** argv) {
    TApplication app("MyApp", &argc, argv);
    run_mass_no_precut();
    return 0;
}
#endif
