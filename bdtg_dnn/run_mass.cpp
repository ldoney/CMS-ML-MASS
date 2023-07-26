#include "TFile.h"
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

int square(int value) {
  return value*value;
}

void run_mass() {
  ROOT::EnableImplicitMT();
  TTimeStamp timestamp;
  // open file and retrieve trees
  auto siginputfile = TFile::Open(SIGNAL_FILE);
  auto bginputfile = TFile::Open(BACKGROUND_FILE);
  auto sigdirectory = siginputfile->Get<TDirectoryFile>("dimuons");
  auto bgdirectory = bginputfile->Get<TDirectoryFile>("dimuons");

  auto unsliced_backgroundtree = bgdirectory->Get<TTree>("tree");
  auto unsliced_signaltree = sigdirectory->Get<TTree>("tree");

  auto sigd = new ROOT::RDataFrame(*unsliced_signaltree);
  auto sig_filtered = sigd->Filter([](ROOT::VecOps::RVec<Int_t> c) {return c.size() == 1 && c[0] == 0;}, {"muPairs.charge"});
  auto bgd = new ROOT::RDataFrame(*unsliced_backgroundtree);
  auto bg_filtered = bgd->Filter([](ROOT::VecOps::RVec<Int_t> c) {return c.size() == 1 && c[0] == 0;}, {"muPairs.charge"});

  //sigd->Foreach([](ROOT::VecOps::RVec<Double_t> d){ std::cout << d.size() << std::endl; }, {"muPairs.mass"});
  //std::vector<std::string> columns = sigd->GetColumnNames();
  //for_each(columns.begin(), columns.end(), [](std::string s) {std::cout << s << std::endl;});
  auto signal_region = sig_filtered.Filter(
      "muPairs.mass.size() == 1 && (muPairs.mass[0] > 120 && muPairs.mass[0] < 130)");
  auto background_region_1 = bg_filtered.Filter(
      "muPairs.mass.size() == 1 && (muPairs.mass[0] > 70 && muPairs.mass[0] < 110)");
  //auto background_region_2 = bg_filtered.Filter(
  //    "muPairs.mass.size() == 1 && (muPairs.mass[0] > 130)");
  auto signal_region_count = *signal_region.Count();
  auto background_region_1_count = *background_region_1.Count();
  //auto background_region_2_count = *background_region_2.Count();
  std::cout << "Signal region: " << std::to_string(signal_region_count) << std::endl;
  std::cout << "Background region 1: " << std::to_string(background_region_1_count) << std::endl;
  //std::cout << "Background region 2: " << std::to_string(background_region_2_count) << std::endl;
  return;

  TTree *backgroundtree = NULL;
  TTree *signaltree = NULL;

  variable_set todo = MUONPAIRS;
  
  std::string outputDir(OUTPUT_DIR);
  Int_t secOffset = 0;
  UInt_t hour = 0, min = 0, sec = 0;
  timestamp.GetTime(kTRUE, secOffset, &hour, &min, &sec);
  std::string timestampString = std::to_string(timestamp.GetDayOfYear()) + "-" + std::to_string(hour) + "-" + std::to_string(min) + "-";
  std::string preAsString = outputDir + timestampString + "MASS/";
  TString *pre = new TString(preAsString);

  gSystem->Exec(("mkdir " + preAsString).c_str());

  RunningProperties originalProperties(todo, 850, 3, 1000, 10000, "");

  std::string metaFileName = preAsString + "metadata.root";
  TFile *metaFile = TFile::Open(metaFileName.c_str(), "RECREATE");

  std::vector<int> toTryNumTrees = {1000};
  std::vector<int> toTryDepth =    {2};
  std::vector<int> toTrySignalNumTrain = {10000};
  std::vector<int> toTryBackgroundNumTrain = {10000};
  std::vector<TString> cutOptions = {""};

  TDirectory* originalDir = gDirectory;
  std::string *originalPhysDir = new std::string(gSystem->pwd());
  int fullSize = cutOptions.size()*toTryNumTrees.size()*toTryDepth.size()*toTrySignalNumTrain.size()*toTryBackgroundNumTrain.size();

  for (int i = 0; i < fullSize; i++) {
    gDirectory->cd();
    gSystem->cd(originalPhysDir->c_str());

    RunningProperties properties = originalProperties.clone();

    properties.numTrees   = toTryNumTrees[i % toTryNumTrees.size()];
    properties.maxDepth   = toTryDepth[(i/toTryNumTrees.size())%toTryDepth.size()];
    properties.numSignalTrain   = toTrySignalNumTrain[(i/(toTryNumTrees.size()*toTryDepth.size()))%toTrySignalNumTrain.size()];
    properties.numBackgroundTrain   = toTryBackgroundNumTrain[(i/(toTryNumTrees.size()*toTryDepth.size()*toTrySignalNumTrain.size()))%toTryBackgroundNumTrain.size()];
    properties.cut = cutOptions[(i/(toTryNumTrees.size()*toTryDepth.size()*toTrySignalNumTrain.size()*toTryBackgroundNumTrain.size()))%cutOptions.size()];
    
    std::cout << "Running iteration " << i << "/" << fullSize << ") ";
    properties.Print();

    TString *runDir = new TString(preAsString + "Run-" + std::to_string(i));
    std::string runDirAsString = std::string(runDir->Data());
    gSystem->Exec(("mkdir " + runDirAsString + "/").c_str());

    gSystem->cd(*runDir);

    TMVA::Factory *factory;
    TMVA::DataLoader *dataloader;
    try {

//       if (properties.performMassCut && 
//          ((backgroundtree->GetEntries("120 < muPairs.mass && muPairs.mass < 150") < properties.numBackgroundTrain) 
//            || (signaltree->GetEntries("120 < muPairs.mass && muPairs.mass < 150") < properties.numSignalTrain))) {
//         throw "Cut off too much";
//       }
       TString *outfileName = new TString("TMVA.root");
  
       std::cout << "Writing to " << *outfileName << "!" << std::endl;
       TFile *outputFile = TFile::Open( *outfileName, "RECREATE" );
       
       factory = new TMVA::Factory( "TMVAClassification", outputFile,
          "Silent:Color:DrawProgressBar:Transformations=I;D;P;G,D:AnalysisType=Classification" );
  
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
   
       properties.fillFactoryForMethod(factory, dataloader, BDTG);
  
       factory->TrainAllMethods();
       factory->TestAllMethods();
       factory->EvaluateAllMethods();
       outputFile->Close();
       
       std::cout << "==> Wrote root file: " << outputFile->GetName() << std::endl;
       std::cout << "==> TMVAClassification is done!" << std::endl;
       properties.isSuccess = true;
     } catch (...) {
       std::cout << "This run failed! " << std::endl;
     }
     
     std::map<std::string, std::string> properties_map = properties.to_map();
     metaFile->WriteObject(&properties_map, std::to_string(i).c_str());
     delete factory;
     delete dataloader;
   }
}

int main(int argc, char ** argv) {
    TApplication app("MyApp", &argc, argv);
    run_mass();
    app.Run();
    return 0;
}

