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

void process_mass() {
   ROOT::EnableImplicitMT();
   TString *toProcessDir = new TString("mass_output_dir/178-13-16-MASS-DNN/");
   TString *metaDataStr = new TString(*toProcessDir + "metadata.root");
   TFile *file = TFile::Open(*metaDataStr, "READ");

   TIter nextkey(file->GetListOfKeys());
   TKey* key;

   TDirectory* originalDir = gDirectory;
   std::string *originalPhysDir = new std::string(gSystem->pwd());
   std::vector<TH1D*> allROCs;
   std::vector<RunningResults> results;
   while ((key = static_cast<TKey*>(nextkey()))) {
     gDirectory->cd();
     gSystem->cd(originalPhysDir->c_str());

     std::map<std::string, std::string> *map = NULL;
     file->GetObject(key->GetName(), map);
      
     if(map == NULL) {
       std::cout << "Map is null!" << std::endl;
       continue;
     }
     RunningProperties *properties = new RunningProperties(*map);
     properties->Print();
     if(true || properties->isSuccess) {
       TString *runDir = new TString(*toProcessDir + "Run-" + key->GetName() + "/");
       gSystem->cd(*runDir);
      
       RunningResults res(0, 0, 0, properties);
       TFile *file = TFile::Open("TMVA.root", "READ");
       TDirectoryFile *dir = NULL;
       TH1D *rocCurve = NULL;
       auto hasBDTG = std::find(properties->methods.begin(), properties->methods.end(), BDTG) != properties->methods.end();
       auto hasDNN = std::find(properties->methods.begin(), properties->methods.end(), DNN) != properties->methods.end();
       try {
       // For now, I'll only deal with one or the other. But, theoretically, we could have both
       if (hasBDTG) {
        dir = file->Get<TDirectoryFile>("dataset")->Get<TDirectoryFile>("Method_BDT")->Get<TDirectoryFile>("BDTG");
        rocCurve = dir->Get<TH1D>("MVA_BDTG_rejBvsS");
       } else if (hasDNN){
        dir = file->Get<TDirectoryFile>("dataset")->Get<TDirectoryFile>("Method_DNN")->Get<TDirectoryFile>("DNN_CPU");
        rocCurve = dir->Get<TH1D>("MVA_DNN_CPU_rejBvsS");
       }
       allROCs.insert(allROCs.end(), rocCurve);
       res.rocIntegral = rocCurve->Integral(rocCurve->FindFixBin(0), rocCurve->FindFixBin(1), "");
       std::cout << key->GetName() << ": " << res.rocIntegral << std::endl;

       TH1* sig = NULL;
       TH1* bgd = NULL;
       TH1* sigOv = NULL;
       TH1* bgdOv = NULL;
       if(hasBDTG) {
         sig = dir->Get<TH1D>("MVA_BDTG_S");
         bgd = dir->Get<TH1D>("MVA_BDTG_B");
         sigOv = dir->Get<TH1D>("MVA_BDTG_Train_S");
         bgdOv = dir->Get<TH1D>("MVA_BDTG_Train_B");
       } else if (hasDNN) {
         sig = dir->Get<TH1D>("MVA_DNN_CPU_S");
         bgd = dir->Get<TH1D>("MVA_DNN_CPU_B");
         sigOv = dir->Get<TH1D>("MVA_DNN_CPU_Train_S");
         bgdOv = dir->Get<TH1D>("MVA_DNN_CPU_Train_B");
       }
       res.kolS = sig->KolmogorovTest( sig, "X" );
       res.kolB = bgd->KolmogorovTest( bgd, "X" );
       results.insert(results.end(), res);
       } catch(...) {
         std::cout << "Failed to process run " << key->GetName() << "!" << std::endl;
         results.insert(results.end(), RunningResults(false));
       }
     } else {
       std::cout << "Run " << key->GetName() << " was a failure!" << std::endl;
       results.insert(results.end(), RunningResults(false));
     }
     delete map;
  }
  std::cout << std::to_string(allROCs.size()) << std::endl;
  allROCs[0]->Draw();
  std::vector<TH1D*>::iterator it = allROCs.begin();
  std::advance(it, 1);
  while(it != allROCs.end()) {
    (*it)->Draw("SAME");
    ++it;
  }
  RunningResults *best = &results[0];
  for(RunningResults res : results) {
    if (!res.failed) {
      best = res.rocIntegral > best->rocIntegral ? &res : best;
    }
  }
  std::cout << "Best found running result with integral " << best->rocIntegral << std::endl;
  best->associatedProperties->Print();
}

int main(int argc, char ** argv) {
    TApplication app("MyApp", &argc, argv);
    process_mass();
    app.Run();
    return 0;
}


