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

void process_mass() {
   // Makes root use multithreading where possible, speeds up program
   ROOT::EnableImplicitMT();

   // Set the file which is going to be accessed to process all of its contents
   TString *toProcessDir = new TString("mass_output_dir/178-13-16-MASS-DNN/");
   TString *metaDataStr = new TString(*toProcessDir + "metadata.root");
   TFile *file = TFile::Open(*metaDataStr, "READ");

   // This is roots way of doing navigation through a TFile, this is generating the iterator
   TIter nextkey(file->GetListOfKeys());
   TKey* key;

   // Keep the original directory handy, because we cd into subdirectories
   TDirectory* originalDir = gDirectory;
   std::string *originalPhysDir = new std::string(gSystem->pwd());

   // Storage for all ROCs generated in this particular run
   std::vector<TH1D*> allROCs;

   // Store all general running results
   std::vector<RunningResults> results;

   // Iterate through all files in directory
   while ((key = static_cast<TKey*>(nextkey()))) {
     // Navigate back to root directory
     gDirectory->cd();
     gSystem->cd(originalPhysDir->c_str());

     // Get running properties in map form
     std::map<std::string, std::string> *runningprop_map = NULL;
     file->GetObject(key->GetName(), runningprop_map);
     if(runningprop_map == NULL) {
       std::cout << "Map is null!" << std::endl;
       continue;
     }

     // Cast map form to a RunProperties object
     RunProperties *properties = new RunProperties(*runningprop_map);

     // Display the properties
     properties->Print();

     // Navigate to a particular run
     TString *runDir = new TString(*toProcessDir + "Run-" + key->GetName() + "/");
     gSystem->cd(*runDir);
    
     // Initialize a new RunningResults object
     RunningResults res(0, 0, 0, properties);

     // Open up the TMVA object associated with this particular run
     TFile *file = TFile::Open("TMVA.root", "READ");


     TDirectoryFile *dir = NULL;
     TH1D *rocCurve = NULL;
     // Detect if this particular run has a BDTG or a DNN or both
     auto hasBDTG = std::find(properties->methods.begin(), properties->methods.end(), BDTG) != properties->methods.end();
     auto hasDNN = std::find(properties->methods.begin(), properties->methods.end(), DNN) != properties->methods.end();

     // Collect the ROC curve
     try {
       // For now, I'll only deal with one or the other (prioritizing BDTG). But, theoretically, we could have both
       if (hasBDTG) {
        dir = file->Get<TDirectoryFile>("dataset")->Get<TDirectoryFile>("Method_BDT")->Get<TDirectoryFile>("BDTG");
        rocCurve = dir->Get<TH1D>("MVA_BDTG_rejBvsS");
       } else if (hasDNN){
        dir = file->Get<TDirectoryFile>("dataset")->Get<TDirectoryFile>("Method_DNN")->Get<TDirectoryFile>("DNN_CPU");
        rocCurve = dir->Get<TH1D>("MVA_DNN_CPU_rejBvsS");
       }
       // Combine all ROCs together for later
       allROCs.insert(allROCs.end(), rocCurve);

       // Integrate ROC curve for this run and print it out
       res.rocIntegral = rocCurve->Integral(rocCurve->FindFixBin(0), rocCurve->FindFixBin(1), "");
       std::cout << key->GetName() << ": " << res.rocIntegral << std::endl;

       // Collect all of the relevant histograms (ROOT loves histograms)
       TH1* sig = NULL;
       TH1* bgd = NULL;
       TH1* sigOv = NULL;
       TH1* bgdOv = NULL;
       // Again, we could do both but I only do one or the other (prioritizing BDTG)
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

       // Compute Kolmogorov Test (overtraining check)
       res.kolS = sig->KolmogorovTest( sig, "X" );
       res.kolB = bgd->KolmogorovTest( bgd, "X" );

       results.insert(results.end(), res);
     } catch(...) {
       // If something fails, let us know!
       std::cout << "Failed to process run " << key->GetName() << "!" << std::endl;
       results.insert(results.end(), RunningResults(false));
     }
     delete map;
  }

  // Print all of the ROC curves
  allROCs[0]->Draw();
  std::vector<TH1D*>::iterator it = allROCs.begin();
  std::advance(it, 1);
  while(it != allROCs.end()) {
    (*it)->Draw("SAME");
    ++it;
  }

  // Pick out the best run of all of them
  RunningResults *best = &results[0];
  for(RunningResults res : results) {
    if (!res.failed) {
      best = res.rocIntegral > best->rocIntegral ? &res : best;
    }
  }

  // Print the properties associated with the best run
  std::cout << "Best found running result with integral " << best->rocIntegral << std::endl;
  best->associatedProperties->Print();
}

int main(int argc, char ** argv) {
    TApplication app("MyApp", &argc, argv);
    process_mass();
    app.Run();
    return 0;
}


