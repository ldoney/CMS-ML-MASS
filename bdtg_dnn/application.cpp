#include <cstdlib>
#include <vector>
#include <iostream>
#include <map>
#include <string>
 
#include "TFile.h"
#include "TTree.h"
#include "TString.h"
#include "TSystem.h"
#include "TROOT.h"
#include "TH1F.h"
#include "TStopwatch.h"
 
#if not defined(__CINT__) || defined(__MAKECINT__)
#include "TMVA/Tools.h"
#include "TMVA/Reader.h"
#include "TMVA/MethodCuts.h"
#endif

Bool_t UseOffsetMethod = kTRUE;
 
#define MUON_PAIR_OUTPUT_DIR "output_dir/muon_pairs_dataset/weights"
#define MUON_PAIR_OUTPUT_WEIGHTS "output_dir/muon_pairs_dataset/weights/TMVAClassification_BDTG.weights.xml"
#define MUON_PAIR_TARGET_FILE "output_dir/muon_pairs_dataset/TMVapplication.root"

void application() {
   std::map<std::string,int> Use;
  
   Use["LikelihoodCat"] = 1;
   Use["FisherCat"]     = 1;

   TMVA::Reader *reader = new TMVA::Reader( "!Color:!Silent" );

   Float_t mass, charge, pt, eta, phi;
   reader->AddVariable( "muPairs.mass", &mass);
   reader->AddVariable( "muPairs.charge", &charge);
   reader->AddVariable( "muPairs.pt", &pt);
   reader->AddVariable( "muPairs.eta", &eta);
   reader->AddVariable( "muPairs.phi", &phi);
 
   for (std::map<std::string,int>::iterator it = Use.begin(); it != Use.end(); it++) {
      if (it->second) {
         TString methodName = it->first + " method";
         TString weightfile = MUON_PAIR_OUTPUT_WEIGHTS;
         reader->BookMVA( methodName, weightfile );
      }
   }

   UInt_t nbin = 100;
   std::map<std::string,TH1*> hist;
   hist["LikelihoodCat"] = new TH1F( "MVA_LikelihoodCat",   "MVA_LikelihoodCat", nbin, -1, 0.9999 );
   hist["FisherCat"]     = new TH1F( "MVA_FisherCat",       "MVA_FisherCat",     nbin, -4, 4 );


   TFile *input = TFile::Open("signal_data.root");
   if (!input) {
      std::cout << "ERROR: could not open data file: " << "signal_data.root" << std::endl;
      exit(1);
   }
 

   TTree* theTree = input->Get<TDirectoryFile>("dimuons")->Get<TTree>("tree");
   input->ls();

   std::cout << "--- Use signal sample for evaluation" << std::endl;

   theTree->SetBranchAddress( "muPairs.mass", &mass);
   theTree->SetBranchAddress( "muPairs.charge", &charge);
   theTree->SetBranchAddress( "muPairs.pt", &pt);
   theTree->SetBranchAddress( "muPairs.eta", &eta);
   theTree->SetBranchAddress( "muPairs.phi", &phi);
 
   std::cout << "--- Processing: " << theTree->GetEntries() << " events" << std::endl;
   TStopwatch sw;
   sw.Start();

   for (Long64_t ievt=0; ievt<theTree->GetEntries();ievt++) {
 
      if (ievt%1000 == 0) std::cout << "--- ... Processing event: " << ievt << std::endl;
 
      theTree->GetEntry(ievt);
      for (std::map<std::string,int>::iterator it = Use.begin(); it != Use.end(); it++) {
         if (!it->second) continue;
         TString methodName = it->first + " method";
         hist[it->first]->Fill( reader->EvaluateMVA( methodName ) );
      }
 
   }
   sw.Stop();
   std::cout << "--- End of event loop: "; sw.Print();
 
   TFile *target  = new TFile(MUON_PAIR_TARGET_FILE ,"RECREATE" );
   for (std::map<std::string,int>::iterator it = Use.begin(); it != Use.end(); it++)
      if (it->second) hist[it->first]->Write();
 
   target->Close();
   std::cout << "--- Created root file: \"TMVApp.root\" containing the MVA output histograms" << std::endl;
 
   delete reader;
   std::cout << "==> TMVAClassificationApplication is done!" << std::endl << std::endl;
}

int main() {
  application();
  return 0;
}
