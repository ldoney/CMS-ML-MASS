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

#define SIGNAL_FILE "signal_data.root"
#define BACKGROUND_FILE "background_data.root"
#define OUTPUT_DIR ""


enum {
  MUONS, JETS, MUONPAIRS, MUONPAIRS_AND_JETS
};

void run_single() {
   ROOT::EnableImplicitMT();
   TTimeStamp timestamp;

   // open file and retrieve trees
   auto siginputfile = TFile::Open(SIGNAL_FILE);
   auto bginputfile = TFile::Open(BACKGROUND_FILE);
   auto sigdirectory = siginputfile->Get<TDirectoryFile>("dimuons");
   auto bgdirectory = bginputfile->Get<TDirectoryFile>("dimuons");

   auto unsliced_backgroundtree = bgdirectory->Get<TTree>("tree");
   auto unsliced_signaltree = sigdirectory->Get<TTree>("tree");

   auto backgroundtree = unsliced_backgroundtree->CloneTree(10000);
   auto signaltree = unsliced_signaltree->CloneTree(1000);
   int todo = MUONPAIRS;

   TString *outputDir = new TString(OUTPUT_DIR);
   Int_t secOffset = 0;
   UInt_t hour = 0, min = 0, sec = 0;
   timestamp.GetTime(kTRUE, secOffset, &hour, &min, &sec);
   TString *timestampString = new TString(std::to_string(hour) + "-" + std::to_string(min) + "-");
   TString *pre = new TString(*outputDir + *timestampString);
   
   TString *outfileName = NULL;
   switch(todo) {
     case MUONS: 
       outfileName = new TString("TMVA_MUONS.root");
       break;
     case JETS:
       outfileName = new TString( "TMVA_JETS.root");
       break;
     case MUONPAIRS:
       outfileName = new TString( "TMVA_MU_PAIRS.root");
       break;
     case MUONPAIRS_AND_JETS:
       outfileName = new TString( "TMVA_MU_PAIRS_JETS.root");
       break;
     default:
       perror("Invalid todo! Exiting...");
       return;
   }
   outfileName = new TString(*pre + *outfileName);

   std::cout << "Writing to " << *outfileName << "!" << std::endl;
   TFile *outputFile = TFile::Open( *outfileName, "RECREATE" );
   
   TMVA::Factory *factory = new TMVA::Factory( "TMVAClassification", outputFile,
      "!V:!Silent:Color:DrawProgressBar:Transformations=I;D;P;G,D:AnalysisType=Classification" );

   TMVA::DataLoader *dataloader=NULL;
   TString *path = NULL;
   switch (todo) {
     case MUONS:
       path = new TString("muons_dataset");
       break;
     case JETS:
       path = new TString("jets_dataset");
       break;
     case MUONPAIRS:
       path = new TString("muon_pairs_dataset");
       break;
     case MUONPAIRS_AND_JETS:
       path = new TString("muon_pairs_and_jets_dataset");
       break;
     default:
       perror("Unimplemented todo");
       return;
   }

   path = new TString(*pre + *path);
   std::string path_as_str(path->Data());
   dataloader = new TMVA::DataLoader(path_as_str);
   switch (todo) {
     case MUONS:
       dataloader->AddVariable( "muons.charge", "Charge", "", 'F' );
       dataloader->AddVariable( "muons.pt", "PT", "", 'F' );
       dataloader->AddVariable( "muons.eta", "Eta", "units", 'F' );
       dataloader->AddVariable( "muons.phi", "Phase", "units", 'F' );
       break;
     case JETS:
       dataloader->AddVariable( "jets.charge", "Charge", "", 'F' );
       dataloader->AddVariable( "jets.pt", "PT", "", 'F' );
       dataloader->AddVariable( "jets.eta", "Eta", "units", 'F' );
       dataloader->AddVariable( "jets.phi", "Phase", "units", 'F' );
       break;
     case MUONPAIRS:
       dataloader->AddVariable( "muPairs.mass", "Mass", "", 'F' );
       dataloader->AddVariable( "muPairs.pt", "PT", "", 'F' );
       dataloader->AddVariable( "muPairs.eta", "Eta", "units", 'F' );
       dataloader->AddVariable( "muPairs.phi", "Phase", "units", 'F' );
       break;
     case MUONPAIRS_AND_JETS:
       dataloader->AddVariable( "muPairs.mass", "MuonPair Mass", "units", 'F' );
       dataloader->AddVariable( "muPairs.charge", "MuonPair Charge", "units", 'F' );
       dataloader->AddVariable( "muPairs.pt", "MuonPair PT", "units", 'F' );
       dataloader->AddVariable( "muPairs.eta", "MuonPair Eta", "units", 'F' );
       dataloader->AddVariable( "muPairs.phi", "MuonPair Phase", "units", 'F' );
       dataloader->AddVariable( "jets.mass", "Jet Mass", "", 'F' );
       dataloader->AddVariable( "jets.charge", "Jet Charge", "", 'F' );
       dataloader->AddVariable( "jets.pt", "Jet PT", "", 'F' );
       dataloader->AddVariable( "jets.eta", "Jet Eta", "units", 'F' );
       dataloader->AddVariable( "jets.phi", "Jet Phase", "units", 'F' );
       break;
     default:
       perror("Unimplemented todo");
       return;
   }

   Double_t signalWeight     = 1.0;
   Double_t backgroundWeight = 1.0;

   dataloader->AddSignalTree    ( signaltree,     signalWeight );
   dataloader->AddBackgroundTree( backgroundtree, backgroundWeight );

   dataloader->SetBackgroundWeightExpression( "PU_wgt" );

   TCut mycuts = ""; // for example: TCut mycuts = "abs(var1)<0.5 && abs(var2-0.5)<1";
   TCut mycutb = ""; // for example: TCut mycutb = "abs(var1)<0.5";

   dataloader->PrepareTrainingAndTestTree( mycuts, mycutb,
      "nTrain_Signal=100:nTrain_Background=1000:SplitMode=Random:NormMode=NumEvents:!V" );
 
   factory->BookMethod( dataloader, TMVA::Types::kBDT, "BDTG",
     "!H:!V:NTrees=800:MinNodeSize=2.5%:BoostType=Grad:Shrinkage=0.10:UseBaggedBoost:BaggedSampleFraction=0.5:nCuts=20:MaxDepth=2" );

   factory->TrainAllMethods();
   factory->TestAllMethods();
   factory->EvaluateAllMethods();
   outputFile->Close();
   std::cout << "==> Wrote root file: " << outputFile->GetName() << std::endl;
   std::cout << "==> TMVAClassification is done!" << std::endl;
   //if (!gROOT->IsBatch()) TMVA::TMVAGui( *outfileName );
   delete factory;
   delete dataloader;
}

int main(int argc, char ** argv) {
    TApplication app("MyApp", &argc, argv);
    run_single();
    app.Run();
    return 0;
}
