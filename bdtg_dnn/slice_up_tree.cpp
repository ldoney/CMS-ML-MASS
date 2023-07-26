#include "TFile.h"
#include "TTree.h"
#include "TGraph.h"
#include "TTimeStamp.h"
#include "TStopwatch.h"
#include "TApplication.h"
#include <ROOT/RDataFrame.hxx>
#include <iostream>
#include <string>
#include "running_properties.cpp"

#define SIGNAL_FILE "signal_data.root"
#define BACKGROUND_FILE "background_data.root"
#define OUTPUT_DIR "tree_output_dir/"

#define TO_TAKE 100000

void slice_up_tree() {
   //ROOT::EnableImplicitMT();

   // open file and retrieve trees
   auto siginputfile = TFile::Open(SIGNAL_FILE);
   auto bginputfile = TFile::Open(BACKGROUND_FILE);
   auto sigdirectory = siginputfile->Get<TDirectoryFile>("dimuons");
   auto bgdirectory = bginputfile->Get<TDirectoryFile>("dimuons");
   auto unsliced_backgroundtree = bgdirectory->Get<TTree>("tree");
   auto unsliced_signaltree = sigdirectory->Get<TTree>("tree");
   
   std::vector<std::pair<std::string, std::string>> columnsToKeep = {
     {"muPairs", "mass"},
     {"muPairs", "charge"},
     {"muPairs", "pt"},
   };

   std::vector<std::string> combinedOldToKeep = {};
   std::transform(columnsToKeep.begin(), columnsToKeep.end(), std::back_inserter(combinedOldToKeep), [](std::pair<std::string, std::string> p) {return p.first + "." + p.second;});
  
   std::vector<std::string> combinedNewToKeep = {};
   std::transform(columnsToKeep.begin(), columnsToKeep.end(), std::back_inserter(combinedNewToKeep), [](std::pair<std::string, std::string> p) {return p.first + "_" + p.second;});

   ROOT::RDataFrame sigdf("tree", sigdirectory, combinedOldToKeep);
   ROOT::RDataFrame bgdf("tree", bgdirectory, combinedOldToKeep);

   auto filter = [](ROOT::VecOps::RVec<Double_t> mass){return mass.size() == 1 && (mass[0] < 140) && (mass[0] > 110); };

   auto sigdf2 = sigdf.Filter(filter, {"muPairs.mass"})
                      .Range(TO_TAKE);
   auto bgdf2 = bgdf.Filter(filter, {"muPairs.mass"})
                      .Range(TO_TAKE);

   std::for_each(columnsToKeep.begin(), columnsToKeep.end(), 
       [&sigdf2, &bgdf2](std::pair<std::string, std::string> p) {
     std::string f = p.first;
     std::string s = p.second;
     sigdf2 = sigdf2.Define(f + "_" + s, f + "." + s + "[0]");
     bgdf2 = bgdf2.Define(f + "_" + s, f + "." + s + "[0]");
   });

   std::for_each(combinedNewToKeep.begin(), combinedNewToKeep.end(), 
                 [](std::string s) {std::cout << s << std::endl;});

   combinedNewToKeep.push_back("PU_wgt");
   sigdf2.Snapshot("tree", std::string(OUTPUT_DIR) + "signal_data.root", combinedNewToKeep);
   bgdf2.Snapshot("tree", std::string(OUTPUT_DIR) + "background_data.root", combinedNewToKeep);
}

int main(int argc, char ** argv) {
    TApplication app("MyApp", &argc, argv);
    slice_up_tree();
    return 0;
}
