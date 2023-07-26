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
#include <string>
#include <iostream>
#include <nlohmann/json.hpp>

#ifndef __RUNNING_PROPERTIES
#define __RUNNING_PROPERTIES

using json = nlohmann::json;

typedef enum {
  MUONS, JETS, MUONPAIRS, MUONPAIRS_AND_JETS, ALL
} variable_set;

typedef enum {
  BDTG, DNN, ALL_METHODS
} method_for_factory;

template<typename T, typename UnaryPredicate>
T* find(std::vector<T> vec, UnaryPredicate pred) {
  auto it = std::find_if(vec.begin(), vec.end(), pred);
  if (it != vec.end()) {
    return &(*it);
  }
  return NULL;
}

template<typename T, typename UnaryPredicate>
bool contains_pred(std::vector<T> vec, UnaryPredicate pred) {
  auto it = std::find_if(vec.begin(), vec.end(), pred);
  if (it != vec.end()) {
    return true;
  }
  return false;
}

template<typename T, typename UnaryPredicate>
bool contains(std::vector<T> vec, T value) {
  return contains_pred(vec, [value](T compare) {return compare == value;});
}

template<typename T, typename UnaryPredicate>
std::vector<T> find_all(std::vector<T> vec, UnaryPredicate pred) {
  std::vector<T> matches;
  std::copy_if(vec.begin(), vec.end(), std::back_inserter(matches), pred);
  return matches;
}

std::string method_to_string(method_for_factory m) {
  switch(m) {
    case ALL_METHODS:
      return "ALL";
    case BDTG:
      return "BDTG";
    case DNN:
      return "DNN";
    default:
      return "UNKNOWN";
  }
}

typedef std::tuple<std::string, std::string, std::string, char> variable_tuple;

method_for_factory get_method_from_string(std::string str) {
   if(str == "BDTG") return BDTG;
   if(str == "DNN") return DNN;
   if(str == "ALL_METHODS") return ALL_METHODS;
   return BDTG;
}

variable_tuple get_tuple_from_string(std::string str) {
   return {str, str, "units", 'F'};
}

bool stob(std::string str) {
  return str == "true";
}

std::string btos(bool b) {
  return b ? "true" : "false";
}

std::vector<variable_tuple> variable_set_to_tuples(variable_set set) {
  std::vector<variable_tuple> toReturn;
  std::vector<std::string> collection;
  switch (set) {
    case MUONS:
      collection = {"muons.charge", "muons.pt", "muons.eta", "muons.phi"};
      break;
    case MUONPAIRS:
      collection = {"muPairs.mass", "muPairs.pt", "muPairs.eta", "muPairs.phi"};
      break;
    case JETS:
      collection = {"jets.charge", "jets.pt", "jets.eta", "jets.mass"};
      break;
    case MUONPAIRS_AND_JETS:
      collection = {
        "muPairs.mass", "muPairs.charge", "muPairs.pt", "muPairs.eta", "muPairs.phi",
        "jets.mass", "jets.charge", "jets.pt", "jets.eta", "jets.phi",
      };
      break;
    case ALL:
      // So for some reason, muPairs.charge just doesnt work. It used to at one point (super early on), but not anymore. That's fine I guess
      collection = {
        /*"muons.charge"*/"Alt$(muons.pt[0],-99)",//, "Alt$(muons.eta[0],-99)", "Alt$(muons.phi[0],-99)", 
        "Alt$(muons.pt[1],-99)",//, "Alt$(muons.eta[1],-99)", "Alt$(muons.phi[1],-99)",
        "muPairs.mass", "muPairs.pt", /*"muPairs.charge",*/ "muPairs.eta", "muPairs.phi", "muPairs.dR", "muPairs.dEta", "muPairs.dPhi", "muPairs.dPhiStar",
          "met.px", "met.py", "met.pt", "met.phi", "met.sumEt",
        /*"jets.mass[0]", "jets.charge",*/ "jets.pt[0]",//, "jets.eta[0]" "jets.phi[0]",
        "jetPairs.mass","jetPairs.pt", "jetPairs.eta", "jetPairs.phi", "jetPairs.dR", "jetPairs.dEta", "jetPairs.dPhi",
      };
      break;
    default:
      perror("Unimplemented todo");
      break;
  }
  std::transform(collection.begin(), collection.end(), std::back_inserter(toReturn), get_tuple_from_string);
  return toReturn;
}

class RunningProperties : public TObject {
  public:
    std::vector<variable_tuple> variables;
    std::vector<method_for_factory> methods;
    Int_t numSignalTrain;
    Int_t numBackgroundTrain;
    Int_t numSignalTest;
    Int_t numBackgroundTest;
    Int_t numTrees;
    Int_t maxDepth;
    Int_t numLayers;
    Int_t convergenceSteps;
    TString layerString;
    TString learningRate;
    TString cut;
    Bool_t isSuccess;

    TString produceDNNString() {
      TString layoutString("Layout=");
      for(int i = 0; i < this->numLayers; i++) {
        layoutString = layoutString + this->layerString + ",";
        if(i != this->numLayers - 1) {
          layoutString = layoutString + "BNORM,";
        }
      }
      layoutString = layoutString + "DENSE|1|LINEAR";

      TString trainingString1("LearningRate=" + learningRate + ",Momentum=0.9,Repetitions=1,"
                              "ConvergenceSteps=" + TString(std::to_string(this->convergenceSteps)) + ",BatchSize=10,TestRepetitions=1,"
                              "MaxEpochs=20,WeightDecay=1e-4,Regularization=None,"
                              "Optimizer=ADAM,DropConfig=0.0+0.0+0.0+0.");
       
      TString trainingStrategyString("TrainingStrategy=");
      trainingStrategyString += trainingString1;


      TString dnnOptions("!H:V:ErrorStrategy=CROSSENTROPY:VarTransform=None:"
                         "WeightInitialization=XAVIER");
      dnnOptions.Append(":");
      dnnOptions.Append(layoutString);
      dnnOptions.Append(":");
      dnnOptions.Append(trainingStrategyString);
       
      return dnnOptions;
    }
    bool containsMethod(method_for_factory m) {
      return (std::find(this->methods.begin(), this->methods.end(), ALL_METHODS) != this->methods.end()) 
          || (std::find(this->methods.begin(), this->methods.end(), m) != this->methods.end());
    }

    RunningProperties(std::vector<variable_tuple> variables, int numSignalTrain, int numBackgroundTrain, TString cut, std::vector<method_for_factory> methods = {BDTG, DNN, ALL_METHODS}) {
     this->numBackgroundTrain = numBackgroundTrain;
     this->numSignalTrain = numSignalTrain;
     this->methods = methods;
     this->variables = variables;
     this->cut = TString(cut);
     this->isSuccess = false;
    }
    
    RunningProperties(variable_set set, int numSignalTrain, int numBackgroundTrain, TString cut, std::vector<method_for_factory> methods = {BDTG, DNN, ALL_METHODS}) : RunningProperties(variable_set_to_tuples(set), numSignalTrain, numBackgroundTrain, cut, methods) {}

    RunningProperties clone() {
      RunningProperties rp(this->variables, this->numSignalTrain, this->numBackgroundTrain, this->cut.Data(), this->methods);
      rp.numTrees = this->numTrees;
      rp.numTrees = this->maxDepth;
      rp.numSignalTest = this->numSignalTest;
      rp.numBackgroundTest = this->numBackgroundTest;
      return rp;
    }

    RunningProperties(std::map<std::string, std::string> data) {
      std::string methodsTString = data["methods"];
      json jsonData = json::parse(methodsTString);
      std::vector<std::string> methods_as_strs = jsonData.get<std::vector<std::string>>();
      std::vector<method_for_factory> methods = {};
      std::transform(methods_as_strs.begin(), methods_as_strs.end(), back_inserter(methods), get_method_from_string);
      this->methods = methods;

      this->numSignalTrain = stoi(data["numSignalTrain"]);
      this->numBackgroundTrain = stoi(data["numBackgroundTrain"]);
      this->numSignalTest = stoi(data["numSignalTest"]);
      this->numBackgroundTest = stoi(data["numBackgroundTest"]);
      if (this->containsMethod(BDTG)) {
        this->numTrees = stoi(data["numTrees"]);
        this->maxDepth = stoi(data["maxDepth"]);
      } 
      if (this->containsMethod(DNN)) {
        this->numLayers = stoi(data["numLayers"]);
        this->convergenceSteps = stoi(data["convergenceSteps"]);
        this->layerString = TString(data["layerString"]);
        this->learningRate = TString(data["learningRate"]);
      }

      if (data["performMassCut"] != NULL) {
        this->cut = stob(data["performMassCut"]) ? "120 < muPairs.mass && muPairs.mass < 150" : "";
      }
      this->cut = data["cut"];
      this->isSuccess = stob(data["isSuccess"]);

      std::string variablesTString = data["variables"];

      jsonData = json::parse(variablesTString);
      std::vector<std::string> variables_as_strs = jsonData.get<std::vector<std::string>>();

      std::vector<variable_tuple> variables = {};
      std::transform(variables_as_strs.begin(), variables_as_strs.end(), back_inserter(variables), get_tuple_from_string);
      this->variables = variables;
    }
 
   void exclude_via_binary(int n) {
     bool enabled[this->variables.size()];
     for(int i = 0; i < this->variables.size(); i++) {
       enabled[i] = (n >> i) & 1;
     }
     for(int i = 0; i < this->variables.size(); i++) {
       if (enabled[i]) {
         this->exclude_variable(std::get<0>(this->variables[i]));
       }
     }
   }
    
    void include_variable(variable_tuple tup) {
      this->variables.insert(this->variables.begin(), tup);
    }

    void include_variable(std::string variable, std::string name, std::string unit, char type) {
      include_variable(make_tuple(variable, name, unit, type));
    }
    
    void include_variable(std::string variable, std::string name) {
      include_variable(variable, name, "units", 'F');
    }

    void exclude_variable(std::string to_exclude) {
      auto pred = [to_exclude](variable_tuple tup) {return std::get<0>(tup) == to_exclude;};
      auto all_matches = find_all(variables, pred);
      if (all_matches.size() == 0) {
        std::cout << "Variable " << to_exclude << " does not exist!" << std::endl;
      } else {
        std::cout << "Found " << std::to_string(all_matches.size()) << " matches for " << to_exclude << std::endl;
        auto it = remove_if(variables.begin(), variables.end(), pred);
        this->variables.erase(it, this->variables.end());
      }
    }

    TMVA::DataLoader *generateDataLoader(std::string path) {
      TMVA::DataLoader *dataloader = new TMVA::DataLoader(path);
      for (auto it = this->variables.begin(); it != this->variables.end(); ++it) {
        dataloader->AddVariable("Alt$(" + std::get<0>(*it) + ",0)", std::get<1>(*it), std::get<2>(*it), std::get<3>(*it));
      }
      return dataloader;
    }

    void fillFactory(TMVA::Factory *factory, TMVA::DataLoader *dataloader) {
      if (this->containsMethod(BDTG)) {
        factory->BookMethod( dataloader, TMVA::Types::kBDT, "BDTG", "!H:!V:NTrees=" + std::to_string(this->numTrees) + ":MinNodeSize=2.5%:BoostType=Grad:Shrinkage=0.10:UseBaggedBoost:BaggedSampleFraction=0.5:nCuts=20:MaxDepth="  + std::to_string(this->maxDepth));
      } 
      if (this->containsMethod(DNN)){
        factory->BookMethod(dataloader, TMVA::Types::kDL, "TMVA_DNN_GPU", produceDNNString() + ":Architecture=GPU");
     }
   }

   void fillDataLoaderForTree(TMVA::DataLoader *dataloader) {
     TCut baseCondition = "";
     for(int i = 0; i < this->variables.size(); i++) {
       std::string varName = std::get<0>(this->variables[i]);
       baseCondition = baseCondition;
     }
     TCut mycuts = baseCondition + TCut(this->cut), mycutb = baseCondition + TCut(this->cut);

     dataloader->PrepareTrainingAndTestTree( mycuts, mycutb,
      "nTrain_Signal=" + std::to_string(this->numSignalTrain) + ":nTrain_Background=" + std::to_string(this->numBackgroundTrain) + ":nTest_Signal=" + std::to_string(this->numSignalTest) + ":nTest_Background=" + std::to_string(this->numBackgroundTest) + ":SplitMode=Random:NormMode=NumEvents:!V" );
   }

   std::map<std::string, std::string> to_map() {
     std::vector<std::string> listOfMethods;
     std::transform(this->methods.begin(), this->methods.end(), back_inserter(listOfMethods), method_to_string);
     TString methodsTString = TBufferJSON::ToJSON(&listOfMethods);
     std::vector<std::string> listOfVariables;
     std::transform(this->variables.begin(), this->variables.end(), back_inserter(listOfVariables), 
       [](variable_tuple tup) {return std::get<0>(tup);});
     TString variablesTString = TBufferJSON::ToJSON(&listOfVariables);
     
     std::map<std::string,std::string> data = {
       {"numSignalTrain",std::to_string(this->numSignalTrain)},
       {"numBackgroundTrain",std::to_string(this->numBackgroundTrain)},
       {"numSignalTest",std::to_string(this->numSignalTest)},
       {"numBackgroundTest",std::to_string(this->numBackgroundTest)},
       {"cut",this->cut.Data()},
       {"isSuccess",std::to_string(this->isSuccess)},
       {"methods",methodsTString.Data()},
       {"variables",variablesTString.Data()},
     };

     std::map<std::string, std::string> addition;
     if(this->containsMethod(BDTG)) {
       addition = {
         {"numTrees",std::to_string(this->numTrees)},
         {"maxDepth",std::to_string(this->maxDepth)},
       };
     } 
     if(this->containsMethod(DNN)) {
       addition = {
         {"numLayers", std::to_string(this->numLayers)},
         {"convergenceSteps", std::to_string(this->convergenceSteps)},
         {"layerString", this->layerString.Data()},
         {"learningRate", this->learningRate.Data()},
      };
     }
     addition.insert(data.begin(), data.end());
     return data;
   }

   void Print() {
     std::cout << "RunningProperties:\n  - methods: [";
     for (std::vector<method_for_factory>::iterator it = this->methods.begin(); it != this->methods.end(); ++it) {
       if (it != this->methods.begin()) {
            std::cout << ", ";
       }
       std::cout << method_to_string(*it);
     }
     std::cout << "]\n  - variables: [";
     for (std::vector<variable_tuple>::iterator it = this->variables.begin(); it != this->variables.end(); ++it) {
       if (it != this->variables.begin()) {
            std::cout << ", ";
       }
       std::cout << std::get<0>(*it);
     }
     std::cout << "]\n  - with ";
     if (this->cut == "") {
       std::cout << "no cut";
     } else {
       std::cout << "cut: " << this->cut;
     } 
     std::cout << "\n  - numSignalTrain: " << this->numSignalTrain << "\n  - numBackgroundTrain: " << this->numBackgroundTrain;
     std::cout << "\n  - numSignalTest: " << this->numSignalTest <<   "\n  - numBackgroundTest: " << this->numBackgroundTest;
     if(this->containsMethod(BDTG)) {
       std::cout << "\n  - BDTG:";
       std::cout << "\n    - numTrees: " << this->numTrees << "\n    - maxDepth: " << this->maxDepth;
     } 
     if(this->containsMethod(DNN)) {
       std::cout << "\n  - DNN:";
       std::cout << "\n    - numLayers: " << this->numLayers << "\n  - convergenceSteps: " << this->convergenceSteps << "\n    - layerString: " << this->layerString << "\n    - learningRate: " << this->learningRate << "\n    - dnn string: " << this->produceDNNString();
     }
     std::cout << "\n  - isSuccess: " << btos(this->isSuccess) << std::endl;
   }
};

class RunningResults {
  public:
    Double_t rocIntegral;
    Double_t kolS;
    Double_t kolB;
    RunningProperties *associatedProperties;
    Bool_t failed;
    RunningResults(Double_t rocIntegral, Double_t kolS, Double_t kolB, RunningProperties *associatedProperties) {
      this->rocIntegral = rocIntegral;
      this->kolS = kolS;
      this->kolB = kolB;
      this->associatedProperties = associatedProperties;
      this->failed = false;
    }
    RunningResults(Bool_t failed) {
      this->failed = failed;
    }
};
#endif
