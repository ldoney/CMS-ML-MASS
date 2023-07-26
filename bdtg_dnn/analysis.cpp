void analysis() {
    TFile *file = new TFile("TMVA.root");

    TDirectoryFile *dataset = (TDirectoryFile*)file->Get("dataset");
    TTree *testtree = (TTree*)dataset->Get("TestTree");
    TTree *traintree = (TTree*)dataset->Get("TrainTree");
    testtree->Draw("BDTG>>testhistogram");
    traintree->Draw("BDTG>>trainhistogram");
    TH1 *testhistogram =  (TH1*)gDirectory->Get("testhistogram");
    TH1 *trainhistogram = (TH1*)gDirectory->Get("trainhistogram");
    auto g1 = testhistogram->DrawNormalized();
    auto g2 = trainhistogram->DrawNormalized("same");
    g1->SetLineWidth(3); g1->SetMarkerColor(kRed);
    g2->SetLineWidth(1); g2->SetMarkerColor(kRed);
}
