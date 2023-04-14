// Remember that to compile w/ g++ you need -> g++ postprocessor_2d.cpp -o postprocessor_2d.exe `root-config --cflags --glibs`
// Note: This should be compiled after sourcing FCCAnalyses' setup.sh so it can be run directly after without having to reset the env.


// This code is a barbaric way of reshaping <vectors>, I expect it to be (much) slower than coffea, but let's have a look -> it's actually much faster, go figure!

#include "TFile.h"
#include "TH1.h"
#include "TH2.h"
#include "TProfile.h"
#include "TRandom.h"
#include "TTree.h"
#include "TInterpreter.h"


#include <iostream>
#include <string>
#include <limits.h>

//gInterpreter->GenerateDictionary("vector<vector<float> >", "vector");


//int main(){
//char *defaults[] = { "chunk0.root"};
int main(int argc, char** argv){


   //TCHAR NPath[MAX_PATH];
   //GetCurrentDirectory(MAX_PATH, NPath);
   char buffer[PATH_MAX-1];
   getcwd(buffer, PATH_MAX-1);
   std::string pwd(buffer);
   std::cout<<pwd<<std::endl;
   //TInterpreter *ggInterpreter;
   gInterpreter->GenerateDictionary("vector<vector<float>>", "vector");
   gInterpreter->GenerateDictionary("vector<vector<double>>", "vector");
   gInterpreter->GenerateDictionary("vector<vector<int>>", "vector");
   std::string fname = "placeholder";
   std::string flav_type = "";
   if(argc<2){
     //std::cout<<"Please input file and flav type (uds, cc, bb)!"<<std::endl; 
     std::cout<<"Please input file!"<<std::endl; 
     return 1;
   }
   //if(argc=2) fname = argv[1];
   //fname = "/afs/cern.ch/user/e/eploerer/private/FCCAna_01122021/FCCAnalyses/outputs/FCCee/TNtupler/p8_ee_Zuds_ecm91.root";//argv[1];
   fname = argv[1];
   //////////flav_type = argv[2];
   
   //std::cout<<fname[fname.size()-2]<<fname[fname.size()-1]<<fname[fname.size()]<<std::endl;
   TFile *f = new TFile(fname.c_str()); // new moves object to heap as described here: https://stackoverflow.com/questions/655065/when-should-i-use-the-new-keyword-in-c
   //TTree *t1 = (TTree*)f->Get("deepntuplizer/tree");
   TTree *t1 = (TTree*)f->Get("events");
   Int_t nentries = (Int_t)t1->GetEntries();
   std::cout<<"The number of events is "<<nentries<<std::endl;
   
   //Jet Flavour
   std::vector<int> *isU = new std::vector<int>;
   std::vector<int> *isD = new std::vector<int>;
   std::vector<int> *isS = new std::vector<int>;
   std::vector<int> *isC = new std::vector<int>;
   std::vector<int> *isB = new std::vector<int>;
   std::vector<int> *isUndefined = new std::vector<int>;
   std::vector<int> *isU_ghost = new std::vector<int>;
   std::vector<int> *isD_ghost = new std::vector<int>;
   std::vector<int> *isS_ghost = new std::vector<int>;
   std::vector<int> *isC_ghost = new std::vector<int>;
   std::vector<int> *isB_ghost = new std::vector<int>;
   std::vector<int> *isUndefined_ghost = new std::vector<int>;
   std::vector<int> *isU_Z = new std::vector<int>;
   std::vector<int> *isD_Z = new std::vector<int>;
   std::vector<int> *isS_Z = new std::vector<int>;
   std::vector<int> *isC_Z = new std::vector<int>;
   std::vector<int> *isB_Z = new std::vector<int>;
   std::vector<int> *isUndefined_Z = new std::vector<int>;
   std::vector<int> *isU_ghost_angle03status70 = new std::vector<int>;
   std::vector<int> *isD_ghost_angle03status70 = new std::vector<int>;
   std::vector<int> *isS_ghost_angle03status70 = new std::vector<int>;
   std::vector<int> *isC_ghost_angle03status70 = new std::vector<int>;
   std::vector<int> *isB_ghost_angle03status70 = new std::vector<int>;
   std::vector<int> *isUndefined_ghost_angle03status70 = new std::vector<int>;
   std::vector<int> *isU_ghost_angle03status20 = new std::vector<int>;
   std::vector<int> *isD_ghost_angle03status20 = new std::vector<int>;
   std::vector<int> *isS_ghost_angle03status20 = new std::vector<int>;
   std::vector<int> *isC_ghost_angle03status20 = new std::vector<int>;
   std::vector<int> *isB_ghost_angle03status20 = new std::vector<int>;
   std::vector<int> *isUndefined_ghost_angle03status20 = new std::vector<int>;
   //std::vector<int> *Z_flavour = new std::vector<int>;
   
   //Jet-level variables
   std::vector<float> *jets_p = new std::vector<float>;
   std::vector<float> *jets_px = new std::vector<float>;
   std::vector<float> *jets_py = new std::vector<float>;
   std::vector<float> *jets_pz = new std::vector<float>;
   std::vector<float> *jets_theta = new std::vector<float>;
   std::vector<float> *jets_phi = new std::vector<float>;
   std::vector<float> *jets_m = new std::vector<float>;
   std::vector<float> *jets_e = new std::vector<float>;
   std::vector<float> *jets_nRP_charged = new std::vector<float>;
   std::vector<float> *jets_nRP_neutral = new std::vector<float>;
   
   std::vector<float> *jets_pt = new std::vector<float>;
   std::vector<float> *jets_eta = new std::vector<float>;
   
   //Charged-constituent variables
   std::vector<std::vector<float>> *RPj_charged_p = new std::vector<std::vector<float>>;
   std::vector<std::vector<float>> *RPj_charged_theta = new std::vector<std::vector<float>>;
   std::vector<std::vector<float>> *RPj_charged_phi = new std::vector<std::vector<float>>;
   std::vector<std::vector<float>> *RPj_charged_mass = new std::vector<std::vector<float>>;
   std::vector<std::vector<float>> *RPj_charged_Z0 = new std::vector<std::vector<float>>;
   std::vector<std::vector<float>> *RPj_charged_D0 = new std::vector<std::vector<float>>;
   std::vector<std::vector<float>> *RPj_charged_Z0_sig = new std::vector<std::vector<float>>;
   std::vector<std::vector<float>> *RPj_charged_D0_sig = new std::vector<std::vector<float>>;
   std::vector<std::vector<float>> *RPj_charged_dTheta = new std::vector<std::vector<float>>;
   std::vector<std::vector<float>> *RPj_charged_dPhi = new std::vector<std::vector<float>>;
   std::vector<std::vector<float>> *RPj_charged_pRel = new std::vector<std::vector<float>>;
   std::vector<std::vector<float>> *RPj_charged_isMuon = new std::vector<std::vector<float>>;
   std::vector<std::vector<float>> *RPj_charged_isElectron = new std::vector<std::vector<float>>;
   
   //Charged-constituent PID variables
   std::vector<std::vector<float>> *RPj_charged_is_S = new std::vector<std::vector<float>>;
   std::vector<std::vector<float>> *RPj_charged_is_Kaon = new std::vector<std::vector<float>>;
   std::vector<std::vector<float>> *RPj_charged_is_Kaon_smearedUniform080 = new std::vector<std::vector<float>>;
   std::vector<std::vector<float>> *RPj_charged_is_Kaon_smearedUniform090 = new std::vector<std::vector<float>>;
   std::vector<std::vector<float>> *RPj_charged_is_Kaon_smearedUniform095 = new std::vector<std::vector<float>>;
   std::vector<std::vector<float>> *RPj_charged_is_Kaon_smearedUniform060 = new std::vector<std::vector<float>>;
   std::vector<std::vector<float>> *RPj_charged_is_Kaon_smearedUniform040 = new std::vector<std::vector<float>>;
   std::vector<std::vector<float>> *RPj_charged_is_Kaon_smearedUniform020 = new std::vector<std::vector<float>>;
   std::vector<std::vector<float>> *RPj_charged_is_Kaon_smearedUniform000 = new std::vector<std::vector<float>>;
   std::vector<std::vector<float>> *RPj_charged_is_Kaon_smearedUniform100 = new std::vector<std::vector<float>>;
   
   //Neutral-constituent variables
   std::vector<std::vector<float>> *RPj_neutral_p = new std::vector<std::vector<float>>;
   std::vector<std::vector<float>> *RPj_neutral_pRel = new std::vector<std::vector<float>>;
   std::vector<std::vector<float>> *RPj_neutral_isPhoton = new std::vector<std::vector<float>>;

   //SV variables
   std::vector<std::vector<float>> *sv_mass = new std::vector<std::vector<float>>;
   std::vector<std::vector<float>> *sv_p = new std::vector<std::vector<float>>;
   std::vector<std::vector<int>> *sv_ntracks = new std::vector<std::vector<int>>;
   std::vector<std::vector<float>> *sv_chi2 = new std::vector<std::vector<float>>;
   std::vector<std::vector<float>> *sv_normchi2 = new std::vector<std::vector<float>>;
   std::vector<std::vector<int>> *sv_ndf = new std::vector<std::vector<int>>;
   std::vector<std::vector<float>> *sv_theta = new std::vector<std::vector<float>>;
   std::vector<std::vector<float>> *sv_phi = new std::vector<std::vector<float>>;
   std::vector<std::vector<float>> *sv_thetarel = new std::vector<std::vector<float>>;
   std::vector<std::vector<float>> *sv_phirel = new std::vector<std::vector<float>>;
   std::vector<std::vector<float>> *sv_costhetasvpv = new std::vector<std::vector<float>>;
   std::vector<std::vector<float>> *sv_dxy = new std::vector<std::vector<float>>;
   std::vector<std::vector<float>> *sv_d3d = new std::vector<std::vector<float>>;

   //V0 variables
   std::vector<std::vector<float>> *v0_pid = new std::vector<std::vector<float>>;
   std::vector<std::vector<float>> *v0_mass = new std::vector<std::vector<float>>;
   std::vector<std::vector<float>> *v0_p = new std::vector<std::vector<float>>;
   std::vector<std::vector<int>> *v0_ntracks = new std::vector<std::vector<int>>;
   std::vector<std::vector<float>> *v0_chi2 = new std::vector<std::vector<float>>;
   std::vector<std::vector<float>> *v0_normchi2 = new std::vector<std::vector<float>>;
   std::vector<std::vector<int>> *v0_ndf = new std::vector<std::vector<int>>;
   std::vector<std::vector<float>> *v0_theta = new std::vector<std::vector<float>>;
   std::vector<std::vector<float>> *v0_phi = new std::vector<std::vector<float>>;
   ///////////////std::vector<std::vector<float>> *v0_thetarel = new std::vector<std::vector<float>>;
   ///////////////std::vector<std::vector<float>> *v0_phirel = new std::vector<std::vector<float>>;
   std::vector<std::vector<float>> *v0_costhetasvpv = new std::vector<std::vector<float>>;
   std::vector<std::vector<float>> *v0_dxy = new std::vector<std::vector<float>>;
   std::vector<std::vector<float>> *v0_d3d = new std::vector<std::vector<float>>;
   
   //std::vector<std::vector<float>> *Cpfcan_BtagPf_trackPtRel = new std::vector<std::vector<float>>;
   //std::vector<float,ROOT::Detail::VecOps::RAdoptAllocator<float>> jets_p;
   t1->SetEntries(nentries);
   //t1->SetBranchAddress("jets_p",&jets_p);
   t1->SetBranchAddress("isU",&isU);
   t1->SetBranchAddress("isD",&isD);
   t1->SetBranchAddress("isS",&isS);
   t1->SetBranchAddress("isC",&isC);
   t1->SetBranchAddress("isB",&isB);
   t1->SetBranchAddress("isUndefined",&isUndefined);
   t1->SetBranchAddress("isU_ghost",&isU_ghost);
   t1->SetBranchAddress("isD_ghost",&isD_ghost);
   t1->SetBranchAddress("isS_ghost",&isS_ghost);
   t1->SetBranchAddress("isC_ghost",&isC_ghost);
   t1->SetBranchAddress("isB_ghost",&isB_ghost);
   t1->SetBranchAddress("isUndefined_ghost",&isUndefined_ghost);
   t1->SetBranchAddress("isU_Z",&isU_Z);
   t1->SetBranchAddress("isD_Z",&isD_Z);
   t1->SetBranchAddress("isS_Z",&isS_Z);
   t1->SetBranchAddress("isC_Z",&isC_Z);
   t1->SetBranchAddress("isB_Z",&isB_Z);
   t1->SetBranchAddress("isUndefined_Z",&isUndefined_Z);
   t1->SetBranchAddress("isU_ghost_angle03status70",&isU_ghost_angle03status70);
   t1->SetBranchAddress("isD_ghost_angle03status70",&isD_ghost_angle03status70);
   t1->SetBranchAddress("isS_ghost_angle03status70",&isS_ghost_angle03status70);
   t1->SetBranchAddress("isC_ghost_angle03status70",&isC_ghost_angle03status70);
   t1->SetBranchAddress("isB_ghost_angle03status70",&isB_ghost_angle03status70);
   t1->SetBranchAddress("isUndefined_ghost_angle03status70",&isUndefined_ghost_angle03status70);
   t1->SetBranchAddress("isU_ghost_angle03status20",&isU_ghost_angle03status20);
   t1->SetBranchAddress("isD_ghost_angle03status20",&isD_ghost_angle03status20);
   t1->SetBranchAddress("isS_ghost_angle03status20",&isS_ghost_angle03status20);
   t1->SetBranchAddress("isC_ghost_angle03status20",&isC_ghost_angle03status20);
   t1->SetBranchAddress("isB_ghost_angle03status20",&isB_ghost_angle03status20);
   t1->SetBranchAddress("isUndefined_ghost_angle03status20",&isUndefined_ghost_angle03status20);
   //t1->SetBranchAddress("Z_flavour",&Z_flavour);

   t1->SetBranchAddress("jets_p",&jets_p);
   t1->SetBranchAddress("jets_px",&jets_px);
   t1->SetBranchAddress("jets_py",&jets_py);
   t1->SetBranchAddress("jets_pz",&jets_pz);
   t1->SetBranchAddress("jets_theta",&jets_theta);
   t1->SetBranchAddress("jets_phi",&jets_phi);
   t1->SetBranchAddress("jets_m",&jets_m);
   t1->SetBranchAddress("jets_e",&jets_e);
   t1->SetBranchAddress("jets_nRP_charged",&jets_nRP_charged);
   t1->SetBranchAddress("jets_nRP_neutral",&jets_nRP_neutral);
   
   t1->SetBranchAddress("jets_pt",&jets_pt);
   t1->SetBranchAddress("jets_eta",&jets_eta);

   t1->SetBranchAddress("RPj_charged_p",&RPj_charged_p);
   t1->SetBranchAddress("RPj_charged_theta",&RPj_charged_theta);
   t1->SetBranchAddress("RPj_charged_phi",&RPj_charged_phi);
   t1->SetBranchAddress("RPj_charged_mass",&RPj_charged_mass);
   t1->SetBranchAddress("RPj_charged_Z0",&RPj_charged_Z0);
   t1->SetBranchAddress("RPj_charged_D0",&RPj_charged_D0);
   t1->SetBranchAddress("RPj_charged_Z0_sig",&RPj_charged_Z0_sig);
   t1->SetBranchAddress("RPj_charged_D0_sig",&RPj_charged_D0_sig);
   t1->SetBranchAddress("RPj_charged_dTheta",&RPj_charged_dTheta);
   t1->SetBranchAddress("RPj_charged_dPhi",&RPj_charged_dPhi);
   t1->SetBranchAddress("RPj_charged_pRel",&RPj_charged_pRel);
   t1->SetBranchAddress("RPj_charged_isMuon",&RPj_charged_isMuon);
   t1->SetBranchAddress("RPj_charged_isElectron",&RPj_charged_isElectron);

   t1->SetBranchAddress("RPj_charged_is_S",&RPj_charged_is_S);
   t1->SetBranchAddress("RPj_charged_is_Kaon",&RPj_charged_is_Kaon);
   t1->SetBranchAddress("RPj_charged_is_Kaon_smearedUniform080",&RPj_charged_is_Kaon_smearedUniform080);
   t1->SetBranchAddress("RPj_charged_is_Kaon_smearedUniform090",&RPj_charged_is_Kaon_smearedUniform090);
   t1->SetBranchAddress("RPj_charged_is_Kaon_smearedUniform095",&RPj_charged_is_Kaon_smearedUniform095);
   t1->SetBranchAddress("RPj_charged_is_Kaon_smearedUniform060",&RPj_charged_is_Kaon_smearedUniform060);
   t1->SetBranchAddress("RPj_charged_is_Kaon_smearedUniform040",&RPj_charged_is_Kaon_smearedUniform040);
   t1->SetBranchAddress("RPj_charged_is_Kaon_smearedUniform020",&RPj_charged_is_Kaon_smearedUniform020);
   t1->SetBranchAddress("RPj_charged_is_Kaon_smearedUniform000",&RPj_charged_is_Kaon_smearedUniform000);
   t1->SetBranchAddress("RPj_charged_is_Kaon_smearedUniform100",&RPj_charged_is_Kaon_smearedUniform100);
   
   
   t1->SetBranchAddress("RPj_neutral_p",&RPj_neutral_p);
   t1->SetBranchAddress("RPj_neutral_pRel",&RPj_neutral_pRel);
   t1->SetBranchAddress("RPj_neutral_isPhoton",&RPj_neutral_isPhoton);

   t1->SetBranchAddress("sv_mass",&sv_mass);
   t1->SetBranchAddress("sv_p",&sv_p);
   t1->SetBranchAddress("sv_ntracks",&sv_ntracks);
   t1->SetBranchAddress("sv_chi2",&sv_chi2);
   t1->SetBranchAddress("sv_normchi2",&sv_normchi2);
   t1->SetBranchAddress("sv_ndf",&sv_ndf);
   t1->SetBranchAddress("sv_theta",&sv_theta);
   t1->SetBranchAddress("sv_phi",&sv_phi);
   t1->SetBranchAddress("sv_thetarel",&sv_thetarel);
   t1->SetBranchAddress("sv_phirel",&sv_phirel);
   t1->SetBranchAddress("sv_costhetasvpv",&sv_costhetasvpv);
   t1->SetBranchAddress("sv_dxy",&sv_dxy);
   t1->SetBranchAddress("sv_d3d",&sv_d3d);
  

   t1->SetBranchAddress("v0_pid",&v0_pid);
   t1->SetBranchAddress("v0_mass",&v0_mass);
   t1->SetBranchAddress("v0_p",&v0_p);
   t1->SetBranchAddress("v0_ntracks",&v0_ntracks);
   t1->SetBranchAddress("v0_chi2",&v0_chi2);
   t1->SetBranchAddress("v0_normchi2",&v0_normchi2);
   t1->SetBranchAddress("v0_ndf",&v0_ndf);
   t1->SetBranchAddress("v0_theta",&v0_theta);
   t1->SetBranchAddress("v0_phi",&v0_phi);
   /////////////t1->SetBranchAddress("v0_thetarel",&v0_thetarel);
   /////////////t1->SetBranchAddress("v0_phirel",&v0_phirel);
   t1->SetBranchAddress("v0_costhetasvpv",&v0_costhetasvpv);
   t1->SetBranchAddress("v0_dxy",&v0_dxy);
   t1->SetBranchAddress("v0_d3d",&v0_d3d);
   
   //Not sure why but this has to stay here or the code fails...
   t1->GetEntry(10);
   
   
   
   fname.resize(fname.size()-5);
   //fname=fname+"_"+flav_type+".root";//+"_uds.root";
   fname=fname+".root";//+"_uds.root";
   std::string fname_buffer;
   for(auto& i : fname){
     if(i=='/') {fname_buffer = {}; continue;}
     fname_buffer.push_back(i);
   }
   // SOME NASTY EOS HARDCODING
   pwd = "/eos/user/e/eploerer/DeepJet_sourceFiles/tmp/";
   if (argc==3) pwd = argv[2];
   fname_buffer = pwd+fname_buffer;
   //TFile newfile("ntuple_short.root", "recreate");
   TFile *ntuple = new TFile(fname_buffer.c_str(),"recreate");
   TDirectory *deepntuplizer = ntuple->mkdir("deepntuplizer");
   deepntuplizer->cd();    // make the "tof" directory the current directory
  
   
   Int_t lenVar;

   Int_t event_index, jet_index;
   
   Int_t isU_;
   Int_t isD_;
   Int_t isS_;
   Int_t isC_;
   Int_t isB_;
   Int_t isUndefined_;
   Int_t isU_ghost_;
   Int_t isD_ghost_;
   Int_t isS_ghost_;
   Int_t isC_ghost_;
   Int_t isB_ghost_;
   Int_t isUndefined_ghost_;
   Int_t isU_Z_;
   Int_t isD_Z_;
   Int_t isS_Z_;
   Int_t isC_Z_;
   Int_t isB_Z_;
   Int_t isUndefined_Z_;
   Int_t isU_ghost_angle03status70_;
   Int_t isD_ghost_angle03status70_;
   Int_t isS_ghost_angle03status70_;
   Int_t isC_ghost_angle03status70_;
   Int_t isB_ghost_angle03status70_;
   Int_t isUndefined_ghost_angle03status70_;
   Int_t isU_ghost_angle03status20_;
   Int_t isD_ghost_angle03status20_;
   Int_t isS_ghost_angle03status20_;
   Int_t isC_ghost_angle03status20_;
   Int_t isB_ghost_angle03status20_;
   Int_t isUndefined_ghost_angle03status20_;
   //Int_t Z_flavour_;

   Float_t jets_p_;
   Float_t jets_px_;
   Float_t jets_py_;
   Float_t jets_pz_;
   Float_t jets_theta_;
   Float_t jets_phi_;
   Float_t jets_m_;
   Float_t jets_e_;
   Float_t jets_nRP_charged_;
   Float_t jets_nRP_neutral_;

   Int_t nCRP;
   Int_t nNRP;
   Int_t nSV;
   Int_t nV0;

   Float_t jets_pt_;
   Float_t jets_eta_;

   //Note that the below version was me trying to convert vectors to arrays cleverly. I gave up pretty quickly because the code is already fast...

   //std::vector<float> RPj_charged_p_;
   Float_t RPj_charged_p_[100];
   Float_t RPj_charged_theta_[100];
   Float_t RPj_charged_phi_[100];
   Float_t RPj_charged_mass_[100];
   Float_t RPj_charged_Z0_[100];
   Float_t RPj_charged_D0_[100];
   Float_t RPj_charged_Z0_sig_[100];
   Float_t RPj_charged_D0_sig_[100];
   Float_t RPj_charged_dTheta_[100];
   Float_t RPj_charged_dPhi_[100];
   Float_t RPj_charged_pRel_[100];
   Float_t RPj_charged_isMuon_[100];
   Float_t RPj_charged_isElectron_[100];
   
   Float_t RPj_charged_is_S_[100];
   Float_t RPj_charged_is_Kaon_[100];
   Float_t RPj_charged_is_Kaon_smearedUniform080_[100];
   Float_t RPj_charged_is_Kaon_smearedUniform090_[100];
   Float_t RPj_charged_is_Kaon_smearedUniform095_[100];
   Float_t RPj_charged_is_Kaon_smearedUniform060_[100];
   Float_t RPj_charged_is_Kaon_smearedUniform040_[100];
   Float_t RPj_charged_is_Kaon_smearedUniform020_[100];
   Float_t RPj_charged_is_Kaon_smearedUniform000_[100];
   Float_t RPj_charged_is_Kaon_smearedUniform100_[100];

   Float_t RPj_neutral_p_[100];
   Float_t RPj_neutral_pRel_[100];
   Float_t RPj_neutral_isPhoton_[100];

   Float_t sv_mass_[100];
   Float_t sv_p_[100];
   Int_t sv_ntracks_[100];
   Float_t sv_chi2_[100];
   Float_t sv_normchi2_[100];
   Int_t sv_ndf_[100];
   Float_t sv_theta_[100];
   Float_t sv_phi_[100];
   Float_t sv_thetarel_[100];
   Float_t sv_phirel_[100];
   Float_t sv_costhetasvpv_[100];
   Float_t sv_dxy_[100];
   Float_t sv_d3d_[100];

   Float_t v0_pid_[100];
   Float_t v0_mass_[100];
   Float_t v0_p_[100];
   Int_t v0_ntracks_[100];
   Float_t v0_chi2_[100];
   Float_t v0_normchi2_[100];
   Int_t v0_ndf_[100];
   Float_t v0_theta_[100];
   Float_t v0_phi_[100];
   Float_t v0_thetarel_[100];
   Float_t v0_phirel_[100];
   Float_t v0_costhetasvpv_[100];
   Float_t v0_dxy_[100];
   Float_t v0_d3d_[100];
   
   TTree *newtree = new TTree("tree", "flattened tree of source");
   newtree->Branch("event_index", &event_index);//, "std::vector<float>");
   newtree->Branch("jet_index", &jet_index);//, "std::vector<float>");
   
   newtree->Branch("isU", &isU_);
   newtree->Branch("isD", &isD_);
   newtree->Branch("isS", &isS_);
   newtree->Branch("isC", &isC_);
   newtree->Branch("isB", &isB_);
   newtree->Branch("isUndefined", &isUndefined_);
   newtree->Branch("isU_ghost", &isU_ghost_);
   newtree->Branch("isD_ghost", &isD_ghost_);
   newtree->Branch("isS_ghost", &isS_ghost_);
   newtree->Branch("isC_ghost", &isC_ghost_);
   newtree->Branch("isB_ghost", &isB_ghost_);
   newtree->Branch("isUndefined_ghost", &isUndefined_ghost_);
   newtree->Branch("isU_Z", &isU_Z_);
   newtree->Branch("isD_Z", &isD_Z_);
   newtree->Branch("isS_Z", &isS_Z_);
   newtree->Branch("isC_Z", &isC_Z_);
   newtree->Branch("isB_Z", &isB_Z_);
   newtree->Branch("isUndefined_Z", &isUndefined_Z_);
   newtree->Branch("isU_ghost_angle03status70", &isU_ghost_angle03status70_);
   newtree->Branch("isD_ghost_angle03status70", &isD_ghost_angle03status70_);
   newtree->Branch("isS_ghost_angle03status70", &isS_ghost_angle03status70_);
   newtree->Branch("isC_ghost_angle03status70", &isC_ghost_angle03status70_);
   newtree->Branch("isB_ghost_angle03status70", &isB_ghost_angle03status70_);
   newtree->Branch("isUndefined_ghost_angle03status70", &isUndefined_ghost_angle03status70_);
   newtree->Branch("isU_ghost_angle03status20", &isU_ghost_angle03status20_);
   newtree->Branch("isD_ghost_angle03status20", &isD_ghost_angle03status20_);
   newtree->Branch("isS_ghost_angle03status20", &isS_ghost_angle03status20_);
   newtree->Branch("isC_ghost_angle03status20", &isC_ghost_angle03status20_);
   newtree->Branch("isB_ghost_angle03status20", &isB_ghost_angle03status20_);
   newtree->Branch("isUndefined_ghost_angle03status20", &isUndefined_ghost_angle03status20_);
   //newtree->Branch("Z_flavour", &Z_flavour_);

   newtree->Branch("jets_p", &jets_p_);
   newtree->Branch("jets_px", &jets_px_);
   newtree->Branch("jets_py", &jets_py_);
   newtree->Branch("jets_pz", &jets_pz_);
   newtree->Branch("jets_theta", &jets_theta_);
   newtree->Branch("jets_phi", &jets_phi_);
   newtree->Branch("jets_m", &jets_m_);
   newtree->Branch("jets_e", &jets_e_);
   newtree->Branch("jets_nRP_charged", &jets_nRP_charged_);
   newtree->Branch("jets_nRP_neutral", &jets_nRP_neutral_);
   
   newtree->Branch("nCRP", &nCRP, "nCRP/I");
   newtree->Branch("nNRP", &nNRP, "nNRP/I");
   newtree->Branch("nSV", &nSV, "nSV/I");
   newtree->Branch("nV0", &nV0, "nV0/I");
   
   newtree->Branch("jets_pt", &jets_pt_);
   newtree->Branch("jets_eta", &jets_eta_);

   newtree->Branch("RPj_charged_p", &RPj_charged_p_, "RPj_charged_p[nCRP]/F");
   newtree->Branch("RPj_charged_theta", &RPj_charged_theta_, "RPj_charged_theta[nCRP]/F");
   newtree->Branch("RPj_charged_phi", &RPj_charged_phi_, "RPj_charged_phi[nCRP]/F");
   newtree->Branch("RPj_charged_mass", &RPj_charged_mass_, "RPj_charged_mass[nCRP]/F");
   newtree->Branch("RPj_charged_Z0", &RPj_charged_Z0_, "RPj_charged_Z0[nCRP]/F");
   newtree->Branch("RPj_charged_D0", &RPj_charged_D0_, "RPj_charged_D0[nCRP]/F");
   newtree->Branch("RPj_charged_Z0_sig", &RPj_charged_Z0_sig_, "RPj_charged_Z0_sig[nCRP]/F");
   newtree->Branch("RPj_charged_D0_sig", &RPj_charged_D0_sig_, "RPj_charged_D0_sig[nCRP]/F");
   newtree->Branch("RPj_charged_dTheta", &RPj_charged_dTheta_, "RPj_charged_dTheta[nCRP]/F");
   newtree->Branch("RPj_charged_dPhi", &RPj_charged_dPhi_, "RPj_charged_dPhi[nCRP]/F");
   newtree->Branch("RPj_charged_pRel", &RPj_charged_pRel_, "RPj_charged_pRel[nCRP]/F");
   newtree->Branch("RPj_charged_isMuon", &RPj_charged_isMuon_, "RPj_charged_isMuon[nCRP]/F");
   newtree->Branch("RPj_charged_isElectron", &RPj_charged_isElectron_, "RPj_charged_isElectron[nCRP]/F");
   
   newtree->Branch("RPj_charged_is_S", &RPj_charged_is_S_, "RPj_charged_is_S[nCRP]/F");
   newtree->Branch("RPj_charged_is_Kaon", &RPj_charged_is_Kaon_, "RPj_charged_is_Kaon[nCRP]/F");
   newtree->Branch("RPj_charged_is_Kaon_smearedUniform080", &RPj_charged_is_Kaon_smearedUniform080_, "RPj_charged_is_Kaon_smearedUniform080[nCRP]/F");
   newtree->Branch("RPj_charged_is_Kaon_smearedUniform090", &RPj_charged_is_Kaon_smearedUniform090_, "RPj_charged_is_Kaon_smearedUniform090[nCRP]/F");
   newtree->Branch("RPj_charged_is_Kaon_smearedUniform095", &RPj_charged_is_Kaon_smearedUniform095_, "RPj_charged_is_Kaon_smearedUniform095[nCRP]/F");
   newtree->Branch("RPj_charged_is_Kaon_smearedUniform060", &RPj_charged_is_Kaon_smearedUniform060_, "RPj_charged_is_Kaon_smearedUniform060[nCRP]/F");
   newtree->Branch("RPj_charged_is_Kaon_smearedUniform040", &RPj_charged_is_Kaon_smearedUniform040_, "RPj_charged_is_Kaon_smearedUniform040[nCRP]/F");
   newtree->Branch("RPj_charged_is_Kaon_smearedUniform020", &RPj_charged_is_Kaon_smearedUniform020_, "RPj_charged_is_Kaon_smearedUniform020[nCRP]/F");
   newtree->Branch("RPj_charged_is_Kaon_smearedUniform000", &RPj_charged_is_Kaon_smearedUniform000_, "RPj_charged_is_Kaon_smearedUniform000[nCRP]/F");
   newtree->Branch("RPj_charged_is_Kaon_smearedUniform100", &RPj_charged_is_Kaon_smearedUniform100_, "RPj_charged_is_Kaon_smearedUniform100[nCRP]/F");

   newtree->Branch("RPj_neutral_p", &RPj_neutral_p_, "RPj_neutral_p[nNRP]/F");
   newtree->Branch("RPj_neutral_pRel", &RPj_neutral_pRel_, "RPj_neutral_pRel[nNRP]/F");
   newtree->Branch("RPj_neutral_isPhoton", &RPj_neutral_isPhoton_, "RPj_neutral_isPhoton[nNRP]/F");

   newtree->Branch("sv_mass", &sv_mass_, "sv_mass[nSV]/F");
   newtree->Branch("sv_p", &sv_p_, "sv_p[nSV]/F");
   newtree->Branch("sv_ntracks", &sv_ntracks_, "sv_ntracks[nSV]/F");
   newtree->Branch("sv_chi2", &sv_chi2_, "sv_chi2[nSV]/F");
   newtree->Branch("sv_normchi2", &sv_normchi2_, "sv_normchi2[nSV]/F");
   newtree->Branch("sv_ndf", &sv_ndf_, "sv_ndf[nSV]/F");
   newtree->Branch("sv_theta", &sv_theta_, "sv_theta[nSV]/F");
   newtree->Branch("sv_phi", &sv_phi_, "sv_phi[nSV]/F");
   newtree->Branch("sv_thetarel", &sv_thetarel_, "sv_thetarel[nSV]/F");
   newtree->Branch("sv_phirel", &sv_phirel_, "sv_phirel[nSV]/F");
   newtree->Branch("sv_costhetasvpv", &sv_costhetasvpv_, "sv_costhetasvpv[nSV]/F");
   newtree->Branch("sv_dxy", &sv_dxy_, "sv_dxy[nSV]/F");
   newtree->Branch("sv_d3d", &sv_d3d_, "sv_d3d[nSV]/F");

   newtree->Branch("v0_pid", &v0_pid_, "v0_pid[nV0]/F");
   newtree->Branch("v0_mass", &v0_mass_, "v0_mass[nV0]/F");
   newtree->Branch("v0_p", &v0_p_, "v0_p[nV0]/F");
   newtree->Branch("v0_ntracks", &v0_ntracks_, "v0_ntracks[nV0]/F");
   newtree->Branch("v0_chi2", &v0_chi2_, "v0_chi2[nV0]/F");
   newtree->Branch("v0_normchi2", &v0_normchi2_, "v0_normchi2[nV0]/F");
   newtree->Branch("v0_ndf", &v0_ndf_, "v0_ndf[nV0]/F");
   newtree->Branch("v0_theta", &v0_theta_, "v0_theta[nV0]/F");
   newtree->Branch("v0_phi", &v0_phi_, "v0_phi[nV0]/F");
   //////////////////newtree->Branch("v0_thetarel", &v0_thetarel_, "v0_thetarel[nV0]/F");
   //////////////////newtree->Branch("v0_phirel", &v0_phirel_, "v0_phirel[nV0]/F");
   newtree->Branch("v0_costhetasvpv", &v0_costhetasvpv_, "v0_costhetasvpv[nV0]/F");
   newtree->Branch("v0_dxy", &v0_dxy_, "v0_dxy[nV0]/F");
   newtree->Branch("v0_d3d", &v0_d3d_, "v0_d3d[nV0]/F");
   
   for(int i=0; i<nentries; ++i){
     t1->GetEntry(i);
     if((*jets_p).size()!=2) std::cout<<"ANOMALY : nJets="<<((*jets_p).size())<<std::endl;
     for(int j=0; j<(*jets_p).size(); ++j){
     //for(int j=0; j<2; ++j){
   
       
       event_index = i;
       jet_index = j;
       //std::cout<<" "<<" with size "<<(*isU).size()<<std::endl;
        
       isU_ = (*isU)[j];
       isD_ = (*isD)[j];
       isS_ = (*isS)[j];
       isC_ = (*isC)[j];
       isB_ = (*isB)[j];
       isUndefined_ = (*isUndefined)[j];
       isU_ghost_ = (*isU_ghost)[j];
       isD_ghost_ = (*isD_ghost)[j];
       isS_ghost_ = (*isS_ghost)[j];
       isC_ghost_ = (*isC_ghost)[j];
       isB_ghost_ = (*isB_ghost)[j];
       isUndefined_ghost_ = (*isUndefined_ghost)[j];
       isU_Z_ = (*isU_Z)[j];
       isD_Z_ = (*isD_Z)[j];
       isS_Z_ = (*isS_Z)[j];
       isC_Z_ = (*isC_Z)[j];
       isB_Z_ = (*isB_Z)[j];
       isUndefined_Z_ = (*isUndefined_Z)[j];
       isU_ghost_angle03status70_ = (*isU_ghost_angle03status70)[j];
       isD_ghost_angle03status70_ = (*isD_ghost_angle03status70)[j];
       isS_ghost_angle03status70_ = (*isS_ghost_angle03status70)[j];
       isC_ghost_angle03status70_ = (*isC_ghost_angle03status70)[j];
       isB_ghost_angle03status70_ = (*isB_ghost_angle03status70)[j];
       isUndefined_ghost_angle03status70_ = (*isUndefined_ghost_angle03status70)[j];
       isU_ghost_angle03status20_ = (*isU_ghost_angle03status20)[j];
       isD_ghost_angle03status20_ = (*isD_ghost_angle03status20)[j];
       isS_ghost_angle03status20_ = (*isS_ghost_angle03status20)[j];
       isC_ghost_angle03status20_ = (*isC_ghost_angle03status20)[j];
       isB_ghost_angle03status20_ = (*isB_ghost_angle03status20)[j];
       isUndefined_ghost_angle03status20_ = (*isUndefined_ghost_angle03status20)[j];
       //Z_flavour_ = (*Z_flavour)[j];

       jets_p_ = (*jets_p)[j];
       jets_px_ = (*jets_px)[j];
       jets_py_ = (*jets_py)[j];
       jets_pz_ = (*jets_pz)[j];
       jets_theta_ = (*jets_theta)[j];
       jets_phi_ = (*jets_phi)[j];
       jets_m_ = (*jets_m)[j];
       jets_e_ = (*jets_e)[j];
       jets_nRP_charged_ = (*jets_nRP_charged)[j];
       
       nCRP = (*jets_nRP_charged)[j];
       nNRP = (*jets_nRP_neutral)[j];
       nSV = (*sv_p)[j].size();
       nV0 = (*v0_p)[j].size();
       //std::cout<<"nSV, nV0 = "<<nSV<<", "<<nV0<<std::endl;
      
       jets_nRP_neutral_ = (*jets_nRP_neutral)[j];
       jets_pt_ = (*jets_pt)[j];
       jets_eta_ = (*jets_eta)[j];

       std::copy((*RPj_charged_p)[j].begin(), (*RPj_charged_p)[j].end(), RPj_charged_p_);//&((*RPj_charged_p)[j])[j]
       std::copy((*RPj_charged_theta)[j].begin(), (*RPj_charged_theta)[j].end(), RPj_charged_theta_);
       std::copy((*RPj_charged_phi)[j].begin(), (*RPj_charged_phi)[j].end(), RPj_charged_phi_);
       std::copy((*RPj_charged_mass)[j].begin(), (*RPj_charged_mass)[j].end(), RPj_charged_mass_);
       std::copy((*RPj_charged_Z0)[j].begin(), (*RPj_charged_Z0)[j].end(), RPj_charged_Z0_);
       std::copy((*RPj_charged_D0)[j].begin(), (*RPj_charged_D0)[j].end(), RPj_charged_D0_);
       std::copy((*RPj_charged_Z0_sig)[j].begin(), (*RPj_charged_Z0_sig)[j].end(), RPj_charged_Z0_sig_);
       std::copy((*RPj_charged_D0_sig)[j].begin(), (*RPj_charged_D0_sig)[j].end(), RPj_charged_D0_sig_);
       std::copy((*RPj_charged_dTheta)[j].begin(), (*RPj_charged_dTheta)[j].end(), RPj_charged_dTheta_);
       std::copy((*RPj_charged_dPhi)[j].begin(), (*RPj_charged_dPhi)[j].end(), RPj_charged_dPhi_);
       std::copy((*RPj_charged_pRel)[j].begin(), (*RPj_charged_pRel)[j].end(), RPj_charged_pRel_);
       std::copy((*RPj_charged_isMuon)[j].begin(), (*RPj_charged_isMuon)[j].end(), RPj_charged_isMuon_);
       std::copy((*RPj_charged_isElectron)[j].begin(), (*RPj_charged_isElectron)[j].end(), RPj_charged_isElectron_);
      
       std::copy((*RPj_charged_is_S)[j].begin(), (*RPj_charged_is_S)[j].end(), RPj_charged_is_S_);
       std::copy((*RPj_charged_is_Kaon)[j].begin(), (*RPj_charged_is_Kaon)[j].end(), RPj_charged_is_Kaon_);
       std::copy((*RPj_charged_is_Kaon_smearedUniform080)[j].begin(), (*RPj_charged_is_Kaon_smearedUniform080)[j].end(), RPj_charged_is_Kaon_smearedUniform080_);
       std::copy((*RPj_charged_is_Kaon_smearedUniform090)[j].begin(), (*RPj_charged_is_Kaon_smearedUniform090)[j].end(), RPj_charged_is_Kaon_smearedUniform090_);
       std::copy((*RPj_charged_is_Kaon_smearedUniform095)[j].begin(), (*RPj_charged_is_Kaon_smearedUniform095)[j].end(), RPj_charged_is_Kaon_smearedUniform095_);
       std::copy((*RPj_charged_is_Kaon_smearedUniform060)[j].begin(), (*RPj_charged_is_Kaon_smearedUniform060)[j].end(), RPj_charged_is_Kaon_smearedUniform060_);
       std::copy((*RPj_charged_is_Kaon_smearedUniform040)[j].begin(), (*RPj_charged_is_Kaon_smearedUniform040)[j].end(), RPj_charged_is_Kaon_smearedUniform040_);
       std::copy((*RPj_charged_is_Kaon_smearedUniform020)[j].begin(), (*RPj_charged_is_Kaon_smearedUniform020)[j].end(), RPj_charged_is_Kaon_smearedUniform020_);
       std::copy((*RPj_charged_is_Kaon_smearedUniform000)[j].begin(), (*RPj_charged_is_Kaon_smearedUniform000)[j].end(), RPj_charged_is_Kaon_smearedUniform000_);
       std::copy((*RPj_charged_is_Kaon_smearedUniform100)[j].begin(), (*RPj_charged_is_Kaon_smearedUniform100)[j].end(), RPj_charged_is_Kaon_smearedUniform100_);

       std::copy((*RPj_neutral_p)[j].begin(), (*RPj_neutral_p)[j].end(), RPj_neutral_p_);//&((*RPj_neutral_p)[j])[j]
       std::copy((*RPj_neutral_pRel)[j].begin(), (*RPj_neutral_pRel)[j].end(), RPj_neutral_pRel_);
       std::copy((*RPj_neutral_isPhoton)[j].begin(), (*RPj_neutral_isPhoton)[j].end(), RPj_neutral_isPhoton_);

       std::copy((*sv_mass)[j].begin(), (*sv_mass)[j].end(), sv_mass_);
       std::copy((*sv_p)[j].begin(), (*sv_p)[j].end(), sv_p_);
       std::copy((*sv_ntracks)[j].begin(), (*sv_ntracks)[j].end(), sv_ntracks_);
       std::copy((*sv_chi2)[j].begin(), (*sv_chi2)[j].end(), sv_chi2_);
       std::copy((*sv_normchi2)[j].begin(), (*sv_normchi2)[j].end(), sv_normchi2_);
       std::copy((*sv_ndf)[j].begin(), (*sv_ndf)[j].end(), sv_ndf_);
       std::copy((*sv_theta)[j].begin(), (*sv_theta)[j].end(), sv_theta_);
       std::copy((*sv_phi)[j].begin(), (*sv_phi)[j].end(), sv_phi_);
       std::copy((*sv_thetarel)[j].begin(), (*sv_thetarel)[j].end(), sv_thetarel_);
       std::copy((*sv_phirel)[j].begin(), (*sv_phirel)[j].end(), sv_phirel_);
       std::copy((*sv_costhetasvpv)[j].begin(), (*sv_costhetasvpv)[j].end(), sv_costhetasvpv_);
       std::copy((*sv_dxy)[j].begin(), (*sv_dxy)[j].end(), sv_dxy_);
       std::copy((*sv_d3d)[j].begin(), (*sv_d3d)[j].end(), sv_d3d_);

       std::copy((*v0_pid)[j].begin(), (*v0_pid)[j].end(), v0_pid_);
       std::copy((*v0_mass)[j].begin(), (*v0_mass)[j].end(), v0_mass_);
       std::copy((*v0_p)[j].begin(), (*v0_p)[j].end(), v0_p_);
       std::copy((*v0_ntracks)[j].begin(), (*v0_ntracks)[j].end(), v0_ntracks_);
       std::copy((*v0_chi2)[j].begin(), (*v0_chi2)[j].end(), v0_chi2_);
       std::copy((*v0_normchi2)[j].begin(), (*v0_normchi2)[j].end(), v0_normchi2_);
       std::copy((*v0_ndf)[j].begin(), (*v0_ndf)[j].end(), v0_ndf_);
       std::copy((*v0_theta)[j].begin(), (*v0_theta)[j].end(), v0_theta_);
       std::copy((*v0_phi)[j].begin(), (*v0_phi)[j].end(), v0_phi_);
       ///////////std::copy((*v0_thetarel)[j].begin(), (*v0_thetarel)[j].end(), v0_thetarel_);
       ///////////std::copy((*v0_phirel)[j].begin(), (*v0_phirel)[j].end(), v0_phirel_);
       std::copy((*v0_costhetasvpv)[j].begin(), (*v0_costhetasvpv)[j].end(), v0_costhetasvpv_);
       std::copy((*v0_dxy)[j].begin(), (*v0_dxy)[j].end(), v0_dxy_);
       std::copy((*v0_d3d)[j].begin(), (*v0_d3d)[j].end(), v0_d3d_);
       newtree->Fill();
     }
   }
   //newtree->Scan("RPj_charged_p");
   ntuple->Write();
   
   delete f;
   delete ntuple;

   //Deleting pointers to heap vars
   
   delete isU;
   delete isD;
   delete isS;
   delete isC;
   delete isB;
   delete isUndefined;
   delete isU_ghost;
   delete isD_ghost;
   delete isS_ghost;
   delete isC_ghost;
   delete isB_ghost;
   delete isUndefined_ghost;
   delete isU_Z;
   delete isD_Z;
   delete isS_Z;
   delete isC_Z;
   delete isB_Z;
   delete isUndefined_Z;
   delete isU_ghost_angle03status70;
   delete isD_ghost_angle03status70;
   delete isS_ghost_angle03status70;
   delete isC_ghost_angle03status70;
   delete isB_ghost_angle03status70;
   delete isUndefined_ghost_angle03status70;
   delete isU_ghost_angle03status20;
   delete isD_ghost_angle03status20;
   delete isS_ghost_angle03status20;
   delete isC_ghost_angle03status20;
   delete isB_ghost_angle03status20;
   delete isUndefined_ghost_angle03status20;
   //delete Z_flavour;

   delete jets_p;
   delete jets_px;
   delete jets_py;
   delete jets_pz;
   delete jets_theta;
   delete jets_phi;
   delete jets_m;
   delete jets_e;
   delete jets_nRP_charged;
   delete jets_nRP_neutral;

   delete jets_pt;
   delete jets_eta;

   delete RPj_charged_p;
   delete RPj_charged_theta;
   delete RPj_charged_phi;
   delete RPj_charged_mass;
   delete RPj_charged_Z0;
   delete RPj_charged_D0;
   delete RPj_charged_Z0_sig;
   delete RPj_charged_D0_sig;
   delete RPj_charged_dTheta;
   delete RPj_charged_dPhi;
   delete RPj_charged_pRel;
   delete RPj_charged_isMuon;
   delete RPj_charged_isElectron;
   
   delete RPj_charged_is_S;
   delete RPj_charged_is_Kaon;
   delete RPj_charged_is_Kaon_smearedUniform080;
   delete RPj_charged_is_Kaon_smearedUniform090;
   delete RPj_charged_is_Kaon_smearedUniform095;
   delete RPj_charged_is_Kaon_smearedUniform060;
   delete RPj_charged_is_Kaon_smearedUniform040;
   delete RPj_charged_is_Kaon_smearedUniform020;
   delete RPj_charged_is_Kaon_smearedUniform000;
   delete RPj_charged_is_Kaon_smearedUniform100;

   delete RPj_neutral_p;
   delete RPj_neutral_pRel;
   delete RPj_neutral_isPhoton;

   delete sv_mass;
   delete sv_p;
   delete sv_ntracks;
   delete sv_chi2;
   delete sv_normchi2;
   delete sv_ndf;
   delete sv_theta;
   delete sv_phi;
   delete sv_thetarel;
   delete sv_phirel;
   delete sv_costhetasvpv;
   delete sv_dxy;
   delete sv_d3d;

   delete v0_pid;
   delete v0_mass;
   delete v0_p;
   delete v0_ntracks;
   delete v0_chi2;
   delete v0_normchi2;
   delete v0_ndf;
   delete v0_theta;
   delete v0_phi;
   ///////////delete v0_thetarel;
   ///////////delete v0_phirel;
   delete v0_costhetasvpv;
   delete v0_dxy;
   delete v0_d3d;
   
   return 0;
}
