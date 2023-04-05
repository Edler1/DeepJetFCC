
from DeepJetCore.TrainData import TrainData, fileTimeOut
import numpy as np
import uproot3 as u3
import uproot as u
import awkward as ak
import pandas as pd

import os





print("Running the updated version...")


GLOBAL_PREFIX = ""

def uproot_root2array(fname, treename, stop=None, branches=None):
    dtypes = np.dtype( [(b, np.dtype("O")) for b in branches] )
    if isinstance(fname, list):
        fname = fname[0]
    tree = u3.open(fname)[treename]

    print ("0",branches[0], fname)

    new_arr = np.empty( len(tree[branches[0]].array()), dtype=dtypes)

    for branch in branches:
        print (branch)
        new_arr[branch] = np.array( ak.to_list( tree[branch].array() ), dtype="O")

    return new_arr

def uproot_tree_to_numpy(fname, inbranches_listlist, nMaxlist, nevents, treename="deepntuplizer/tree", stop=None, branches=None, flat=True):
    #open the root file, tree and get the branches list wanted for building the array
    ######tree  = u.open(fname)[treename]
    tree  = u3.open(fname)[treename]
    branches = [tree[branch_name].array() for branch_name in inbranches_listlist]
        
    #Initialize the output_array with the correct dimension and 0s everywhere. We will fill the correct 
    if nMaxlist == 1:
        output_array = np.zeros(shape=(nevents, len(inbranches_listlist)))
        
        #Loop and fill our output_array
        for i in range(nevents):
            for j, branch in enumerate(inbranches_listlist):
                output_array[i,j] = branches[j][i]
                
    if nMaxlist > 1:
        output_array = np.zeros(shape=(nevents, len(inbranches_listlist), nMaxlist))
        
        #Loop and fill w.r.t. the zero padding method our output_array
        for i in range(nevents):
            lenght = len(branches[0][i])
            for j, branch in enumerate(inbranches_listlist):
                if lenght >= nMaxlist:
                    output_array[i,j,:] = branches[j][i,:nMaxlist]
                if lenght < nMaxlist:
                    output_array[i,j,:lenght] = branches[j][i,:]
                    
        output_array = np.transpose(output_array, (0, 2, 1))
    
    
    return  output_array

def uproot_MeanNormZeroPad(Filename_in,MeanNormTuple,inbranches_listlist, nMaxslist,nevents):
    # savely copy lists (pass by value)
    import copy
    inbranches_listlist=copy.deepcopy(inbranches_listlist)
    nMaxslist=copy.deepcopy(nMaxslist)

    # Read in total number of events
    totallengthperjet = 0
    for i in range(len(nMaxslist)):
        if nMaxslist[i]>=0:
            totallengthperjet+=len(inbranches_listlist[i])*nMaxslist[i]
        else:
            totallengthperjet+=len(inbranches_listlist[i]) #flat branch

    print("Total event-length per jet: {}".format(totallengthperjet))

    #shape could be more generic here... but must be passed to c module then
    array = numpy.zeros((nevents,totallengthperjet) , dtype='float32')

    # filling mean and normlist
    normslist=[]
    meanslist=[]
    for inbranches in inbranches_listlist:
        means=[]
        norms=[]
        for b in inbranches:
            if MeanNormTuple is None:
                means.append(0)
                norms.append(1)
            else:
                means.append(MeanNormTuple[b][0])
                norms.append(MeanNormTuple[b][1])
        meanslist.append(means)
        normslist.append(norms)

    # now start filling the array


def map_prefix(elements):
    if isinstance(elements, list):
        return list(map( lambda x: GLOBAL_PREFIX + x, elements))
    elif isinstance(elements, tuple):
        return tuple(map( lambda x: GLOBAL_PREFIX + x, elements))
    elif isinstance(elements, (str)):
        return GLOBAL_PREFIX + elements
    elif isinstance(elements, bytes):
        return GLOBAL_PREFIX + elements.decode("utf-8")
    else:
        print("Error, you gave >>{}<< which is unknown".format(elements))
        raise NotImplementedError

class TrainData_ParT(TrainData):
    def __init__(self):
        

        TrainData.__init__(self)        
        
        self.description = "ParT inputs"
        
        #self.truth_branches = ['isB','isBB','isGBB','isLeptonicB','isLeptonicB_C','isC','isGCC','isCC','isUD','isS','isG']
        ##self.truth_branches = ['isB','isC','isU','isD','isS']
        self.truth_branches = ['isB_Z','isC_Z','isU_Z','isD_Z','isS_Z']
        #self.undefTruth=['isUndefined', 'isPU']
        ##self.undefTruth=['isUndefined']
        self.undefTruth=['isUndefined_Z']
        self.weightbranchX='jets_pt'
        self.weightbranchY='jets_eta'
        self.remove = False
        ###self.remove = True
        ##self.referenceclass= 'isB'  #'flatten'  #Choose 'flatten' for flat or one of the truth branch for ref
        self.referenceclass= 'isB_Z'  #'flatten'  #Choose 'flatten' for flat or one of the truth branch for ref
        #self.red_classes = ['cat_B','cat_C','cat_UDS','cat_G'] #reduced classes (flat only)
        self.red_classes = ['cat_B','cat_C','cat_U','cat_D','cat_S']
        #self.truth_red_fusion = [('isB','isBB','isGBB','isLeptonicB','isLeptonicB_C'),('isC','isGCC','isCC'),('isUD','isS'),('isG')] #Indicates how you are making the fusion of your truth branches to the reduced classes for the flat reweighting
        ##self.truth_red_fusion = [('isB'),('isC'),('isU'),('isD'),('isS')]
        self.truth_red_fusion = [('isB_Z'),('isC_Z'),('isU_Z'),('isD_Z'),('isS_Z')]
        ##self.class_weights = np.array([1.00,1.00,2.5,5.0], dtype=float)  #Ratio between our reduced classes (flat only)
        ##self.weight_binX = np.array([15, 20, 26, 35, 46, 61, 80, 106, 141, 186, 247, 326, 432, 571, 756, 1000],dtype=float) #Flat reweighting
        #self.weight_binX = np.arange(100,2001,100)
        #self.weight_binX = np.array([100,125,150,175,200,250,300,400,500,600,2000],dtype=float)
        #self.weight_binX = (np.exp(np.linspace(np.log(15), np.log(1000), 16))).tolist()
        #self.weight_binX = np.array([10,25,30,35,40,45,50,60,75,100,125,150,175,200,250,300,400,500,600,2000],dtype=float) #Ref reweighting
        ##self.weight_binY = np.array([-2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5], dtype=float) #Flat reweighting
        #self.weight_binY = np.array([-2.5,-1.136,-0.516,-0.235,-0.107,-0.0484,-0.0220, 0.0220, 0.0484, 0.107, 0.235,0.516, 1.136, 2.5], dtype=float) #Flat-hybrid reweighting
         #self.weight_binY = np.array(
         #   [-2.5, -2.0, -1.5, -1.0, -0.5, 0.5, 1.0, 1.5, 2.0, 2.5], dtype=float) #Ref reweighting


        self.class_weights = [1.00,1.00,1.00,1.00,1.00]  #Ratio between our reduced classes (flat only)
        self.weight_binX = np.array([35.0,40.0,45.0])
        #self.weight_binX = np.array([
        #    10,25,30,35,40,45,50,60,75,100,
        #    125,150,175,200,250,300,400,500,
        #    600,2000],dtype=float)

        self.weight_binY = np.array(
            [-2.5,-2.,-1.5,-1.,-0.5,0.5,1,1.5,2.,2.5],
            dtype=float
        )

        self.global_branches = ['jets_pt', 'jets_eta',
                               'jets_phi', 'jets_e',
                                #'nCpfcand','nNpfcand',
                                'jets_theta',
                                'jets_p','jets_m',
                                ]


        self.cpf_branches = ['RPj_charged_p',
                             'RPj_charged_theta',
                             'RPj_charged_phi',
                             'RPj_charged_mass',
                             #'Cpfcan_BtagPf_trackDeltaR',
                             #'Cpfcan_quality',
                             'RPj_charged_Z0',
                             'RPj_charged_D0',
                             #'RPj_charged_Z0_sig',
                             #'RPj_charged_D0_sig',
                             #'RPj_charged_dTheta',
                             #'RPj_charged_dPhi',
                             #'RPj_charged_pRel',
                             'RPj_charged_isMuon',
                             #'RPj_charged_is_S',
                             'RPj_charged_is_Kaon_smearedUniform090',
                             ]
        self.n_cpf = 25

        self.npf_branches = ['RPj_neutral_p','RPj_neutral_pRel','RPj_neutral_isPhoton']
        self.n_npf = 25

        self.vtx_branches = ['sv_mass','sv_p','sv_ntracks','sv_dxy','sv_d3d',
        ]
        self.n_vtx = 4

        self.v0_branches = ['sv_mass','sv_p','sv_ntracks','sv_dxy','sv_d3d',
        ]
        self.n_v0 = 4
        
        # These are the spectator variables relevant during inference. They have the same struct as glob vars.
        self.spectator_branches = ['event_index', 'jets_px', 'jets_py',
        ]
        
        #self.reduced_truth = ['isB','isBB','isGBB','isLeptonicB','isLeptonicB_C','isC','isGCC','isCC','isUD','isS','isG']
        ##self.reduced_truth = ['isB','isC','isU','isD','isS']
        self.reduced_truth = ['isB_Z','isC_Z','isU_Z','isD_Z','isS_Z']

        
    def createWeighterObjects(self, allsourcefiles):
        # 
        # Calculates the weights needed for flattening the pt/eta spectrum
        
        from DeepJetCore.Weighter import Weighter
        weighter = Weighter()
        weighter.undefTruth = self.undefTruth
        weighter.class_weights = self.class_weights

        branches = [self.weightbranchX,self.weightbranchY]
        branches.extend(self.truth_branches)

        if self.remove:
            weighter.setBinningAndClasses(
                [self.weight_binX,self.weight_binY],
                self.weightbranchX,self.weightbranchY,
                self.truth_branches, self.red_classes,
                self.truth_red_fusion, method = self.referenceclass
            )

        counter=0
        #import ROOT
        #from root_numpy import tree2array, root2array
        #import uproot3 as u3
        #import pandas as pd
        
        if self.remove:
            for fname in allsourcefiles:
                fileTimeOut(fname, 120)
                events = u.open(fname)["deepntuplizer/tree"]
                nparray = events.arrays(branches, library = 'np')
                keys = list(nparray.keys())
                for k in range(len(branches)):
                    nparray[branches[k]] = nparray.pop(keys[k])
                nparray = pd.Series(nparray)
#                nparray = root2array(
 #                   fname,
  #                  treename = "deepntuplizer/tree",
   #                 stop = None,
    #                branches = branches
     #           )
    
                norm_hist = True
                if self.referenceclass == 'flatten':
                    norm_hist = False
                weighter.addDistributions(nparray, norm_h = norm_hist)
                #del nparray
                counter=counter+1
            weighter.createRemoveProbabilitiesAndWeights(self.referenceclass)
            weighter.printHistos('/afs/cern.ch/user/a/ademoor/Flatten/') #If you need to print the 2D histo, choose your output dir
            return {'weigther':weighter}
    
    def convertFromSourceFile(self, filename, weighterobjects, istraining):

        # Function to produce the numpy training arrays from root files

        from DeepJetCore.Weighter import Weighter
        from DeepJetCore.stopwatch import stopwatch
        sw=stopwatch()
        swall=stopwatch()
        if not istraining:
            self.remove = False
        
        def reduceTruth(uproot_arrays):
            
            #b = uproot_arrays[b'isB']
            b = uproot_arrays[b'isB_Z']
            
            #bb = uproot_arrays[b'isBB']
            #gbb = uproot_arrays[b'isGBB']
            
            #bl = uproot_arrays[b'isLeptonicB']
            #blc = uproot_arrays[b'isLeptonicB_C']
            #lepb = bl+blc
            
            #c = uproot_arrays[b'isC']
            c = uproot_arrays[b'isC_Z']
            #cc = uproot_arrays[b'isCC']
            #gcc = uproot_arrays[b'isGCC']
            
            #u = uproot_arrays[b'isU']
            #d = uproot_arrays[b'isD']
            #s = uproot_arrays[b'isS']
            u = uproot_arrays[b'isU_Z']
            d = uproot_arrays[b'isD_Z']
            s = uproot_arrays[b'isS_Z']
            #uds = ud+s
            
            #g = uproot_arrays[b'isG']
            
            #return np.vstack((b,bb+gbb,lepb,c+cc+gcc,uds,g)).transpose()
            #return np.vstack((b+bb+gbb+lepb,c+cc+gcc,uds,g)).transpose()
            return np.vstack((b,c,s,u,d)).transpose()
        
        print('reading '+filename)
        
        import ROOT
        #from root_numpy import tree2array, root2array
        fileTimeOut(filename,120) #give eos a minute to recover
        rfile = ROOT.TFile(filename)
        tree = rfile.Get("deepntuplizer/tree")
        self.nsamples = tree.GetEntries()
        
        # user code, example works with the example 2D images in root format generated by make_example_data
        from DeepJetCore.preprocessing import MeanNormZeroPad,MeanNormZeroPadParticles


        x_global = uproot_tree_to_numpy(filename,
                                        self.global_branches,1,self.nsamples,
                                        treename='deepntuplizer/tree', flat = True)

        x_cpf = uproot_tree_to_numpy(filename,
                                     self.cpf_branches,self.n_cpf,self.nsamples,
                                     treename='deepntuplizer/tree', flat = False)

        x_npf = uproot_tree_to_numpy(filename,
                                         self.npf_branches,self.n_npf,self.nsamples,
                                     treename='deepntuplizer/tree', flat = False)

        x_vtx = uproot_tree_to_numpy(filename,
                                         self.vtx_branches,self.n_vtx,self.nsamples,
                                     treename='deepntuplizer/tree', flat = False)

        x_v0 = uproot_tree_to_numpy(filename,
                                         self.v0_branches,self.n_v0,self.nsamples,
                                     treename='deepntuplizer/tree', flat = False)
        
        x_spectator = uproot_tree_to_numpy(filename,
                                        self.spectator_branches,1,self.nsamples,
                                        treename='deepntuplizer/tree', flat = True)
        #cpf_pts = uproot_tree_to_numpy(filename,
        #                                self.cpf_pts_branches,self.n_cpf,self.nsamples,
        #                                treename='deepntuplizer/tree', flat = True)

        #npf_pts = uproot_tree_to_numpy(filename,
        #                             self.npf_pts_branches,self.n_npf,self.nsamples,
        #                             treename='deepntuplizer/tree', flat = False)

        #vtx_pts = uproot_tree_to_numpy(filename,
        #                                 self.vtx_pts_branches,self.n_vtx,self.nsamples,
        #                             treename='deepntuplizer/tree', flat = False)


#        import uproot3 as uproot
        urfile = u3.open(filename)["deepntuplizer/tree"]
        truth_arrays = urfile.arrays(self.truth_branches)
        truth = reduceTruth(truth_arrays)
        truth = truth.astype(dtype='float32', order='C') #important, float32 and C-type!


        x_global = x_global.astype(dtype='float32', order='C')
        x_cpf = x_cpf.astype(dtype='float32', order='C')
        x_npf = x_npf.astype(dtype='float32', order='C')
        x_vtx = x_vtx.astype(dtype='float32', order='C')
        x_v0 = x_v0.astype(dtype='float32', order='C')
        x_spectator = x_spectator.astype(dtype='float32', order='C')
        #cpf_pts = cpf_pts.astype(dtype='float32', order='C')
        #npf_pts = npf_pts.astype(dtype='float32', order='C')
        #vtx_pts = vtx_pts.astype(dtype='float32', order='C')
        
        if self.remove:
#            import uproot as u
            b = [self.weightbranchX,self.weightbranchY]
            b.extend(self.truth_branches)
            b.extend(self.undefTruth)
            fileTimeOut(filename, 120)
            events = u.open(filename)["deepntuplizer/tree"]
            for_remove = events.arrays(b, library = 'pd')

#            for_remove = root2array(
 #               filename,
  #              treename = "deepntuplizer/tree",
   #             stop = None,
    #            branches = b
     #       )
            #exit()
            notremoves=weighterobjects['weigther'].createNotRemoveIndices(for_remove, use_uproot = True)
            ######notremoves=weighterobjects['weigther'].createNotRemoveIndices(for_remove)
            #undef=for_remove['isUndefined']
            undef=for_remove['isUndefined_Z']
            notremoves-=undef
            ##pu=for_remove['isPU']
            ##notremoves-=pu
            print('took ', sw.getAndReset(), ' to create remove indices')

#        random_pick = np.random.rand(x_global.shape[0])
 #       selection1 = (x_global[:,0] <= 80)
  #      selection2 = (x_global[:,0] > 80) & (x_global[:,0] <= 140)
   #     selection3 = (x_global[:,0] > 140) & (x_global[:,0] <= 186)
    #    selection4 = (x_global[:,0] > 186) & (x_global[:,0] <= 247)

#        notremoves[selection1] -= 4*random_pick[selection1]
 #       notremoves[selection2] -= 2.667*random_pick[selection2]
  #      notremoves[selection3] -= 2*random_pick[selection3]
   #     notremoves[selection4] -= 1*random_pick[selection4]

        if self.remove:
            print('remove')
            x_global=x_global[notremoves > 0]
            x_cpf=x_cpf[notremoves > 0]
            x_npf=x_npf[notremoves > 0]
            x_vtx=x_vtx[notremoves > 0]
            x_v0=x_v0[notremoves > 0]
            x_spectator=x_spectator[notremoves > 0]
            #cpf_pts=cpf_pts[notremoves > 0]
            #npf_pts=npf_pts[notremoves > 0]
            #vtx_pts=vtx_pts[notremoves > 0]
            truth=truth[notremoves > 0]

        newnsamp=x_global.shape[0]
        print('reduced content to ', int(float(newnsamp)/float(self.nsamples)*100),'%')
        
        print('remove nans')
        x_global = np.where(np.isfinite(x_global) , x_global, 0)
        x_cpf = np.where(np.isfinite(x_cpf), x_cpf, 0)
        x_npf = np.where(np.isfinite(x_npf), x_npf, 0)
        x_vtx = np.where(np.isfinite(x_vtx), x_vtx, 0)
        x_v0 = np.where(np.isfinite(x_v0), x_v0, 0)
        x_spectator = np.where(np.isfinite(x_spectator) , x_spectator, 0)
        #cpf_pts = np.where(np.isfinite(cpf_pts), cpf_pts, 0)
        #npf_pts = np.where(np.isfinite(npf_pts), npf_pts, 0)
        #vtx_pts = np.where(np.isfinite(vtx_pts), vtx_pts, 0)
        
        #if istraining:
        #    print('shuffling')
        #    index = np.arange(len(x_global))
        #    np.random.shuffle(index)
        #    x_global = x_global[index]
        #    x_cpf = x_cpf[index]
        #    x_npf = x_npf[index]
        #    x_vtx = x_vtx[index]
        #    x_v0 = x_v0[index]
        #    x_spectator = x_spectator[index]
        
        #return [x_global, x_cpf, x_npf, x_vtx, cpf_pts, npf_pts, vtx_pts], [truth], []
        return [x_global, x_cpf, x_npf, x_vtx, x_v0, x_spectator], [truth], []

    # defines how to write out the prediction
    def writeOutPrediction(self, predicted, features, truth, weights, outfilename, inputfile):
        # predicted will be a list
        
        from root_numpy import array2root
        out = np.core.records.fromarrays(np.vstack( (predicted[0].transpose(),truth[0].transpose(), features[0][:,0:2].transpose())),
                                        #names='prob_isB, prob_isC, prob_isUDS, prob_isG, isB, isC, isUDS, isG, jet_pt, jet_eta')
                                        names='prob_isB, prob_isBB, prob_isLeptB, prob_isC, prob_isUDS, prob_isG, isB, isBB, isLeptB, isC, isUDS, isG, jet_pt, jet_eta')
                                        
        array2root(out, outfilename, 'tree')
