#!/usr/bin/env python3
from argparse import ArgumentParser

parser = ArgumentParser('Apply a model to a (test) source sample.')
parser.add_argument('model')
parser.add_argument('inputModel')
parser.add_argument('trainingDataCollection', help="the training data collection. Used to infer data format and batch size.")
parser.add_argument('inputSourceFileList', help="can be text file or a DataCollection file in the same directory as the sample files, or just a single traindata file.")
parser.add_argument('outputDir', help="will be created if it doesn't exist.")
parser.add_argument("-b", help="batch size, overrides the batch size from the training data collection.",default="-1")
parser.add_argument("--gpu",  help="select specific GPU", metavar="OPT", default="")
parser.add_argument("--unbuffered", help="do not read input in memory buffered mode (for lower memory consumption on fast disks)", default=False, action="store_true")
parser.add_argument("--pad_rowsplits", help="pad the row splits if the input is ragged", default=False, action="store_true")

args = parser.parse_args()
batchsize = int(args.b)

import imp
import numpy as np
from DeepJetCore.DataCollection import DataCollection
from DeepJetCore.dataPipeline import TrainDataGenerator
import tempfile
import atexit
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_deepjet import DeepJet
from pytorch_deepjet_transformer import DeepJetTransformer
from pytorch_deepjet_transformer_V0 import DeepJetTransformerV0
from torch.optim import Adam, SGD
from tqdm import tqdm

inputdatafiles=[]
inputdir=None

print("RUNNING THE V0 VERS //////////////////////////////////")

def test_loop(dataloader, model, nbatches, pbar):
    predictions = 0
    global_vars = 0
    y_vars = 0
    
    with torch.no_grad():
        for b in range(nbatches):
            features_list, truth_list = next(dataloader)
            glob = torch.Tensor(features_list[0]).to(device)
            cpf = torch.Tensor(features_list[1]).to(device)
            npf = torch.Tensor(features_list[2]).to(device)
            vtx = torch.Tensor(features_list[3]).to(device)
            v0 = torch.Tensor(features_list[4]).to(device)
            #pxl = torch.Tensor(features_list[4]).to(device)
            y = torch.Tensor(truth_list[0]).to(device)    
            # Compute prediction
            pred = nn.Softmax(dim=1)(model(glob,cpf,npf,vtx,v0)).cpu().numpy()
            if b == 0:
                predictions = pred 
                global_vars = glob.cpu().detach().numpy()
                y_vars = y.cpu().detach().numpy()
            else:
                predictions = np.concatenate((predictions, pred), axis=0)
                global_vars = np.concatenate((global_vars, glob.cpu().detach().numpy()), axis=0)
                y_vars = np.concatenate((y_vars, y.cpu().detach().numpy()), axis=0)
            desc = 'Predicting probs : '
            pbar.set_description(desc)
            pbar.update(1)
        
    return predictions, y_vars, global_vars

## prepare input lists for different file formats
if args.inputSourceFileList[-6:] == ".djcdc":
    print('reading from data collection',args.inputSourceFileList)
    predsamples = DataCollection(args.inputSourceFileList)
    inputdir = predsamples.dataDir
    for s in predsamples.samples:
        inputdatafiles.append(s)
        
elif args.inputSourceFileList[-6:] == ".djctd":
    inputdir = os.path.abspath(os.path.dirname(args.inputSourceFileList))
    infile = os.path.basename(args.inputSourceFileList)
    inputdatafiles.append(infile)
else:
    print('reading from text file',args.inputSourceFileList)
    inputdir = os.path.abspath(os.path.dirname(args.inputSourceFileList))
    with open(args.inputSourceFileList, "r") as f:
        for s in f:
            inputdatafiles.append(s.replace('\n', '').replace(" ",""))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if args.model == 'DeepJet':
    model = DeepJet(num_classes = 5)
if args.model == 'DeepJetTransformer':
    model = DeepJetTransformer(num_classes = 5)
if args.model == 'DeepJetTransformerV0':
    model = DeepJetTransformerV0(num_classes = 5)
    
check = torch.load(args.inputModel, map_location=torch.device('cpu'))
model.load_state_dict(check['state_dict'])

model.to(device)
model.eval()

dc = None
if args.inputSourceFileList[-6:] == ".djcdc" and not args.trainingDataCollection[-6:] == ".djcdc":
    dc = DataCollection(args.inputSourceFileList)
    if batchsize < 1:
        batchsize = 1
    print('No training data collection given. Using batch size of',batchsize)
else:
    dc = DataCollection(args.trainingDataCollection)

outputs = []
os.system('mkdir -p '+args.outputDir)

for inputfile in inputdatafiles:
    
    print('predicting ',inputdir+"/"+inputfile)
    
    use_inputdir = inputdir
    if inputfile[0] == "/":
        use_inputdir=""
    outfilename = "pred_"+os.path.basename( inputfile )
    
    td = dc.dataclass()
    print('printing random things...')
    print(dc)
    print(td)

    if inputfile[-5:] == 'djctd':
        if args.unbuffered:
            td.readFromFile(use_inputdir+"/"+inputfile)
        else:
            td.readFromFileBuffered(use_inputdir+"/"+inputfile)
    else:
        print('converting '+inputfile)
        td.readFromSourceFile(use_inputdir+"/"+inputfile, dc.weighterobjects, istraining=False)

    gen = TrainDataGenerator()
    if batchsize < 1:
        batchsize = dc.getBatchSize()
    print('batch size',batchsize)
    gen.setBatchSize(batchsize)
    gen.setSquaredElementsLimit(dc.batch_uses_sum_of_squares)
    gen.setSkipTooLargeBatches(False)
    gen.setBuffer(td)

    with tqdm(total = gen.getNBatches()) as pbar:
        pbar.set_description('Predicting : ')
    ###########print(model) 
    predicted = test_loop(gen.feedNumpyData(), model, nbatches=gen.getNBatches(), pbar = pbar)
    #print("predictions...")
    #print(predicted[0])    
    #print("truths...")
    #print(predicted[1])    
    #print("globs...")
    #print(predicted[2])
    predict_np = predicted[0] 
    truths_np = predicted[1]
    globs_np = predicted[2]
    np.savez("raw_predictions.npz", predict_np, truths_np, globs_np) 
    quit()   
    x = td.transferFeatureListToNumpy()
    w = td.transferWeightListToNumpy()
    y = td.transferTruthListToNumpy()

    td.clear()
    gen.clear()
    
    if not type(predicted) == list: #circumvent that keras return only an array if there is just one list item
        predicted = [predicted]   
    overwrite_outname = td.writeOutPrediction(predicted, x, y, w, args.outputDir + "/" + outfilename, use_inputdir+"/"+inputfile)
    if overwrite_outname is not None:
        outfilename = overwrite_outname
    outputs.append(outfilename)
    
with open(args.outputDir + "/outfiles.txt","w") as f:
    for l in outputs:
        f.write(l+'\n')
