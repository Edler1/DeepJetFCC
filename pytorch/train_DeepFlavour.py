import torch 
import torch.nn as nn
#from DeepJetCore.training.pytorch_first_try import training_base
##from pytorch_first_try_NOTV0 import training_base
from pytorch_first_try import training_base
from pytorch_deepjet import *
from pytorch_deepjet_transformer import DeepJetTransformer
from pytorch_deepjet_transformer_V0 import DeepJetTransformerV0
#from pytorch_deepjet_transformer_v2 import DeepJetTransformerv2
from pytorch_ranger import Ranger



from prettytable import PrettyTable
#https://stackoverflow.com/questions/49201236/check-the-total-number-of-parameters-in-a-pytorch-model
def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params+=params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params
    






def cross_entropy_one_hot(input, target):
    _, labels = target.max(dim=1)
    return nn.CrossEntropyLoss()(input, labels)

#num_epochs = 80
#num_epochs = 50
num_epochs = 70
#num_epochs = 120
#num_epochs = 5
#num_epochs = 20
#num_epochs = 1

lr_epochs = max(1, int(num_epochs * 0.3))
lr_rate = 0.01 ** (1.0 / lr_epochs)
mil = list(range(num_epochs - lr_epochs, num_epochs))

#######model = DeepJet(num_classes = 5) #DeepJetTransformer(num_classes = 5)
#model = DeepJetTransformer(num_classes = 5)
model = DeepJetTransformerV0(num_classes = 5)

print(model)
pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Total trainable params = "+str(pytorch_total_params))
count_parameters(model)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

criterion = cross_entropy_one_hot
optimizer = Ranger(model.parameters(), lr = 5e-3) #torch.optim.Adam(model.parameters(), lr = 0.003, eps = 1e-07)
#scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones = [1], gamma = 0.1)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones = mil, gamma = lr_rate)

train=training_base(model = model, criterion = criterion, optimizer = optimizer, scheduler = scheduler, testrun=False)

train.train_data.maxFilesOpen=1

model = train.trainModel(nepochs=num_epochs+lr_epochs, batchsize=4000)
#model,history = train.trainModel(nepochs=num_epochs+lr_epochs, batchsize=4000)
#model,history = train.trainModel(nepochs=num_epochs+lr_epochs, batchsize=512)
