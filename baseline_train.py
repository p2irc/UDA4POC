import torch
from torch import nn,optim
from torch.utils.data import DataLoader,Dataset
from torchsummary import summary
from torch.autograd import Function
from torch.optim.lr_scheduler import StepLR
import torchvision.transforms.functional as TF
import torchvision

from torchvision import transforms
from PIL import Image
import pickle
from tqdm import tqdm
import random
from sklearn import metrics
from skimage import io, filters
import joblib
import json

import numpy as np
import matplotlib.pyplot as plt
import glob

import data_loader as dl
import baseline
from argparse import ArgumentParser
from torch.utils.tensorboard import SummaryWriter

parser = ArgumentParser()
parser.add_argument("-s", "--source", help="source domain path")
parser.add_argument("-o", "--write_weight",help="write weight to")
parser.add_argument("-e", "--epoch",help="number of epoch")
parser.add_argument("-w", "--weight",help="model weight")


args = parser.parse_args()

SOURCE_DATA = args.source
weight_path = args.write_weight
EPOCHS = int(args.epoch)
weight = args.weight


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
writer = SummaryWriter()
transf = transforms.Compose([transforms.ToTensor()])

SOURCE_TRAIN_SIZE = 0.8
TRAINING_BATCH = 8


source_data_keys = dl.get_keys_from_pickle_dict(SOURCE_DATA)
random.shuffle(source_data_keys)

source_ds_size = len(source_data_keys)

source_train_keys,source_valid_keys = source_data_keys[:int(source_ds_size*SOURCE_TRAIN_SIZE)],source_data_keys[int(source_ds_size*SOURCE_TRAIN_SIZE):]

train_ds = dl.Data_baseline(SOURCE_DATA,ds_order=source_train_keys,transform=transf)
valid_ds = dl.Data_baseline(SOURCE_DATA,ds_order=source_valid_keys,transform=transf)

train_ds_loader = DataLoader(train_ds,batch_size=TRAINING_BATCH,drop_last=True,shuffle=False)
valid_ds_loader = DataLoader(valid_ds,batch_size=8,drop_last=True,shuffle=False)

baseline_model = baseline.CountEstimate()
if weight!=None:
	baseline_model.load_state_dict(torch.load(weight))
baseline_model.to(device)

criterion = nn.MSELoss()

learning_rate = 1e-3
optimizer = optim.Adam(baseline_model.parameters(),lr=learning_rate)


with open('imgz_for_plot.pkl','rb') as fp:
	prv_x = pickle.load(fp)

train_loss = []
valid_loss = []



for epoch in tqdm(range(EPOCHS)):
    baseline_model.train()
    epoch_loss,steps = 0,0
    
    for x,y in train_ds_loader:
        
        x,y = x.to(device),y.to(device)
        
        optimizer.zero_grad()
        
        pred_density_map = baseline_model(x)
        
        cost1 = criterion(pred_density_map,y)
        cost = torch.log(cost1)

        epoch_loss+=((cost1))
        steps+=1
        
        cost.backward()
        optimizer.step()

    cur_train_loss = epoch_loss/steps
    train_loss.append(cur_train_loss)

    img_grid0 = torchvision.utils.make_grid(prv_x)
    writer.add_image('input',img_grid0 ,epoch)

    pred_out = baseline_model(prv_x.to(device))
    pred_out = (pred_out-pred_out.min())/(pred_out.max()-pred_out.min()) 
    img_grid = torchvision.utils.make_grid(pred_out)
    writer.add_image('target prediction',img_grid,epoch)

    
    v_loss = 0 
    steps = 0
    with torch.no_grad():
        baseline_model.eval()
        for x,y in valid_ds_loader:
            x,y = x.to(device), y.to(device)
            
            pred_density_map = baseline_model(x)
            
            cost1 = criterion(pred_density_map,y)            
            cost = torch.log(cost1) 
            
            v_loss+= cost
            steps+=1

    cur_val_loss = v_loss/steps
    
    valid_loss.append(cur_val_loss)
    writer.add_scalar('Loss/train', cur_train_loss, epoch)
    writer.add_scalar('Loss/valid', cur_val_loss, epoch)


torch.save(baseline_model.state_dict(),weight_path)
