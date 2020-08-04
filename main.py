import os
import torch
import albumentations

import cv2
import numpy as np
import pandas as pd


import torch.nn as nn
from sklearn import metrics
from sklearn import model_selection
from torch.nn import functional as F
from torchvision import transforms

from torch.utils.data import Dataset
from tqdm import tqdm


from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from PIL import Image
from wtfml.utils import EarlyStopping
from wtfml.engine import Engine
from wtfml.data_loaders.image import ClassificationLoader

import pretrainedmodels



def weighted_cross_entropy_loss(prediction, target, weights= ([0.02 , 0.98])):        # 0.98 for '0' class and 0.02 for positive class
	target = target.view(-1,1)    
	if weights is not None:
	assert len(weights) == 2

	loss = weights[0] * (target.cpu() * torch.log(prediction.cpu())) + \
	       weights[1] * ((1 - target) * torch.log(1 - prediction.cpu()))
	else:
	loss = target * torch.log(prediction) + (1 - target) * torch.log(1 - prediction)

	return torch.neg(torch.mean(loss))






def csv_read():
	df = pd.read_csv("../input/siim-isic-melanoma-classification/train.csv")
	df["kfold"] = -1    
	#print (df.head())
	df = df.sample(frac=1).reset_index(drop=True)
	#print (df.head())
	y = df.target.values
	#print(len(y))
	a =y[y==0]
	#print (len(a))
	kf = model_selection.StratifiedKFold(n_splits=5)
	print(len(df))
	a = (df["image_name"][1])
	print((a))
	#for i in range(0,len(a)):
	 #   print(a[i])
	for f, (t_, v_) in enumerate(kf.split(X=df, y=y)):
	 #   print (t_,v_) 
	  #  print (f)
	    df.loc[v_, 'kfold'] = f


	#print (df)    
	df.to_csv("train_folds.csv", index=False)
	

def validation(fold, valid_loader,valid_targets,model):
	tk0 = tqdm(valid_loader, total=len(valid_loader)) 
	final_predictions =[]
	model.eval()
	#print (tk0)
	losses_valid =AverageMeter()
	with torch.no_grad():
	for idx, pack in enumerate(tk0):
	    
	    prediction = model(pack["image"].cuda())
	    loss = weighted_cross_entropy_loss(prediction,pack["label"])
	    #running_loss += loss.item()*pack["image"].size(0)
	    losses_valid.update(loss.item(), valid_loader.batch_size)
	    tk0.set_postfix(loss=losses_valid.avg)
	    final_predictions.append(prediction.cpu())

	tk0.close()

	auc = metrics.roc_auc_score(valid_targets,np.vstack(final_predictions).ravel() )

	return losses_valid.avg,auc



def train(fold):
	training_data_path = "../input/siic-isic-224x224-images/train/"
	df = pd.read_csv("/kaggle/working/train_folds.csv")
	device = "cuda"
	epochs = 50
	train_bs = 32
	valid_bs = 16

	df_train = df[df.kfold != fold].reset_index(drop=True)
	df_valid = df[df.kfold == fold].reset_index(drop=True)
	valid_targets = df_valid.target.values
	#print(valid_targets.shape)
	model = SEResnext50_32x4d(pretrained="imagenet")
	model.to(device)

	mean = (0.485, 0.456, 0.406)
	std = (0.229, 0.224, 0.225)
	transform_train = transforms.Compose([
	transforms.ToPILImage(),
	transforms.RandomHorizontalFlip(p=0.5),
	transforms.RandomRotation(degrees=(-90, 90)),
	transforms.RandomVerticalFlip(p=0.5),
	transforms.ToTensor(),
	transforms.Normalize(mean, std),
	])

	transform_valid = transforms.Compose([
	#transforms.ToPILImage(),
	transforms.ToTensor(),
	transforms.Normalize(mean, std),
	])

	train_data = MelenomaDataset(training_data_path, df_train, transform_train)
	valid_data = MelenomaDataset(training_data_path,df_valid,transform_valid)

	train_loader = DataLoader(train_data, batch_size= train_bs, shuffle =False, num_workers=4)

	valid_loader = DataLoader(valid_data,batch_size = valid_bs, shuffle = False, num_workers = 4)





	optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
	optimizer,
	patience=3,
	threshold=0.001,
	mode="max"
	)

	es = EarlyStopping(patience=5, mode="max")

	#loss_v , auc = validation(fold,valid_loader,valid_targets,model) 

	for epoch in range(epochs):
	count = 0
	tk0 = tqdm(train_loader, total=len(train_loader)) 
	#print (tk0)
	losses =AverageMeter()
	for idx, pack in enumerate(tk0):
	    
	    if (idx == 0):
		optimizer.zero_grad()
	    model.train()
	    prediction = model(pack["image"].cuda())
	    loss = weighted_cross_entropy_loss(prediction,pack["label"])
	    #running_loss += loss.item()*pack["image"].size(0)
	    loss.backward()
	    optimizer.step()
	    scheduler.step(loss)
	    optimizer.zero_grad()
	    losses.update(loss.item(), train_loader.batch_size)
	    tk0.set_postfix(loss=losses.avg)
	    

	print (epoch)
	print (f"training_loss for {epoch} = {losses.avg}")    
	loss_v , auc = validation(fold,valid_loader,valid_targets,model) 

	print (f"validation_loss for {epoch}= {loss_v}")
	print(f"Epoch = {epoch}, AUC = {auc}")

	es(auc, model, model_path=f"model_fold_{fold}.bin")
	if es.early_stop:
	    print("Early stopping")
	    break        


	tk0.close()


def predict(fold):
	test_data_path = "../input/siic-isic-224x224-images/test/"
	df = pd.read_csv("../input/siim-isic-melanoma-classification/test.csv")
	device = "cuda"
	model_path=f"model_fold_{fold}.bin"

	mean = (0.485, 0.456, 0.406)
	std = (0.229, 0.224, 0.225)
	aug = albumentations.Compose(
	[
	    albumentations.Normalize(mean, std, max_pixel_value=255.0, always_apply=True)
	]
	)

	images = df.image_name.values.tolist()
	images = [os.path.join(test_data_path, i + ".png") for i in images]
	targets = np.zeros(len(images))

	test_dataset = ClassificationLoader(
	image_paths=images,
	targets=targets,
	resize=None,
	augmentations=aug,
	)

	test_loader = torch.utils.data.DataLoader(
	test_dataset, batch_size=16, shuffle=False, num_workers=4
	)

	model = SEResnext50_32x4d(pretrained=None)
	model.load_state_dict(torch.load(model_path))
	model.to(device)

	predictions = Engine.predict(test_loader, model, device=device)
	predictions = np.vstack((predictions)).ravel()

	return predictions	

if __name__ == '__main__':
	train(0)
	train(1)
	train(2)
	train(3)
	train(4)
	predict(0)
	predict(1)
	predict(2)
	predict(3)
	predict(4)
