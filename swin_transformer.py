import os
import pandas as pd
import numpy as np
import cv2 as cv
from sklearn import model_selection
from tqdm.auto import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import timm
import glob
from xgboost import XGBClassifier
from torchinfo import summary

import albumentations
from albumentations.pytorch.transforms import ToTensorV2

import gc
gc.enable()

import warnings
import sklearn.exceptions
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)
train_holdout_csv_path = 'E:\Petfinder_data/regular_csv/train_holdout.csv'
test_holdout_csv_path = 'E:\Petfinder_data/regular_csv/test_holdout.csv'
train_csv_path = 'E:\Petfinder_data\petfinder-pawpularity-score/train.csv'
test_csv_path = 'E:\Petfinder_data\petfinder-pawpularity-score/test.csv'
train_image_path = 'E:/Petfinder_data/petfinder-pawpularity-score/train'
submission_csv_path = 'E:/Petfinder_data/petfinder-pawpularity-score/sample_submission.csv'
pretrained_model_path = 'E:/Petfinder_data/swin_small_patch4_window7_224.pth'
base_save_path = 'E:/Petfinder_data'
holdout_save_path = os.path.join(base_save_path, 'holdout_save')
if not os.path.exists(holdout_save_path):
    os.mkdir(holdout_save_path)
    




def train_transform_object(DIM = 384):
    return albumentations.Compose(
        [
            albumentations.HorizontalFlip(p=0.5),
            albumentations.VerticalFlip(p=0.5),
            albumentations.Rotate(limit=180, p=0.5),
            albumentations.RandomBrightnessContrast(
                brightness_limit=(-0.1, 0.1),
                contrast_limit=(-0.1, 0.1), p=0.5
            ),
            albumentations.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(p=1.0),
        ]
    )

def valid_transform_object(DIM = 384):
    return albumentations.Compose(
        [
            albumentations.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(p=1.0)
        ]
    )

class PetDataset(Dataset):
    
    def __init__(self, image_paths, dense_features, targets, transform=None):
        self.image_paths = image_paths
        self.dense_feats = dense_features
        self.targets = targets
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index):
        #read the image using the path.
        img = cv.imread(self.image_paths[index], 1)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img = cv.resize(img, (224, 224), interpolation = cv.INTER_AREA)
        
        if self.transform is not None:
            img = self.transform(image=img)['image']
            
        img = img.float()
            
        #get the dense features.
        dense = self.dense_feats[index, :]
        
        #get the label and convert it to 0 to 1.
        label = torch.tensor(self.targets[index]).float()
        
        return (img, dense, label)

class PetTestset(Dataset):
    
    def __init__(self, image_paths, dense_features, transform=None):
        self.image_paths = image_paths
        self.dense_feats = dense_features
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index):
        #read the image using the path.
        img = cv.imread(self.image_paths[index], 1)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img = cv.resize(img, (224, 224), interpolation = cv.INTER_AREA)
        
        if self.transform is not None:
            img = self.transform(image=img)['image']
            
        img = img.float()
            
        #get the dense features.
        dense = self.dense_feats[index, :]
        
        return (img, dense)

class PetNet(nn.Module):
    def __init__(self, model_name, out_features, inp_channels, pretrained, num_dense):
        super().__init__()
        self.model = timm.create_model(model_name=model_name, pretrained=pretrained, in_chans=inp_channels)
        self.model.load_state_dict(torch.load(pretrained_model_path)['model'])
        n_features = self.model.head.in_features
        self.model.head = nn.Linear(n_features, 128)
        self.fc = nn.Sequential(
            nn.Linear(128 + num_dense, 64),
            nn.ReLU(),
            nn.Linear(64, out_features)
        )
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, image, dense):
        embeddings = self.model(image)
        x = self.dropout(embeddings)
        x = torch.cat([x, dense], dim=1)
        output = self.fc(x)
        return output


def usr_rmse_score(output, target):
    y_pred = torch.sigmoid(output).cpu()
    y_pred = y_pred.detach().numpy()*100
    target = target.cpu()*100
    
    return mean_squared_error(target, y_pred, squared=False)

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

def train_fn(train_loader, model, loss_fn, optimizer, epoch, device, scheduler=None):
    model.train()
    stream = tqdm(train_loader)
    loss_sum = 0.0
    count=0.0
    rmse_sum = 0.0
    skf = model_selection.StratifiedKFold(n_splits=5)
    for i, (image, dense, target) in enumerate(stream, start=1):
        image = image.to(device, non_blocking=True)
        dense = dense.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True).float().view(-1, 1)
        
        output = model(image, dense)
        loss = loss_fn(output, target)
        rmse = usr_rmse_score(output, target)
        loss_sum += loss
        rmse_sum += rmse
        count += 1
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        stream.set_description(f"Epoch {epoch:02}. Train. Loss {loss}. RMSE {rmse}")
    return float(loss_sum) / float(count) , float(rmse_sum) / float(count)

def validation_fn(validation_loader, model, loss_fn, epoch, device):
    model.eval()
    stream = tqdm(validation_loader)
    final_targets = []
    final_outputs = []
    loss_sum = 0.0
    count=0.0
    rmse_sum = 0.0
    with torch.no_grad():
        for i, (image, dense, target) in enumerate(stream, start=1):
            image = image.to(device, non_blocking=True)
            dense = dense.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True).float().view(-1, 1)
            
            output = model(image, dense)
            loss = loss_fn(output, target)
            rmse_score = usr_rmse_score(output, target)
            loss_sum += loss
            rmse_sum += rmse_score
            count += 1
            stream.set_description(f"Epoch: {epoch:02}. Valid. Loss {loss}. RMSE {rmse_score}")
            
            targets = (target.detach().cpu().numpy()*100).tolist()
            outputs = (torch.sigmoid(output).detach().cpu().numpy()*100).tolist()
            
            final_targets.extend(targets)
            final_outputs.extend(outputs)
        
    return final_targets, final_outputs, float(loss_sum) / float(count) , float(rmse_sum) / float(count)

def test_fn(test_loader, model, device):
    model.eval()
    stream = tqdm(test_loader)
    final_outputs = []
    
    with torch.no_grad():
        for i, (image, dense) in enumerate(stream, start=1):
            image = image.to(device, non_blocking=True)
            dense = dense.to(device, non_blocking=True)
            output = model(image, dense)
            outputs = (torch.sigmoid(output).detach().cpu().numpy()*100).tolist()
            final_outputs.extend(outputs)
        
    return final_outputs

best_models_of_each_fold = []
rmse_tracker = []
FOLDS = 1
EPOCHS = 30

def get_dataset(df, images, state='training'):
    ids = list(df['Id'])
    image_paths = [os.path.join(images, idx + '.jpg') for idx in ids]
    df['Pawpularity'] = df['Pawpularity']/100
    target = df['Pawpularity'].values
    df.drop(['Id', 'Pawpularity', 'index'], inplace=True, axis=1)
    dense_feats = df.values

    if state == 'training':
        transform = train_transform_object(224)
    elif state == 'validation' or state == 'testing':
        transform = valid_transform_object(224)
    else:
        transform = None

    return PetDataset(image_paths, dense_feats, target, transform)
sfk = model_selection.StratifiedKFold(n_splits=5)
train_csv = pd.read_csv(train_holdout_csv_path)
images = train_image_path
for index, (train_index, valid_index) in enumerate(sfk.split(train_csv, train_csv['Pawpularity'])):
    train = train_csv.iloc[train_index]
    valid = train_csv.iloc[valid_index]

    train_dataset = get_dataset(train, images)
    val_dataset = get_dataset(valid, images, state='validation')

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    model_params = {
        'model_name' : 'swin_small_patch4_window7_224',
        'out_features' : 1,
        'inp_channels' : 3,
        'pretrained' : False,
        'num_dense' : 12,
    }
    print('Load model')
    model = PetNet(**model_params)
    model1 = model.to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-6, amsgrad=False)
    schedular = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=1000, eta_min=1e-6)

    best_rmse = np.inf
    best_epoch = np.inf
    best_model_name = None
    for epoch in range(EPOCHS):
        print('epoch',epoch)
        train_loss_avg, train_rmse_avg = train_fn(train_loader, model, loss_fn, optimizer, epoch, device, scheduler=schedular)
        valid_targets, predictions, valid_loss_avg, valid_rmse_avg  = validation_fn(val_loader, model, loss_fn, epoch, device)
        rmse = round(mean_squared_error(valid_targets, predictions, squared=False), 3)
        writer.add_scalar("Fold_{}".format(index)+"Loss/train", train_loss_avg, epoch)
        writer.add_scalar("Fold_{}".format(index)+"RMSE/train", train_rmse_avg, epoch)
        writer.add_scalar("Fold_{}".format(index)+"Loss/valid", valid_loss_avg, epoch)
        writer.add_scalar("Fold_{}".format(index)+"RMSE/valid", rmse, epoch)
        writer.add_scalar("Fold_{}".format(index)+"Learning_rate", optimizer.param_groups[0]['lr'],epoch)
        print("train_loss",train_loss_avg,"train_rmse",train_rmse_avg,"valid_loss",valid_loss_avg,"valid_rmse",rmse)
        if rmse < best_rmse:
            best_rmse = rmse
            best_epoch = epoch
            if best_model_name is not None:
                os.remove(best_model_name)
            torch.save(model.state_dict(),os.path.join(holdout_save_path, f"{model_params['model_name']}_{index}_fold_{epoch}_epoch_{rmse}_rmse.pth"))
            best_model_name = os.path.join(holdout_save_path, f"{model_params['model_name']}_{index}_fold_{epoch}_epoch_{rmse}_rmse.pth")

            print(f'The Best saved model is: {best_model_name}')
    schedular.step()
    best_models_of_each_fold.append(best_model_name)
    rmse_tracker.append(best_rmse)
    print(''.join(['#']*50))
    del model
    gc.collect()
    torch.cuda.empty_cache()
    writer.flush()
    writer.close()
        

predicted_labels = None
models_dir = './'
model_params = {
    'model_name' : 'swin_small_patch4_window7_224',
    'out_features' : 1,
    'inp_channels' : 3,
    'num_dense' : 12,
    'pretrained' : False
}

def get_testset(df, images):
    ids = list(df['Id'])
    image_paths = [os.path.join(images, idx + '.jpg') for idx in ids]
    df.drop(['Id'], inplace=True, axis=1)
    dense_feats = df.values
    test_transform = valid_transform_object()
    return PetTestset(image_paths, dense_feats, test_transform)

outputs = None
for model_name in glob.glob(models_dir + '/*.pth'):
    model = PetNet(**model_params)
    model.load_state_dict(torch.load(model_name))
    model = model.to(device)
    
    test_images = test_images
    test_df = pd.read_csv(test_holdout_csv_path)
    testset = get_testset(test_df, test_images)
    test_loader = DataLoader(testset, batch_size=16, shuffle=False)
    
    if outputs is None:
        outputs = test_fn(test_loader, model, device)
    else:
        temp = test_fn(test_loader, model, device)
        for i in range(len(temp)):
            outputs[i].append(temp[i][0])
            
for i in range(len(outputs)):
    outputs[i] = [sum(outputs[i]) / (len(glob.glob(models_dir + '/*.pth')))]
        
sub_csv = pd.read_csv(submission_csv_path)
for i in range(len(outputs)):
    sub_csv.loc[i, 'Pawpularity'] = outputs[i][0]

sub_csv.to_csv('submission.csv', index=False)

