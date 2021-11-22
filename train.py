# Basic library
import os
import pandas as pd
import numpy as np
import cv2 as cv
from tqdm.auto import tqdm
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import glob
from xgboost import XGBClassifier
import warnings
import sklearn.exceptions
import copy
import trainer

# Torch
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import timm
from torchinfo import summary
import albumentations
from albumentations.pytorch.transforms import ToTensorV2

from omegaconf import DictConfig, OmegaConf
import gc
gc.enable()

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)

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
        self.model.load_state_dict(torch.load('/media/ddl/새 볼륨/Git/kaggle_project2/save_model/pre-trained/swin_small_patch4_window7_224.pth')['model'])
        #self.model.load_state_dict(torch.load('/media/ddl/새 볼륨/Git/kaggle_project2/save_model/previous/swin_small_patch4_window7_224_0_fold_3_epoch_27.483_rmse.pth').model)
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

########################## Training ##############################
train = pd.read_csv('/media/ddl/새 볼륨/Git/kaggle_project2/hold_out/regular/train_holdout.csv')
val = pd.read_csv('/media/ddl/새 볼륨/Git/kaggle_project2/hold_out/regular/test_holdout.csv')
images = '/media/ddl/새 볼륨/Git/kaggle_project2/dataset/petfinder-pawpularity-score/train'

train_dataset = get_dataset(train, images)
val_dataset = get_dataset(val, images, state='validation')

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)

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
#model1.model.load_state_dict(torch.load('/media/ddl/새 볼륨/Git/kaggle_project2/f/pre-trained/swin_small_patch4_window7_224.pth'))
#model.model.load_state_dict(torch.load('/media/ddl/새 볼륨/Git/kaggle_project2/save_model/previous/swin_small_patch 4_window7_224_0_fold_3_epoch_27.483_rmse.pth'))
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-6, amsgrad=False)

best_rmse = np.inf
best_epoch = np.inf
best_model_name = None
for epoch in range(EPOCHS):
    print('epoch',epoch)
    train_loss_avg, train_rmse_avg = train_fn(train_loader, model, loss_fn, optimizer, epoch, device)
    valid_targets, predictions, valid_loss_avg, valid_rmse_avg  = validation_fn(val_loader, model, loss_fn, epoch, device)
    rmse = round(mean_squared_error(valid_targets, predictions, squared=False), 3)
    writer.add_scalar("Loss/train", train_loss_avg, epoch)
    writer.add_scalar("RMSE/train", train_rmse_avg, epoch)
    writer.add_scalar("Loss/valid", valid_loss_avg, epoch)
    writer.add_scalar("RMSE/valid", rmse, epoch)
    print("train_loss",train_loss_avg,"train_rmse",train_rmse_avg,"valid_loss",valid_loss_avg,"valid_rmse",rmse)
    torch.save(model.state_dict(),f"{model_params['model_name']}_{fold}_fold_{epoch}_epoch_{rmse}_rmse.pth")
    if rmse < best_rmse:
        best_rmse = rmse
        best_epoch = epoch
        if best_model_name is not None:
            os.remove(best_model_name)
        torch.save(model.state_dict(),f"{model_params['model_name']}_{fold}_fold_{epoch}_epoch_{rmse}_rmse.pth")
        best_model_name = f"{model_params['model_name']}_{fold}_fold_{epoch}_epoch_{rmse}_rmse.pth"

        print(f'The Best saved model is: {best_model_name}')
        
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
    
    test_images = '/media/ddl/새 볼륨/Git/kaggle_project2/dataset/petfinder-pawpularity-score/test'
    test_df = pd.read_csv('/media/ddl/새 볼륨/Git/kaggle_project2/dataset/petfinder-pawpularity-score/test.csv')
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
        
sub_csv = pd.read_csv('/media/ddl/새 볼륨/Git/kaggle_project2/dataset/petfinder-pawpularity-score/sample_submission.csv')
for i in range(len(outputs)):
    sub_csv.loc[i, 'Pawpularity'] = outputs[i][0]

sub_csv.to_csv('submission.csv', index=False)

class Trainer():
    def __init__(self, conf) -> None:
        self.conf = copy.deepcopy(conf)

    def build_model(self, num_classes=-1):
        model = trainer.architecture.create(self.conf.architecture)
        model = model.to(device=self.rank, non_blocking=True)

        return model

    def build_optimizer(self, model):
        optimizer = trainer.optimizer.create(self.conf.optimizer, model)
        return optimizer

    def build_dataloader(self, ):
        train_loader, train_sampler = trainer.new_dataset.create(conf = self.conf.dataset,
        world_size=self.conf.base.world_size,
        local_rank=self.rank,
        batch_size = self.conf.hyperparameter.batch_size,
        mode = 'train')

        valid_loader, valid_sampler = trainer.new_dataset.create(conf = self.conf.dataset,
        world_size=self.conf.base.world_size,
        local_rank=self.rank,
        batch_size = self.conf.hyperparameter.batch_size,
        mode = 'train')

        test_loader, test_sampler = trainer.new_dataset.create(conf = self.conf.dataset,
        world_size=self.conf.base.world_size,
        local_rank=self.rank,
        batch_size = self.conf.hyperparameter.batch_size,
        mode = 'test')


        return train_loader, train_sampler, valid_loader, valid_sampler, test_loader, test_sampler

    def build_loss(self):
        criterion = trainer.loss.create(self.conf.loss, self.rank)
        criterion.to(device=self.rank, non_blocking=True)

        return criterion

    def build_saver(self, model, optimizer, scaler):
        saver = trainer.saver.create(self.conf.saver, model, optimizer, scaler)

        return saver
    
    def load_model(self, model, path):
        data = torch.load(path)
        key = 'model' if 'model' in data else 'state_dict'

        if not isinstance(model, (DataParallel, DDP)):
            model.load_state_dict({k.replace('module.', ''): v for k, v in data[key].items()})
        else:
            model.load_state_dict({k if 'module.' in k else 'module.'+k: v for k, v in data[key].items()})
        return model

    def train_one_epoch(self, epoch, model, dl, criterion, optimizer,logger):
        # for step, (image, label) in tqdm(enumerate(dl), total=len(dl), desc="[Train] |{:3d}e".format(epoch), disable=not flags.is_master):
        train_hit = 0
        train_total = 0
        one_epoch_loss = 0
        # 0: train_loss, 1: train_hit, 2: train_total, 3: len(dl)
        counter = torch.zeros((4, ), device=self.rank)
        #torch.set_default_tensor_type(torch.cuda.LONG)
        model.train()
        target = torch.empty(3, dtype=torch.long).random_(5)
        print("target",target)
        pbar = tqdm(
            enumerate(dl), 
            bar_format='{desc:<15}{percentage:3.0f}%|{bar:18}{r_bar}', 
            total=len(dl), 
            desc=f"train:{epoch}/{self.conf.hyperparameter.epochs}", 
            disable=not self.is_master
            )
        current_step = epoch
        prediclist = []
        labellist = []
        for step, (image, label) in pbar:
            image = image.to(device=self.rank, non_blocking=True).float()
            label = label.to(device=self.rank, non_blocking=True).float()
            with self.amp_autocast():
                input = image
                y_pred = model(input)
                label = label.to(torch.int64)
                loss = criterion(y_pred, label).float()
            optimizer.zero_grad(set_to_none=True)
            
            if self.scaler is None:
                loss.backward()
                optimizer.step()
            else:
                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()
            # counter [loss, correct, image_num_in_one_epoch, image_num_in_one_epoch??]
            counter[0] += loss.item()

            # _, y_pred = y_pred.unsqueeze(0).max(1)
            y_pred = y_pred.squeeze()
            counter[1] += y_pred.detach().eq(label).sum()
            counter[2] += image.shape[0]
            
            pred_index = torch.unsqueeze(torch.argmax(y_pred.detach().cpu(), dim=0),0)
            prediclist.append(pred_index)
            labellist.append(label.cpu().numpy())
            if step % 100 == 0:
                score = accuracy_score(label.cpu().numpy(), pred_index)
                loss_for_vis = round(loss.item(), 2)
                pbar.set_postfix({'train_Acc': score,'train_Loss': loss_for_vis}) 

        counter[3] += len(dl)
        torch.distributed.reduce(counter, 0)
        if self.is_master:
            counter = counter.detach().cpu().numpy()
            labellist = np.array(list(itertools.chain(*labellist)))
            prediclist = np.array(list(itertools.chain(*prediclist)))
            fpr,tpr,thresholds  = metrics.roc_curve(labellist,prediclist,pos_label=1)

            prescore = precision_score(labellist,prediclist > 0.6)
            acccore = accuracy_score(labellist,prediclist > 0.6)
            # print(f'[Train_{epoch}] Acc: {train_hit / train_total} Loss: {one_epoch_loss / len(dl)}')
            metric = {'AUROC':metrics.auc(fpr, tpr),'Acc': acccore,'pre':prescore, 'Loss': counter[0] / counter[3],'optimizer':optimizer}
            logger.update_log(metric,current_step,'train') # update logger step
            logger.update_histogram(model,current_step,'train') # update weight histogram 
            logger.update_image(image,current_step,'train') # update transpose image
            logger.update_metric(labellist,prediclist,current_step,'train')
        # return loss, accuracy
        return counter[0] / counter[3], counter[1] / counter[2], dl


    @torch.no_grad()
    def eval(self, epoch, model, dl, criterion,logger):
        # 0: val_loss, 1: val_hit, 2: val_total, 3: len(dl)
        counter = torch.zeros((4, ), device=self.rank)
        model.eval()
        pbar = tqdm(
            enumerate(dl),
            bar_format='{desc:<15}{percentage:3.0f}%|{bar:18}{r_bar}', 
            total=len(dl),
            desc=f"val  :{epoch}/{self.conf.hyperparameter.epochs}", 
            disable=not self.is_master
            ) # set progress bar
        current_step = epoch
        prediclist = []
        labellist = []

        for step, (image, label) in pbar:
            image = image.to(device=self.rank, non_blocking=True).float()
            label = label.to(device=self.rank, non_blocking=True).float()
            with self.amp_autocast():
                input = image
                y_pred = model(input)
                label = label.to(torch.int64)
                loss = criterion(y_pred, label).float()
            counter[0] += loss.item()
            y_pred_copy = torch.tensor(y_pred)
            if self.conf.loss.type == 'bce':
                y_pred_copy = torch.round(torch.sigmoid(y_pred_copy))
            else:
                _, y_pred_copy = y_pred_copy.max(1)
                # one_hot encoding
                if len(list(label.shape)) > 2:
                    _, label = label.max(1)
            counter[1] += y_pred_copy.detach().eq(label).sum()
            counter[2] += image.shape[0]
            pred_index = torch.unsqueeze(torch.argmax(y_pred.detach().cpu(), dim=0),0)
            prediclist.append(pred_index)
            labellist.append(label.cpu().numpy())
            if step % 100 == 0:
                score = accuracy_score(label.cpu().numpy(), pred_index)
                loss_for_vis = round(loss.item(), 2)
                pbar.set_postfix({'valid_Acc': score,'valid_Loss': loss_for_vis}) 
        counter[3] += len(dl)
        torch.distributed.reduce(counter, 0)
        if self.is_master:
            counter = counter.detach().cpu().numpy()
            labellist = np.array(list(itertools.chain(*labellist)))
            prediclist = np.array(list(itertools.chain(*prediclist)))
            fpr,tpr,thresholds  = metrics.roc_curve(labellist,prediclist,pos_label=1)
            prescore = precision_score(labellist,prediclist > 0.6)
            acccore = accuracy_score(labellist,prediclist>0.6)

            # print(f'[Val_{epoch}] Acc: {counter[1] / counter[2]} Loss: {counter[0] / counter[3]}')
            # metric = {'Acc':counter[1] / counter[2], 'Loss': counter[0] / counter[3]}
            metric = {'AUROC':metrics.auc(fpr, tpr),'Acc': acccore,'pre':prescore, 'Loss': counter[0] / counter[3]}
            logger.update_log(metric,current_step,'valid') # update logger step
            logger.update_histogram(model,current_step,'valid') # add image 
            # logger.update_image(image,current_step,'valid') # update transpose image
            # y_pred_ = np.array(predic_metric)[:,0].flatten()
            # label_ = np.array(predic_metric)[:,1].flatten()
            logger.update_metric(labellist,prediclist,current_step,'valid') # update transpose image
            
        return counter[0] / counter[3], counter[1] / counter[2]

    def train_eval(self):
        model = self.build_model()
        criterion = self.build_loss()
        optimizer = self.build_optimizer(model)

        scheduler = self.build_scheduler(optimizer)
        train_dl, train_sampler, valid_dl, valid_sampler, test_dl, test_sampler= self.build_dataloader()

        logger = self.build_looger(is_use=self.is_master)
        saver = self.build_saver(model, optimizer, self.scaler)
        # Wrap the model
        
        # initialize
        for name, x in model.state_dict().items():
            dist.broadcast(x, 0)
        torch.cuda.synchronize()

        # add graph to tensorboard
        if logger is not None:
            logger.update_graph(model, torch.rand((1,1,28,28)).float())

        # load checkpoint
        if self.conf.base.resume == True:
            self.start_epoch = saver.load_for_training(model,optimizer,self.rank,scaler=None)
        
        for epoch in range(self.start_epoch, self.conf.hyperparameter.epochs + 1):
            train_sampler.set_epoch(epoch)
            # train
            train_loss, train_acc, train_dl = self.train_one_epoch(epoch, model, train_dl, criterion, optimizer, logger)
            scheduler.step()

            # eval
            valid_loss, valid_acc = self.eval(epoch, model, valid_dl, criterion, logger)
            
            torch.cuda.synchronize()

            # save_model
            saver.save_checkpoint(epoch=epoch, model=model, loss=train_loss, rank=self.rank, metric=valid_acc)

            if self.is_master:
                print(f'Epoch {epoch}/{self.conf.hyperparameter.epochs} - train_Acc: {train_acc:.3f}, train_Loss: {train_loss:.3f}, valid_Acc: {valid_acc:.3f}, valid_Loss: {valid_loss:.3f}')

    def run(self):
        if self.conf.base.mode == 'train':
            pass
        elif self.conf.base.mode == 'train_eval':
            self.train_eval()
        elif self.conf.base.mode == 'finetuning':
            pass

def set_seed(conf):
    if conf.base.seed is not None:
        conf.base.seed = int(conf.base.seed, 0)
        print(f'[Seed] :{conf.base.seed}')
        os.environ['PYTHONHASHSEED'] = str(conf.base.seed)
        random.seed(conf.base.seed)
        np.random.seed(conf.base.seed)
        torch.manual_seed(conf.base.seed)
        torch.cuda.manual_seed(conf.base.seed)
        torch.cuda.manual_seed_all(conf.base.seed)  # if use multi-G
        torch.backends.cudnn.deterministic = True

@hydra.main(config_path='conf', config_name='mine')
def main(conf: DictConfig) -> None:
    print(f'Configuration\n{OmegaConf.to_yaml(conf)}')
    
    

if __name__ == '__main__':
    main()

