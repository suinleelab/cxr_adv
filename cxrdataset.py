#!/usr/bin/env python3
import os 

import numpy
import torch
import pandas
import sklearn.model_selection
from torchvision import transforms
from PIL import Image
import random

# use imagenet mean,std for normalization
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

def _get_patient_id(path):
    return path.split('/')[2]

def _get_unique_patient_ids(dataframe):
    ids = list(dataframe.index)
    ids = [_get_patient_id(i) for i in ids]
    ids = list(set(ids))
    ids.sort()
    return ids
    
def grouped_split(dataframe, random_state=None, test_size=0.05):
    '''
    Split a dataframe such that patients are disjoint in the resulting folds.
    The dataframe must have an index that contains strings that may be processed
    by _get_patient_id to return the unique patient identifiers.
    '''
    groups = _get_unique_patient_ids(dataframe)
    traingroups, testgroups = sklearn.model_selection.train_test_split(
            groups,
            random_state=random_state,
            test_size=test_size)
    traingroups = set(traingroups)
    testgroups = set(testgroups)

    trainidx = []
    testidx = []
    for idx, row in dataframe.iterrows():
        patient_id = _get_patient_id(idx)
        if patient_id in traingroups:
            trainidx.append(idx)
        elif patient_id in testgroups:
            testidx.append(idx)
    traindf = dataframe.loc[dataframe.index.isin(trainidx),:]
    testdf = dataframe.loc[dataframe.index.isin(testidx),:]
    return traindf, testdf

class CXRDataset(torch.utils.data.Dataset):
    '''
    Base class for chest radiograph datasets.
    '''
    # define torchvision transforms as class attribute
    _transforms = {
        'train': transforms.Compose([
            transforms.Scale(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
        'val': transforms.Compose([
            transforms.Scale(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
    }
    _transforms['test'] = _transforms['val']

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        image = Image.open(
            os.path.join(
                self.path_to_images,
                self.df.index[idx]))
        image = image.convert('RGB')

        label = numpy.zeros(len(self.labels), dtype=int)
        for i in range(0, len(self.labels)):
            if self.labels[i] != "N/A":
                if(self.df[self.labels[i].strip()].iloc[idx].astype('int') > 0):
                    label[i] = self.df[self.labels[i].strip()
                                       ].iloc[idx].astype('int')

        if self.transform:
            image = self.transform(image)

        return (image, label, self.df.index[idx], ['None'])

    def get_all_labels(self):
        '''
        Return a numpy array of shape (n_samples, n_dimensions) that includes 
        the ground-truth labels for all samples.
        '''
        ndim = len(self.labels)
        nsamples = len(self)
        output = numpy.zeros((nsamples, ndim))
        for isample in range(len(self)):
            output[isample] = self[isample][1]
        return output
            
class CheXpertDataset(CXRDataset):
    def __init__(
            self,
            fold,
            include_lateral=False,
            random_state=30493):
        '''
        Create a dataset of the CheXPert images for use in a PyTorch model.

        Args:
            fold (str): The shard of the CheXPert data that the dataset should
                contain. One of either 'train', 'val', or 'test'. The 'test'
                fold corresponds to the images specified in 'valid.csv' in the 
                CheXPert data, while the the 'train' and 'val' folds
                correspond to disjoint subsets of the patients in the 
                'train.csv' provided with the CheXpert data.
            random_state (int): An integer used to see generation of the 
                train/val split from the patients specified in the 'train.csv'
                file provided with the CheXpert dataset. Used to ensure 
                reproducability across runs.
            include_lateral (bool): If True, include the lateral radiograph
                views in the dataset. If False, include only frontal views.
        '''

        self.transform = self._transforms[fold]
        self.path_to_images = "../data/CheXpert/"
        self.fold = fold

        # Load files containing labels, and perform train/valid split if necessary
        if fold == 'train' or fold == 'val':
            trainvalpath = os.path.join(
                    self.path_to_images, 
                    'CheXpert-v1.0-small/train.csv')
            self.df = pandas.read_csv(trainvalpath)
            self.df.set_index("Path", inplace=True)
            
            if not include_lateral:
                self.df = self.df[self.df['Frontal/Lateral'] == 'Frontal']
            
            train, val = grouped_split(
                self.df,
                random_state=random_state,
                test_size=0.05)
            if fold == 'train':
                self.df = train
            else:
                self.df = val
        elif fold == 'test':
            testpath = os.path.join(
                    self.path_to_images, 
                    'CheXpert-v1.0-small/valid.csv')
            self.df = pandas.read_csv(testpath)
            self.df.set_index("Path", inplace=True)
            if not include_lateral:
                self.df = self.df[self.df['Frontal/Lateral'] == 'Frontal']
        else:
            raise ValueError("Invalid fold: {:s}".format(str(fold)))
            
        self.labels = [
            'Enlarged Cardiomediastinum',
            'Cardiomegaly',
            'Lung Opacity',
            'Lung Lesion',
            'Edema',
            'Consolidation',
            'Pneumonia',
            'Atelectasis',
            'Pneumothorax',
            'Pleural Effusion',
            'Pleural Other',
            'Fracture',
            'Support Devices']
    
    def __getitem__(self, idx):

        image = Image.open(
            os.path.join(
                self.path_to_images,
                self.df.index[idx]))
        image = image.convert('RGB')

        label = numpy.zeros(len(self.labels), dtype=int)
        for i in range(0, len(self.labels)):
            if self.labels[i] != "N/A":
                if(self.df[self.labels[i].strip()].iloc[idx].astype('int') > 0):
                    label[i] = self.df[self.labels[i].strip()
                                       ].iloc[idx].astype('int')

        if self.transform:
            image = self.transform(image)

        appa = self.df['AP/PA'][idx] == 'AP'

        return (image, label, self.df.index[idx], appa)
    
    def get_all_view_labels(self):
        '''
        Return a numpy array of shape (n_samples, n_dimensions) that includes 
        the ground-truth labels for all samples.
        '''
        ndim = 1
        nsamples = len(self)
        output = numpy.zeros((nsamples, 1))
        for isample in range(len(self)):
            output[isample] = self.df['AP/PA'][isample] == 'AP'
        return output
        
class MIMICDataset(CXRDataset):
    def __init__(
            self,
            fold,
            include_lateral=False,
            random_state=30493):
        '''
        Create a dataset of the MIMIC-CXR images for use in a PyTorch model.

        Args:
            fold (str): The shard of the MIMIC-CXR data that the dataset should
                contain. One of either 'train', 'val', or 'test'.
            random_state (int): An integer used to see generation of the 
                train/val split
            include_lateral (bool): If True, include the lateral radiograph
                views in the dataset. If False, include only frontal views.
        '''
        self.transform = self._transforms[fold]
        self.path_to_images = "../data/MIMIC-CXR/"
        self.fold = fold

        # Load files containing labels, and perform train/valid split if necessary
        if fold == 'train' or fold == 'val':
            trainvalpath = os.path.join(self.path_to_images, 'train.csv')
            self.df = pandas.read_csv(trainvalpath)
            self.df.set_index("path", inplace=True)
            if not include_lateral:
                self.df = self.df[self.df['view'] == 'frontal']
            train, val = grouped_split(
                    self.df,
                    random_state=random_state,
                    test_size=0.05)
            if fold == 'train':
                self.df = train
            else:
                self.df = val
        elif fold == 'test':
            testpath = os.path.join(self.path_to_images, 'valid.csv')
            self.df = pandas.read_csv(testpath)
            self.df.set_index("path", inplace=True)
            if not include_lateral:
                self.df = self.df[self.df['view'] == 'frontal']
        else:
            raise ValueError("Invalid fold: {:s}".format(str(fold)))

        self.labels = [
            'Enlarged Cardiomediastinum',
            'Cardiomegaly',
            'Lung Opacity',
            'Lung Lesion',
            'Edema',
            'Consolidation',
            'Pneumonia',
            'Atelectasis',
            'Pneumothorax',
            'Pleural Effusion',
            'Pleural Other',
            'Fracture',
            'Support Devices']