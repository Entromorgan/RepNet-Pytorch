import torch
import cv2
import glob
import os
import time

import numpy as np
import pandas as pd

from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from random import randint, choice, shuffle
from torch.utils.data import Dataset, DataLoader, ConcatDataset
# import torch.nn.functional as F


# returns a 64 length array that goes low->mid->high->mid
def getRandomTransformParameter(high, mid, low, length=64):
    retarr = []
    midpos = randint(length//4, length//2)
    highpos = randint(length//2, 3*length//4)
    
    retarr = list(np.linspace(start=low, stop=mid, num=midpos))
    retarr.extend(list(np.linspace(start=mid, stop=high, num=highpos-midpos)))
    retarr.extend(list(np.linspace(start=high, stop=mid, num=length - highpos)))
    
    retarr = np.array(retarr)
    retarr = retarr[::choice([-1, 1])]
    return retarr


def randomTransform(frames):
    scaleParams = getRandomTransformParameter(0.9, 0.75, 0.5)
    zRotateParams = getRandomTransformParameter(45, 0, -45)
    xRotateParams = getRandomTransformParameter(0.2, 0.0, -0.2, 32)
    yRotateParams = getRandomTransformParameter(0.2, 0.0, -0.2, 32)
    
    h, w, c = frames[0].shape
    erParams = [randint(0,h-h/2), randint(0,w-w/2), h//2, w//2]
    erVal = getRandomTransformParameter(1.0, 0.5, 0.0)
    horizTransParam = (h/4)*getRandomTransformParameter(0.4, 0.0, -0.4)
    verticalTransParam = (w/4)*getRandomTransformParameter(0.4, 0.0, -0.4)
    
    newFrames = []

    for i, frame in enumerate(frames):
        img = Image.fromarray(frame)
        preprocess = transforms.Compose([
            transforms.Resize((112, 112), 2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        frame = preprocess(img).unsqueeze(0)
        #frame = transforms.functional.erase(frame, erParams[0] , erParams[1], erParams[2], erParams[3], erVal[i])

        frame = transforms.functional.affine(frame,
                                             zRotateParams[i],
                                             [horizTransParam[i], verticalTransParam[i]],
                                             scaleParams[i],
                                             [0.0, 0.0],
                                             0)
        newFrames.append(frame)
    
    frames = torch.cat(newFrames)
        
    return frames


def getCombinedDataset(dfPath, videoDir, videoPrefix):
    df = pd.read_csv(dfPath)
    path_prefix = videoDir + '/' + videoPrefix
    
    files_present = []
    for i in range(0, len(df)):
        path_to_video = path_prefix + str(i) + '.mp4'
        if os.path.exists(path_to_video):
            files_present.append(i)

    df = df.iloc[files_present]
    
    miniDatasetList = []
    for i in range(0, len(df)):
        dfi = df.iloc[[i]]
        path_to_video = path_prefix + str(dfi.index.item()) +'.mp4'
        miniDatasetList.append(miniDataset(dfi, path_to_video))
        
    megaDataset = ConcatDataset(miniDatasetList)
    return megaDataset


"""Creates one sequence from each video"""
class miniDataset(Dataset):
    
    def __init__(self, df, path_to_video, path_to_merged_video='/workspace/data/synthvids/train*.mp4'):
        self.path = path_to_video
        self.df = df.reset_index()
        self.count = self.df.loc[0, 'count']
        self.m_path = path_to_merged_video

        
    def getFrames(self, path = None):
        """returns frames"""
        frames = []
        if path is None:
            path = self.path
        
        cap = cv2.VideoCapture(path)
        while cap.isOpened():
            ret, frame = cap.read()
            if ret is False:
                break
            
            img = Image.fromarray(frame)
            frames.append(img)
        
        cap.release()
        
        return frames

    
    def __getitem__(self, index):
        curFrames = self.getFrames()
        
        output_len = min(len(curFrames), randint(44, 64))
                
        newFrames = []
        for i in range(1, output_len + 1):
            newFrames.append(curFrames[i * len(curFrames)//output_len  - 1])

        a = randint(0, 64 - output_len)
        b = 64 - output_len - a
        
        randpath = choice(glob.glob(self.m_path))
        randFrames = self.getFrames(randpath)
        newRandFrames = []
        for i in range(1, a + b + 1):
            newRandFrames.append(randFrames[i * len(randFrames)//(a+b)  - 1])

        
        same = np.random.choice([0, 1], p = [0.5, 0.5])
        if same:
            finalFrames = [newFrames[0] for i in range(a)]
            finalFrames.extend( newFrames )        
            finalFrames.extend([newFrames[-1] for i in range(b)] )
        else:
            finalFrames = newRandFrames[:a]
            finalFrames.extend( newFrames )        
            finalFrames.extend( newRandFrames[a:] )

        Xlist = []
        for img in finalFrames:
            preprocess = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
            frameTensor = preprocess(img).unsqueeze(0)
            Xlist.append(frameTensor)
        
        Xlist = [Xlist[i] if a<i<(64-b) else torch.nn.functional.dropout(Xlist[i], 0.2) for i in range(64)]  
        X = torch.cat(Xlist)
        y = [0 for i in range(0,a)]
        y.extend([output_len/self.count if 1<output_len/self.count<32 else 0 for i in range(0, output_len)])
        
        y.extend( [ 0 for i in range(0, b)] )
        y = torch.FloatTensor(y).unsqueeze(-1)
        
        return X, y
        
    def __len__(self):
        return 1

    
class dataset_with_indices(Dataset):
    """
    Modifies the given Dataset class to return a tuple data, target, index
    instead of just data, target.
    """
    def __init__(self, ds):
        self.ds = ds

        
    def __getitem__(self, index):
        X, y = self.ds[index]
        return X, y, index
    
    
    def getPeriodDist(self):
        arr = np.zeros(32,)
        
        for i in tqdm(range(self.__len__())):
            _, p,_ = self.__getitem__(i)
            per = max(p)
            arr[per] += 1
        return arr
    
    
    def __len__(self):
        return len(self.ds)


class BlenderDataset(Dataset):
    
    def __init__(self, parentDir, vidDir, annotDir, frame_per_vid):    
        self.vidPath = parentDir + '/' + vidDir
        self.annotPath = parentDir + '/' + annotDir

        self.videos =  list(glob.glob(self.vidPath + '/*.mkv'))
        shuffle(self.videos)
        self.frame_per_vid = frame_per_vid

        
    def getFrames(self, path):
        """returns frames"""
        frames = []
        cap = cv2.VideoCapture(path)
        while cap.isOpened():
            ret, frame = cap.read()
            if ret is False:
                break
            
            img = Image.fromarray(frame)
            frames.append(img)
        
        cap.release()
        return frames

    
    def __getitem__(self, index):
        parts = 64//self.frame_per_vid
        nindex = index//parts


        videoFile = self.videos[nindex]
        curFrames = self.getFrames(videoFile)

        sz = curFrames[0].size
        curFrames[0] = Image.new("RGB", sz, (0,0,0))
        curFrames[-1] = Image.new("RGB", sz, (0,0,0))

        Xlist = []
        for img in curFrames:
        
            preprocess = transforms.Compose([
            transforms.Resize((182, 182)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225])])
            frameTensor = preprocess(img).unsqueeze(0)
            Xlist.append(frameTensor)
        

        ipart = nindex % parts
        X = torch.cat(Xlist[ipart*self.frame_per_vid:(ipart+1)*self.frame_per_vid])

        annot = self.annotPath + '/' + self.videos[nindex][len(self.vidPath) + 1:-4]
        labels = glob.glob(annot + '/*')

        y = np.load(labels[0])
        y[0] = 0
        y[-1] = 0
        for i in range(len(y)):
            if y[i] >= 32:
                y[i] = 0
        y = torch.FloatTensor(y[ipart*self.frame_per_vid:(ipart+1)*self.frame_per_vid]).unsqueeze(-1)
        
        assert X.shape[0] == self.frame_per_vid, str(X.shape[0]) + " "+str(self.frame_per_vid)
        assert(y.shape[0] == self.frame_per_vid)

        return X, y
        
        
    def __len__(self):
        return len(self.videos) * (64//self.frame_per_vid)


class SyntheticDataset(Dataset):
    
    def __init__(self, videoPath, filename, extension, length):
        self.sourcePath = videoPath + '/' + filename + '.' + extension
        self.length = length

        
    def __getitem__(self, index):
        X, periodLength, period = self.generateRepVid()
        return X, periodLength
    
    
    def getPeriodDist(self, samples):
        arr = np.zeros(32,)
        
        for i in tqdm(range(samples)):
            _, _, p = self.generateRepVid()
            arr[p] += 1
        return arr
    
    
    def getNFrames(self, frames, n):  
        newFrames = []
        for i in range(1, n + 1):
            newFrames.append(frames[i * len(frames)//n  - 1])
            
        assert(len(newFrames) == n)
        return newFrames
            
        
    def generateRepVid(self):
        while True:
            path = choice(glob.glob(self.sourcePath))
            assert os.path.exists(path), "No file with this pattern exist" + self.sourcePath

            cap = cv2.VideoCapture(path)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total > 64:
                break
            else:
                os.remove(path)
        
        mirror = np.random.choice([0, 1], p = [0.8, 0.2])
        halfperiod = randint(2 , 31) // (mirror + 1)
        period = (mirror + 1) * halfperiod
        count = randint(max(2, 16//period), 64//(period))
        
        clipDur = randint(min(total//(64/period - count + 1), max(period, 30)), 
                          min(total//(64/period - count + 1), 60))

        repDur = count * clipDur
        noRepDur =  int((64 / (period*count) - 1) * repDur)
         
        assert(noRepDur >= 0)
        begNoRepDur = randint(0,  noRepDur)
        endNoRepDur = noRepDur - begNoRepDur
        totalDur = noRepDur + repDur
            
        startFrame = randint(0, total - (clipDur + noRepDur))
        cap.set(cv2.CAP_PROP_POS_FRAMES, startFrame)
        
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if ret is False or len(frames) == clipDur + noRepDur:
                break
            frame = cv2.resize(frame , (112, 112), interpolation = cv2.INTER_AREA)
            frames.append(frame)
        
        cap.release()
        
        numBegNoRepFrames = begNoRepDur*64//totalDur
        periodLength = np.zeros((64, 1))
        begNoRepFrames = self.getNFrames(frames[:begNoRepDur], numBegNoRepFrames)
        finalFrames = begNoRepFrames
        
        repFrames = frames[begNoRepDur : -endNoRepDur]
        repFrames.extend(repFrames[::-1])

        if len(repFrames) >= period:
            curf = numBegNoRepFrames
            for i in range(count):
                if period > 18:
                    noisyPeriod = np.random.choice([max(period-1, 2), period, min(31, period + 1)])
                    noisyPeriod = min(noisyPeriod, 64 - curf)
                else:
                    noisyPeriod = period
                noisyFrames = self.getNFrames(repFrames, noisyPeriod)
                finalFrames.extend(noisyFrames)

                for p in range(noisyPeriod):
                    
                    try:
                        periodLength[curf] = noisyPeriod
                    except: 
                        print(curf, numBegNoRepFrames, totalDur, begNoRepDur)
                    assert(noisyPeriod < 32)
                    curf+=1
                                                
        else:
            period = 0
            
        numEndNoRepFrames = 64 - len(finalFrames) 
        endNoRepFrames = self.getNFrames(frames[-endNoRepDur:], numEndNoRepFrames)
        finalFrames.extend(endNoRepFrames)
        
        frames = randomTransform(finalFrames)
        
        numBegNoRepFrames = begNoRepDur*64//totalDur
        if count == 1:
            numEndNoRepFrames = 64 - numBegNoRepFrames
            period = 0
            
        #assert(len(frames) == 64)
        
        #frames = F.dropout(frames, p = 0.1)
        periodLength = torch.LongTensor(periodLength)
        
        return frames, periodLength, period

    
    def __len__(self):
        return self.length
