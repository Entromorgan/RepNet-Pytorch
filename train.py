import torch
from random import shuffle
from Model import RepNet
from torch.utils.data import ConcatDataset, Subset
from Dataset import getCombinedDataset, SyntheticDataset, BlenderDataset
from train_utils import training_loop

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
# device = torch.device("cpu")

print("Initializing...")

frame_per_vid = 64
# multiple = False

trainDatasetC = getCombinedDataset('countix/countix_train.csv', '/workspace/data/trainvids', 'train')# , frame_per_vid=frame_per_vid, multiple=multiple)
trainDatasetS3 = SyntheticDataset('/workspace/data/synthvids', 'train*', 'mp4', 3000) # , frame_per_vid=frame_per_vid)
trainDatasetB = BlenderDataset('/workspace/data/blendervids', 'videos', 'annotations', frame_per_vid)
trainList = [trainDatasetC, trainDatasetS3] #, trainDatasetB]
shuffle(trainList)
trainDataset = ConcatDataset(trainList)

testDatasetC = getCombinedDataset('countix/countix_test.csv', '/workspace/data/testvids', 'test')# , frame_per_vid=frame_per_vid, multiple=multiple)
testDatasetS = SyntheticDataset('/workspace/data/synthvids', 'train*', 'mp4', 2000) # , frame_per_vid=frame_per_vid)
testList = [testDatasetC, testDatasetS]
shuffle(testList)
testDataset = ConcatDataset(testList)

model = RepNet(frame_per_vid)
model = model.to(device)

"""Testing the training loop with sample datasets"""
 
sampledTrain = Subset(trainDataset, range(0, len(trainDataset)))
sampledTest = Subset(testDataset, range(0, len(testDataset)))

print("Initialization finished.")

print("Start training...")

trLoss, valLoss = training_loop(n_epochs=10,
                                device = device,
                                model = model,
                                train_set = sampledTrain,
                                val_set = sampledTest,
                                batch_size = 1,
                                lr = 6e-5,
                                ckpt_name = 'x3dbb',
                                use_count_error = True,
                                saveCkpt = True,
                                train = True,
                                validate = True,
                                lastCkptPath = None)

print("Training finished.")
