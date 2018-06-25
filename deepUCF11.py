import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
import os, math
from torch.utils.data import Dataset, DataLoader
import pandas as pd

parser = argparse.ArgumentParser(description='deepVideo UCF11')
parser.add_argument('-m', '--model', metavar='model numebr', default=25,
                    help='model number over 25 cv')
parser.add_argument('-r', '--run', metavar='run numebr', default=2,
                    help='run numeber experiment')
parser.add_argument('-lr', '--learning-rate', default=0.0002, type=float,
                    metavar='LR', help='initial learning rate (default: 0.0005)')
parser.add_argument('--weight-decay', '-wd', default=5e-4, type=float,
                    metavar='Wdecay', help='weight decay (default: 1e-4)')
parser.add_argument('--epochs', '-e', default=80, type=int, metavar='E',
                    help='number of total epochs to run')
parser.add_argument('--batch-size', '-b', default=32, type=int,
                    metavar='B', help='mini-batch size (default: 32)')


class ResFeatureDataset(Dataset):

    def __init__(self, folder_path, list_files, dict_labels, transforms=None):
        self.folder_path = folder_path
        self.transforms = transforms
        self.features = list_files
        self.labels = dict_labels

    def __getitem__(self, item):
        path_ = self.features[item]
        path_ = path_.replace("..",".")
        path = os.path.join(self.folder_path, path_)
        feature = np.load(path).astype(np.float32)
        label = self.labels[path_[2:6]]
        # transform the sequence in a base 2 sequence (first 128) with padding if using fft!
        if feature.shape[0] > 128:
            feature = feature[:128, :]
        elif feature.shape[0] < 128:
            missing = 128 - feature.shape[0]
            if missing < feature.shape[0]:
                new = feature[-(missing+1):-1].copy()
                new = np.flip(new, 0)
                feature = np.append(feature, new, axis=0)
            else:
                rep = int(np.ceil(128/feature.shape[0]))
                repeated = np.tile(feature, reps=(rep, 1))
                feature = repeated[:128, :]
            #zeros = np.zeros([128 - feature.shape[0], 2048], dtype=np.float32)
            #feature = np.append(feature, zeros, axis=0)
        '''# transform the sequence in a base 2 sequence (middle 128) with padding if using fft!
        if feature.shape[0] > 128:
            middle = int(np.floor(feature.shape[0]/2))
            min, max = middle-64, middle+64
            feature = feature[min:max, :]
        elif feature.shape[0] < 128:
            zeros = np.zeros([int((128-feature.shape[0])/2), 2048], dtype=np.float64)
            feature = np.append(zeros, feature, axis=0)
            zeros = np.zeros([128-feature.shape[0], 2048], dtype=np.float64)
            feature = np.append(feature, zeros, axis=0)
        '''
        ''' # initial trasform to have the upper log2 
        if np.log2(feature.shape[0]) % 1 > 0:
            num_sample = 2**math.ceil(math.log2(feature.shape[0]))  #2**math.ceil(math.log2(x))
            zeros = np.zeros([num_sample - feature.shape[0], 2048], dtype=np.float64)
            feature = np.append(feature, zeros, axis=0)
        '''
        return feature, label

    def __len__(self):
        return len(self.features)


class DeepFFT(nn.Module):
    def __init__(self, num_freq=1, num_classes=11):
        super(DeepFFT, self).__init__()
        self.num_freq = num_freq

        #self.bn0 = nn.BatchNorm1d(2048,track_running_stats=False).cuda()

        #self.convRes1 = nn.Conv1d(2048, 2048, 3, stride=2, padding=1).cuda()
        #self.convRes1_bn = nn.BatchNorm1d(2048,track_running_stats=False).cuda()
        self.convRes2 = nn.Conv1d(2048, 2048, 3, stride=2, padding=1).cuda()
        self.convRes2_bn = nn.BatchNorm1d(2048,track_running_stats=False).cuda()
        self.convRes3 = nn.Conv1d(2048, 1024, 3, stride=2, padding=0).cuda()
        self.convRes3_bn = nn.BatchNorm1d(1024,track_running_stats=False).cuda()
        self.convRes4 = nn.Conv1d(1024, 512, 3, stride=2, padding=0).cuda()
        self.convRes4_bn = nn.BatchNorm1d(512,track_running_stats=False).cuda()
        self.convRes5 = nn.Conv1d(512, 512, 3, stride=2, padding=0).cuda()
        self.convRes5_bn = nn.BatchNorm1d(512,track_running_stats=False).cuda()
        self.convRes6 = nn.Conv1d(512, 256, 3, stride=2, padding=0).cuda()
        self.convRes6_bn = nn.BatchNorm1d(256,track_running_stats=False).cuda()
        #self.maxpool1 = nn.MaxPool1d(2, stride=2).cuda()

        self.convFreq1 = nn.Conv1d(2048, 2048, 3, stride=2).cuda()
        self.convFreq1_bn = nn.BatchNorm1d(2048,track_running_stats=False).cuda()
        self.convFreq2 = nn.Conv1d(2048, 1024, 3, stride=2).cuda()
        self.convFreq2_bn = nn.BatchNorm1d(1024,track_running_stats=False).cuda()
        self.convFreq3 = nn.Conv1d(1024, 512, 3, stride=2).cuda()
        self.convFreq3_bn = nn.BatchNorm1d(512,track_running_stats=False).cuda()
        self.convFreq4 = nn.Conv1d(512, 256, 3, stride=2).cuda()
        self.convFreq4_bn = nn.BatchNorm1d(256,track_running_stats=False).cuda()
        #self.maxpool2 = nn.MaxPool2d(2, stride=2).cuda()

        # self.fc = nn.Linear(num_freq*2*2048+2048, num_classes).cuda()  # *2 if real+imaginary parts + 2048 first_frame

        self.fc = nn.Linear(1024, num_classes).cuda()  # 1536  #7680

        #self.fc1 = nn.Linear(1024, 22).cuda()
        #self.dropout = nn.Dropout(p=0.1)
        #self.fc2 = nn.Linear(22, num_classes).cuda()

    def forward(self, x):
        # x = F.relu(self.convRes1_bn(self.convRes1(x)))
        # x = self.bn0(x)

        # fft_r, fft_i = fft.FFT_torch(x.double())

        im = torch.zeros_like(x).view(x.shape[0], x.shape[1], x.shape[2], 1)
        x2 = x.view(x.shape[0], x.shape[1], x.shape[2], 1)
        xc = torch.cat([x2, im], dim=3)
        fr = torch.fft(xc, signal_ndim=1)  # tensor with last dimension 2 (real+imag) , 1 is signal dimension

        fft_r = fr[:, :, :self.num_freq, 0]
        fft_i = fr[:, :, :self.num_freq, 1]

        fft = torch.sqrt(fft_r**2+fft_i**2)

        #x = F.relu(self.convRes1_bn(self.convRes1(x)))
        x = F.relu(self.convRes2_bn(self.convRes2(x)))
        x = F.relu(self.convRes3_bn(self.convRes3(x)))
        x = F.relu(self.convRes4_bn(self.convRes4(x)))
        x = F.relu(self.convRes5_bn(self.convRes5(x)))
        x = F.relu(self.convRes6_bn(self.convRes6(x)))

        xfft = F.relu(self.convFreq1_bn(self.convFreq1(fft)))
        xfft = F.relu(self.convFreq2_bn(self.convFreq2(xfft)))
        xfft = F.relu(self.convFreq3_bn(self.convFreq3(xfft)))
        xfft = F.relu(self.convFreq4_bn(self.convFreq4(xfft)))

        x = torch.cat([x, xfft], dim=2)

        x = self.fc(x.view(x.shape[0], -1))
        #x = self.dropout(self.fc1(x.view(x.shape[0], -1)))
        #x = self.fc2(x)

        return x


def split_crossval25(folder, seed=3):
    list_features = os.listdir(folder)
    # np.random.shuffle(list_features)
    groups = {}
    for f in list_features:
        if int(f[-9:-7]) in groups:
            groups[int(f[-9:-7])].append(f)
        else:
            groups[int(f[-9:-7])] = [f]
    return groups


def train_model(run, model_num, folder, dict_labels, freq=15, batch_s=10, lr=0.001, wd=0.002, epochs=8):

    if not os.path.exists("Results_UCF11/run"+str(run)+"/"+str(model_num)):
        os.makedirs("Results_UCF11/run"+str(run)+"/"+str(model_num))

    f = open("Results_UCF11/run"+str(run)+"/"+str(model_num)+"/training_model_%d"%model_num, "w+")  # write result training
    f.write("FREQUENCIES USED %d :\n\n" % freq)

    groups = split_crossval25(folder, seed=3)

    test = groups[model_num].copy()
    del groups[model_num]  # delete test group from training
    train = list()
    for key, value in groups.items():
        for v in value:
            train.append(v)

    dataset = ResFeatureDataset(folder, train, dict_labels)
    dataloader = DataLoader(dataset, batch_size=batch_s, num_workers=0)
    dataset_test = ResFeatureDataset(folder, test, dict_labels)
    dataloader_test = DataLoader(dataset_test, batch_size=batch_s, num_workers=2)

    model = DeepFFT(num_freq=freq)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1)
    best_acc = 0

    for epoch in range(epochs):
        print("Model: %d  --  Epoch: %d of %d:" % (model_num, epoch+1, epochs))
        loss_epoch = 0.0; total = 0; correct = 0
        model.train()
        dataloader_iter = iter(dataloader)
        for it, data in enumerate(dataloader_iter):
            features, labels = data
            features.transpose_(1, 2)
            features = Variable(features, requires_grad=True).cuda()#.double()
            labels = Variable(labels).cuda()

            output = model(features)
            loss = criterion(output, labels)
            loss_epoch += loss.item()

            _, predicted = torch.max(output.data, 1)
            total += len(labels)
            correct += (predicted == labels.data).sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #if (it*batch_s)%300 == 0:
                #print("Training model %d -- epoch %d (%d/%d) %3.2f%%" % (model_num,epoch+1,it*batch_s,len(train),100*(it*batch_s/len(train))))

        train_loss = loss_epoch  #/(len(train) / batch_s)
        train_accuracy = 100 * float(correct.tolist() / total)

        print("Model %d epoch %d -- Training Loss: %6.1f - Accuracy: %3.3f%%" % (model_num, epoch+1, train_loss, train_accuracy))
        f.write("Epoch %d -- Training Loss: %6.1f - Accuracy: %3.3f%%\t" % (epoch+1, train_loss, train_accuracy))

        # EVALUATION
        loss_eval = 0.0; total = 0; correct = 0
        model.eval()
        dataloader_test_iter = iter(dataloader_test)
        for it, data in enumerate(dataloader_test_iter):
            features, labels = data
            if features.shape[0] == 1:
                continue
            features.transpose_(1, 2)
            features = Variable(features, requires_grad=False).cuda()#.double()
            labels = Variable(labels).cuda()

            output = model(features)
            loss = criterion(output, labels)
            loss_eval += loss.item()

            _, predicted = torch.max(output.data, 1)
            total += len(labels)
            correct += (predicted == labels.data).sum()  # accuracy = correct/total

            #if (it*batch_s)%180 == 0:
                #print("Evaluation model %d epoch %d (%d/%d) %3.2f%%" % (model_num,epoch+1,it*batch_s,len(test),100*(it*batch_s/len(test))))

        eval_loss = loss_eval  #/(len(test) / batch_s)
        scheduler.step(eval_loss)
        eval_accuracy = 100 * float(correct.tolist() / total)

        if eval_accuracy > best_acc:
            best_acc = eval_accuracy
            #torch.save(model, "Results_UCF11/run"+str(run)+"/"+str(model_num) + "/model_"+str(model_num))

        print("model %d epoch %d -- Evaluation Loss: %6.1f - Accuracy: %3.3f%%" % (model_num, epoch+1, eval_loss, eval_accuracy))
        f.write("Evaluation Loss: %6.1f - Accuracy: %3.3f%%\n" % (eval_loss, eval_accuracy))

    f.write("\nBest Accuracy model: %3.3f%%"%best_acc)
    f.close()
    d = open("Results_UCF11/run"+str(run)+"/"+str(model_num)+"/best_accuracy_model_%d"%model_num + ".csv", "w+")
    d.write("%3.4f"%best_acc)


if __name__ == '__main__':

    dict_labels = {"biki": 0, "divi": 1, "golf": 2, "jugg": 3, "jump": 4, "ridi": 5, "shoo": 6, "spik": 7, "swin": 8,
                   "tenn": 9, "walk": 10}
    global args
    args = parser.parse_args()

    #if not os.path.isdir("Results_UCF11"):
        #os.mkdir("Results_UCF11")

    folder = "features"

    batch_s = args.batch_size
    freq = 46
    learning_rate = args.learning_rate
    weight_decay = args.weight_decay
    model_number = int(args.model)
    run = int(args.run)

    if model_number == 1:
        d = open("Results_UCF11/run"+str(run)+"/specs", "w+")
        d.write("Run %d: learning rate = %1.6f - weight_decay = %1.5f - freq = %d\n\n"%(run,learning_rate,weight_decay,freq))
        d.close()

    train_model(run, model_number, folder, dict_labels, freq, batch_s, lr=learning_rate, wd=weight_decay, epochs=args.epochs)

    if model_number == 25:
        acc = list()
        for dir in os.listdir("Results_UCF11/run"+str(run)):
            if os.path.exists("Results_UCF11/run"+str(run)+"/"+dir+"/best_accuracy_model_%s"%dir + ".csv"):
                acc.append(pd.read_csv("Results_UCF11/run"+str(run)+"/"+dir+"/best_accuracy_model_%s"%dir + ".csv",header=None))
        totacc = pd.concat(acc, ignore_index=True)
        mean = totacc.sum()/len(totacc)
        d = open("Results_UCF11/run"+str(run)+"/mean_accuracy_and_specs", "w+")
        d.write("Run %d: learning rate = %1.5f - weight_decay = %1.4f - freq = %d\n\n"%(run,learning_rate,weight_decay,freq))
        d.write("Mean Accuracy LOOCV: %3.3f%%" % mean)
        d.close()
