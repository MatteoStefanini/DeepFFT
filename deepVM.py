import argparse
import numpy as np
from torch.utils.data import Dataset, DataLoader
import json, math, os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from sklearn.metrics import precision_recall_fscore_support
import pandas as pd
from tensorboardX import SummaryWriter


columns_set = ['small', 'medium', 'complete']
model_names = ['DeepConv', 'DeepFFT', 'DeepMix', 'DeepMix2']
parser = argparse.ArgumentParser(description='deepVM training PyTorch')
parser.add_argument('-data', metavar='Folder', default='VMdata/',
                    help='path to dataset folder')
parser.add_argument('--columns', '-c', metavar='Cols_set', default='complete',
                    choices=columns_set, help='columns set: ' + ' | '.join(columns_set) + ' (default: complete)')
parser.add_argument('--window', '-w', default=128, type=int,
                    help='number of timestep for each window')
parser.add_argument('--arch', '-a', metavar='ARCH', default='DeepFFT',
                    choices=model_names, help='model architecture: ' + ' | '.join(model_names) + ' (default: DeepMix)')
parser.add_argument('--epochs', '-e', default=110, type=int, metavar='E',
                    help='number of total epochs to run')
parser.add_argument('--batch-size', '-b', default=64, type=int,
                    metavar='B', help='mini-batch size (default: 64)')
parser.add_argument('-lr', '--learning-rate', default=0.00002, type=float,
                    metavar='LR', help='initial learning rate (default: 0.0005)')
parser.add_argument('--weight-decay', '-wd', default=5e-4, type=float,
                    metavar='Wdecay', help='weight decay (default: 1e-4)')
parser.add_argument('--threshold', '-th', default=0, type=int,
                    help='percentage threshold to consider valid data')
parser.add_argument('--freqs', '-fq', default=1., action="store", dest="freqs", type=float,
                    help='perentage of frequences to take into account (only with DeepFFT ARCH)')
parser.add_argument('--workers', '-j',  default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 0)')

					
class VMDataset(Dataset):
    def __init__(self, data, kind):
        super(VMDataset, self).__init__()
        self.data = data
        self.kind = kind

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        item = self.data[i]
        return torch.from_numpy(np.array(item.drop('label', axis=1), dtype='float32')).requires_grad_(), \
               torch.from_numpy(np.array(item.iloc[0]['label'], dtype='float32'))

    def __str__(self):
        s = "[Kind: {0} Lenght: {1})]".format(self.kind, str(len(self.data)))
        return s


class DeepConv(nn.Module):
    def __init__(self, channels=16, window=128, num_classes=2):
        super(DeepConv, self).__init__()
        self.num_classes = num_classes
        self.channels = channels

        self.batch_norm0 = nn.BatchNorm1d(channels,track_running_stats=False).cuda()

        self.conv1 = nn.Conv1d(channels, 2*channels, 3, stride=2, padding=1).cuda()
        self.conv1_bn = nn.BatchNorm1d(2*channels,track_running_stats=False).cuda()
        self.conv2 = nn.Conv1d(2*channels, 2*channels, 3, stride=2, padding=1).cuda()
        self.conv2_bn = nn.BatchNorm1d(2*channels,track_running_stats=False).cuda()
        self.conv3 = nn.Conv1d(2*channels, 4*channels, 3, stride=2, padding=1).cuda()
        self.conv3_bn = nn.BatchNorm1d(4*channels,track_running_stats=False).cuda()
        self.conv4 = nn.Conv1d(4*channels, 8*channels, 3, stride=2, padding=1).cuda()
        self.conv4_bn = nn.BatchNorm1d(8*channels,track_running_stats=False).cuda()
        #self.conv5 = nn.Conv1d(8*channels, 8*channels, 3, stride=2, padding=1).cuda()
        #self.conv5_bn = nn.BatchNorm1d(8*channels).cuda()

        self.timestep = int(window/16) if int(window/16) > 0 else 1
        self.fc = nn.Linear(8*channels*self.timestep, num_classes).cuda()

    def forward(self, x):
        x = self.batch_norm0(x)
        x = F.relu(self.conv1_bn(self.conv1(x)))
        x = F.relu(self.conv2_bn(self.conv2(x)))
        x = F.relu(self.conv3_bn(self.conv3(x)))
        x = F.relu(self.conv4_bn(self.conv4(x))).view(x.shape[0], -1)
        #x = F.relu(self.conv5_bn(self.conv5(x))).view(x.shape[0], -1)

        x = self.fc(x)

        return x


class DeepFFT(nn.Module):
    def __init__(self, channels=16, window=128, num_classes=2):
        super(DeepFFT, self).__init__()
        self.num_classes = num_classes
        self.channels = channels
        self.num_freq = int(window*args.freqs)

        self.batch_norm0 = nn.BatchNorm1d(channels,track_running_stats=False).to(device)#.cuda()

        self.conv1 = nn.Conv1d(2*channels, 4*channels, 3, stride=2, padding=1).cuda()
        self.conv1_bn = nn.BatchNorm1d(4*channels,track_running_stats=False).cuda()
        self.conv2 = nn.Conv1d(4*channels, 4*channels, 3, stride=2, padding=1).cuda()
        self.conv2_bn = nn.BatchNorm1d(4*channels,track_running_stats=False).cuda()
        self.conv3 = nn.Conv1d(4*channels, 8*channels, 3, stride=2, padding=1).cuda()
        self.conv3_bn = nn.BatchNorm1d(8*channels,track_running_stats=False).cuda()
        self.conv4 = nn.Conv1d(8*channels, 8*channels, 3, stride=2, padding=1).cuda()
        self.conv4_bn = nn.BatchNorm1d(8*channels,track_running_stats=False).cuda()
        #self.conv5 = nn.Conv1d(16*channels, 16*channels, 3, stride=2, padding=1).cuda()
        #self.conv5_bn = nn.BatchNorm1d(16*channels,track_running_stats=False).cuda()

        step = math.floor(((self.num_freq + 2 - 3) / 2) + 1)
        step = math.floor(((step + 2 - 3) / 2) + 1)
        step = math.floor(((step + 2 - 3) / 2) + 1)
        self.timestep = math.floor(((step + 2 - 3) / 2) + 1)
        if self.timestep == 0:
            self.timestep = 1
        self.fc = nn.Linear(8*channels*self.timestep, num_classes).cuda()

    def forward(self, x):
        x = self.batch_norm0(x)
        # xr, xi = fft.FFT_torch(x)

        im = torch.zeros_like(x).view(x.shape[0], x.shape[1], x.shape[2], 1)
        x2 = x.view(x.shape[0], x.shape[1], x.shape[2], 1)
        xc = torch.cat([x2, im], dim=3)
        fr = torch.fft(xc, signal_ndim=1)  # tensor with last dimension 2 (real+imag) , 1 is signal dimension

        xr = fr[:, :, :self.num_freq, 0]
        xi = fr[:, :, :self.num_freq, 1]
        x = torch.cat([xr, xi], dim=1)

        x = F.relu(self.conv1_bn(self.conv1(x)))
        x = F.relu(self.conv2_bn(self.conv2(x)))
        x = F.relu(self.conv3_bn(self.conv3(x)))
        x = F.relu(self.conv4_bn(self.conv4(x))).view(x.shape[0], -1)
       # x = F.relu(self.conv5_bn(self.conv5(x))).view(x.shape[0], -1)

        x = self.fc(x)

        return x


class DeepMix(nn.Module):
    def __init__(self, channels=16, window=128, num_classes=2):
        super(DeepMix, self).__init__()
        self.num_classes = num_classes
        self.channels = channels

        self.batch_norm0 = nn.BatchNorm1d(channels,track_running_stats=False).cuda()

        self.conv1 = nn.Conv1d(channels, 2*channels, 3, stride=2, padding=1).cuda()
        self.conv1_bn = nn.BatchNorm1d(2*channels,track_running_stats=False).cuda()
        self.conv2 = nn.Conv1d(2*channels, 2*channels, 3, stride=2, padding=1).cuda()
        self.conv2_bn = nn.BatchNorm1d(2*channels,track_running_stats=False).cuda()
        self.conv3 = nn.Conv1d(2*channels, 4*channels, 3, stride=2, padding=1).cuda()
        self.conv3_bn = nn.BatchNorm1d(4*channels,track_running_stats=False).cuda()
        self.conv4 = nn.Conv1d(4*channels, 8*channels, 3, stride=2, padding=1).cuda()
        self.conv4_bn = nn.BatchNorm1d(8*channels,track_running_stats=False).cuda()
        #self.conv5 = nn.Conv1d(8*channels, 8*channels, 3, stride=2, padding=1).cuda()
        #self.conv5_bn = nn.BatchNorm1d(8*channels,track_running_stats=False).cuda()

        self.conv1_fft = nn.Conv1d(2*channels, 4*channels, 3, stride=2, padding=1).cuda()
        self.conv1_bn_fft = nn.BatchNorm1d(4*channels,track_running_stats=False).cuda()
        self.conv2_fft = nn.Conv1d(4*channels, 4*channels, 3, stride=2, padding=1).cuda()
        self.conv2_bn_fft = nn.BatchNorm1d(4*channels,track_running_stats=False).cuda()
        self.conv3_fft = nn.Conv1d(4*channels, 8*channels, 3, stride=2, padding=1).cuda()
        self.conv3_bn_fft = nn.BatchNorm1d(8*channels,track_running_stats=False).cuda()
        self.conv4_fft = nn.Conv1d(8*channels, 8*channels, 3, stride=2, padding=1).cuda()
        self.conv4_bn_fft = nn.BatchNorm1d(8*channels,track_running_stats=False).cuda()
        #self.conv5_fft = nn.Conv1d(16*channels, 16*channels, 3, stride=2, padding=1).cuda()
        #self.conv5_bn_fft = nn.BatchNorm1d(16*channels,track_running_stats=False).cuda()

        self.finalConv = nn.Conv1d(8*channels*2, 8*channels*2, 3, stride=2, padding=1).cuda()
        self.finalConv_bn = nn.BatchNorm1d(8*channels*2,track_running_stats=False).cuda()

        self.timestep = int(window/32) if int(window/32) > 0 else 1
        self.fc = nn.Linear(2*8*channels*self.timestep, num_classes).cuda()

    def forward(self, x):
        x = self.batch_norm0(x)

        im = torch.zeros_like(x).view(x.shape[0], x.shape[1], x.shape[2], 1)
        x2 = x.view(x.shape[0], x.shape[1], x.shape[2], 1)
        xc = torch.cat([x2, im], dim=3)
        fr = torch.fft(xc, signal_ndim=1)  # tensor with last dimension 2 (real+imag) , 1 is signal dimension

        xr = fr[:, :, :, 0]
        xi = fr[:, :, :, 1]

        x = F.relu(self.conv1_bn(self.conv1(x)))
        x = F.relu(self.conv2_bn(self.conv2(x)))
        x = F.relu(self.conv3_bn(self.conv3(x)))
        x = F.relu(self.conv4_bn(self.conv4(x)))
        #x = F.relu(self.conv5_bn(self.conv5(x)))

        xfft = torch.cat([xr, xi], dim=1)

        xfft = F.relu(self.conv1_bn_fft(self.conv1_fft(xfft)))
        xfft = F.relu(self.conv2_bn_fft(self.conv2_fft(xfft)))
        xfft = F.relu(self.conv3_bn_fft(self.conv3_fft(xfft)))
        xfft = F.relu(self.conv4_bn_fft(self.conv4_fft(xfft)))
        #xfft = F.relu(self.conv5_bn_fft(self.conv5_fft(xfft)))

        xt = torch.cat([x, xfft], dim=1)

        xt = F.relu(self.finalConv_bn(self.finalConv(xt))).view(x.shape[0], -1)

        xt = self.fc(xt)

        return xt


class DeepMix2(nn.Module):
    def __init__(self, channels=16, window=128, num_classes=2):
        super(DeepMix2, self).__init__()
        self.num_classes = num_classes
        self.channels = channels

        self.batch_norm0 = nn.BatchNorm1d(channels,track_running_stats=False).cuda()

        self.conv1 = nn.Conv1d(3*channels, 2*3*channels, 3, stride=2, padding=1).cuda()
        self.conv1_bn = nn.BatchNorm1d(2*3*channels,track_running_stats=False).cuda()
        self.conv2 = nn.Conv1d(2*3*channels, 2*3*channels, 3, stride=2, padding=1).cuda()
        self.conv2_bn = nn.BatchNorm1d(2*3*channels,track_running_stats=False).cuda()
        self.conv3 = nn.Conv1d(2*3*channels, 4*3*channels, 3, stride=2, padding=1).cuda()
        self.conv3_bn = nn.BatchNorm1d(4*3*channels,track_running_stats=False).cuda()
        self.conv4 = nn.Conv1d(4*3*channels, 8*3*channels, 3, stride=2, padding=1).cuda()
        self.conv4_bn = nn.BatchNorm1d(8*3*channels,track_running_stats=False).cuda()
        #self.conv5 = nn.Conv1d(8*channels, 8*channels, 3, stride=2, padding=1).cuda()
        #self.conv5_bn = nn.BatchNorm1d(8*channels,track_running_stats=False).cuda()

        self.finalConv = nn.Conv1d(8*3*channels, 8*3*channels, 3, stride=2, padding=1).cuda()
        self.finalConv_bn = nn.BatchNorm1d(8*3*channels,track_running_stats=False).cuda()

        self.timestep = int(window/32) if int(window/32) > 0 else 1
        self.fc = nn.Linear(8*3*channels*self.timestep, num_classes).cuda()

    def forward(self, x):
        x = self.batch_norm0(x)

        im = torch.zeros_like(x).view(x.shape[0], x.shape[1], x.shape[2], 1)
        x2 = x.view(x.shape[0], x.shape[1], x.shape[2], 1)
        xc = torch.cat([x2, im], dim=3)
        fr = torch.fft(xc, signal_ndim=1)  # tensor with last dimension 2 (real+imag) , 1 is signal dimension

        xr = fr[:, :, :, 0]
        xi = fr[:, :, :, 1]

        xt = torch.cat([x, xr, xi], dim=1)  # cat dati e freq sui canali

        xt = F.relu(self.conv1_bn(self.conv1(xt)))
        xt = F.relu(self.conv2_bn(self.conv2(xt)))
        xt = F.relu(self.conv3_bn(self.conv3(xt)))
        xt = F.relu(self.conv4_bn(self.conv4(xt)))
        xt = F.relu(self.finalConv_bn(self.finalConv(xt))).view(x.shape[0], -1)

        xt = self.fc(xt)

        return xt


class ChunkGenerator:

    def __init__(self, df, size, offset, overlap, thr):
        self.valid = False
        self.currData = None
        self.df = df
        self.size = size
        self.overlap = overlap
        self.thr = thr
        self.nxt = max(0, offset - overlap)

    def again(self):
        if self.nxt is None:
            return False
        else:
            return True

    def get_chunk(self):
        if self.nxt + self.size > self.df.shape[0]:
            self.nxt = None
        if not self.again():
            return None
        d = self.df[self.nxt:self.nxt + self.size]
        self.nxt = max(0, self.nxt + self.size - self.overlap)
        # check for idle data
        if d[['CPU%']].mean().values[0] < self.thr:
            self.valid = False
            self.currData = None
        else:
            self.valid = True
            self.currData = d
        return d

    def __str__(self):
        s = "valid: %s, next: %d" % (self.valid, self.nxt if self.nxt is not None else -1)
        return s


def merge_filter(files, interestingColumns):
    for f in files:
        df = pd.read_table("VMdata/"+f, delim_whitespace=True, usecols=interestingColumns)
        try:
            df_tot = df_tot.append(df, ignore_index=True)
        except NameError:
            df_tot = df
    #df_tot = df_tot[interestingColumns]
    return df_tot


def load_chunks():

    filesWeb = ["WEB1PRO.data", "WEB2PRO.data", "WEB3PRO.data", "WEB4PRO.data"]
    filesSql = ["SQL1PRO.data", "SQL2PRO.data", "SQL3PRO.data", "SQL4PRO.data"]
    channels = 0

    if args.columns == 'small':
        interestingColumns=['CPU%', 'Memory%', 'SysCallRate', 'InPktRate', 'OutPktRate']
        channels = 5
    elif args.columns == 'medium':
        interestingColumns=['CPU%', 'Memory%', 'SysCallRate', 'InPktRate', 'OutPktRate', 'NetworkPktRt', 'AliveProc']
        channels = 7
    elif args.columns == 'complete':
        interestingColumns = ['SysCallRate', 'CPU%', 'IdleCPU%', 'PkFSSp%', 'CacheRdRt', 'Memory%', 'UserMem%', 'PgOutRate',
                          'PageOut', 'Sys+Cache%', 'SysMem%', 'InPktRate', 'OutPktRate', 'NetworkPktRt', 'AliveProc',
                          'ActiveProc']
        channels = 16

    w = args.window  # 256
    overlap = 0.75
    thr = args.threshold  # 0

    # merge dataframes
    dfWeb = merge_filter(filesWeb, interestingColumns)
    dfWeb.name = "Web"
    dfWeb['label'] = 1
    dfSql = merge_filter(filesSql, interestingColumns)
    dfSql.name = "SQL"
    dfSql['label'] = 0

    # Summary print
    for df in [dfSql, dfWeb]:
        print(df.name, df.shape)

    data = dict()
    for df in [dfSql, dfWeb]:
        print("df size: ", df.shape)
        cg = ChunkGenerator(df, w, 0, int(w * overlap), thr)
        df.c = len(interestingColumns)
        df.w = w
        df.overlap = int(w * overlap)

        data[df.name] = list()
        while cg.again():
            d = cg.get_chunk()
            data[df.name].append(d)

    return data, channels


def split(data, train_fraction=0.7, val_fraction=0.2, test_fraction=0.1):
    num_records = []
    num_records.append(len(data["Web"]))
    num_records.append(len(data["SQL"]))

    items_per_class = min(num_records)

    ntrain = int(items_per_class * train_fraction)
    web_train = data["Web"][:ntrain]
    sql_train = data["SQL"][:ntrain]
    train = web_train + sql_train
    print('lenght train data: ', len(train))

    nval = int(items_per_class * val_fraction)
    web_val = data["Web"][ntrain : ntrain + nval]
    sql_val = data["SQL"][ntrain : ntrain + nval]
    val = web_val + sql_val
    print('lenght val data: ', len(val))

    ntest = int(items_per_class * test_fraction)
    web_test = data["Web"][ntrain + nval : ntrain + nval + ntest - 1]
    sql_test = data["SQL"][ntrain + nval : ntrain + nval + ntest - 1]
    test = web_test + sql_test
    print('lenght test data: ', len(test))

    return train, val, test


def training(train, val, test, channels=16):
    writer = SummaryWriter()

    model_name = args.arch
    window = args.window
    batch_s = args.batch_size
    epochs = args.epochs
    lr = args.learning_rate
    wd = args.weight_decay
    freq = args.freqs

    if model_name == "DeepConv":
        model = DeepConv(channels=channels, window=window).to(device)
    elif model_name == "DeepFFT":
        model = DeepFFT(channels=channels, window=window).to(device) 
    elif model_name == "DeepMix":
        model = DeepMix(channels=channels, window=window).to(device)
    elif model_name == "DeepMix2":
        model = DeepMix2(channels=channels, window=window).to(device)
    else:
        model = DeepConv(channels=channels, window=window).to(device)
    print(model)

    if not os.path.exists("Results_VM"):
        os.makedirs("Results_VM")
    if not os.path.exists("Results_VM/train"):
        os.makedirs("Results_VM/train")
    if not os.path.exists("Results_VM/train/csv"):
        os.makedirs("Results_VM/train/csv")
    if not os.path.exists("Results_VM/test"):
        os.makedirs("Results_VM/test")
    if not os.path.exists("models"):
        os.makedirs("models")

    dataset_train = VMDataset(train, 'train')
    dataset_val = VMDataset(val, 'val')
    dataset_test = VMDataset(test, 'test')

    train_dataloader = DataLoader(dataset_train, batch_size=batch_s, num_workers=args.workers)
    val_dataloader = DataLoader(dataset_val, batch_size=batch_s, num_workers=args.workers)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=60, gamma=0.5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.6)
    best_acc = 0
    name_model = args.columns+"_window"+str(window)+"_"+model_name+"_freqs"+str(freq)+"_lr%0.5f_wd%0.5f_epochs"%(lr, wd)+".pt"

    for epoch in range(epochs):
        print("Epoch: %d of %d" % (epoch+1, epochs))
        train_loss = 0.0; total = 0; correct = 0
        model.train()
        dataloader_iter = iter(train_dataloader)
        for it, (vmdata, labels) in enumerate(dataloader_iter):
            vmdata.transpose_(1, 2)
            vmdata = torch.tensor(vmdata).to(device)#.float()
            labels = torch.tensor(labels).to(device).long()

            output = model(vmdata)
            loss = criterion(output, labels)
            train_loss += loss.item()

            _, predicted = torch.max(output.data, 1)
            total += len(labels)
            correct += (predicted == labels.data).sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (it*batch_s)%400 == 0:
                print("Processing Training epoch %d (%d/%d) %3.2f%%" % (epoch+1,it*batch_s,len(dataset_train),100*(it*batch_s/len(dataset_train))))

        train_accuracy = 100 * float(correct.tolist() / total)
        writer.add_scalar('data/loss_train', train_loss, epoch)
        writer.add_scalar('data/accuracy_train', train_accuracy, epoch)
        print("Training epoch %d -- Loss: %6.2f - Accuracy: %3.3f%%" % (epoch+1, train_loss, train_accuracy))

        # EVALUATION
        eval_loss = 0.0; total = 0; correct = 0
        tot_labels = list(); tot_predicted = list()
        model.eval()
        dataloader_val_iter = iter(val_dataloader)
        for it, (vmdata, labels) in enumerate(dataloader_val_iter):
            vmdata.transpose_(1, 2)
            vmdata = torch.tensor(vmdata).to(device)#.float()
            labels = torch.tensor(labels).to(device).long()

            output = model(vmdata)
            loss = criterion(output, labels)
            eval_loss += loss.item()

            _, predicted = torch.max(output.data, 1)
            total += len(labels)
            correct += (predicted == labels.data).sum()
            tot_predicted.append(predicted.cpu().detach().numpy())
            tot_labels.append(labels.cpu().detach().numpy())

            if (it*batch_s)%280 == 0:
                print("Evaluation epoch %d (%d/%d) %3.2f%%" % (epoch+1,it*batch_s,len(dataset_val),100*(it*batch_s/len(dataset_val))))

        tot_predicted = np.concatenate(tot_predicted, axis=0)
        tot_labels = np.concatenate(tot_labels, axis=0)
        m = precision_recall_fscore_support(tot_labels, tot_predicted, average='macro')
        scheduler.step(eval_loss)
        eval_accuracy = 100 * float(correct.tolist() / total)
        writer.add_scalar('data/loss_val', eval_loss, epoch)
        writer.add_scalar('data/accuracy_val', eval_accuracy, epoch)

        if eval_accuracy > best_acc:
            best_acc = eval_accuracy
            torch.save(model, 'models/' + name_model)

        print("Evaluation epoch %d -- Loss: %6.2f - Accuracy: %3.3f%% - F-score: %2.3f" %
              (epoch+1, eval_loss, eval_accuracy,m[2]))

    # TEST final
    model = torch.load('models/' + name_model)
    test_dataloader = DataLoader(dataset_test, batch_size=batch_s, num_workers=args.workers)
    loss_test = 0.0; total = 0; correct = 0
    tot_labels = list(); tot_predicted = list()
    model.eval()
    dataloader_test_iter = iter(test_dataloader)
    for it, (vmdata, labels) in enumerate(dataloader_test_iter):
        vmdata.transpose_(1, 2)
        vmdata = torch.tensor(vmdata).to(device)
        labels = torch.tensor(labels).to(device).long()

        output = model(vmdata)
        loss = criterion(output, labels)
        loss_test += loss.item()

        _, predicted = torch.max(output.data, 1)
        total += len(labels)
        correct += (predicted == labels.data).sum()
        tot_predicted.append(predicted.cpu().detach().numpy())
        tot_labels.append(labels.cpu().detach().numpy())

    tot_predicted = np.concatenate(tot_predicted, axis=0)
    tot_labels = np.concatenate(tot_labels, axis=0)
    m = precision_recall_fscore_support(tot_labels, tot_predicted, average='macro')
    test_accuracy = 100 * float(correct.tolist() / total)
	writer.add_scalar('data/accuracy_test', test_accuracy)
	writer.add_scalar('data/PRF_test', m)

    writer.close()


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    global args
    args = parser.parse_args()
    print("\nTASK: "+args.columns+"_window"+str(args.window)+"_"+args.arch+"_lr%0.5f_wd%0.5f"%(args.learning_rate, args.weight_decay))

    data, channels = load_chunks()
    train, val, test = split(data, train_fraction=0.7, val_fraction=0.2, test_fraction=0.1)

    training(train, val, test, channels=channels)
