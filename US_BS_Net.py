import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier as KNN
from common_usages import *
from Shrub_data_loader import ShrubCLSData
from losses import SupConLoss
from Preprocessing import Processor


def l1_regularization(model, l1_alpha):
    l1_loss = []
    for name, module in model.named_modules():
        if name == "channel_weight_layer":
            l1_loss.append(torch.abs(module.weight).sum())
    return l1_alpha * sum(l1_loss)


class Projector(nn.Module):
    def __init__(self, n_channel, projection_dim):
        super(Projector, self).__init__()
        self.projection_dim = projection_dim

        self.projector = nn.Sequential(
            nn.Linear(in_features=n_channel, out_features=self.projection_dim, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=self.projection_dim, out_features=self.projection_dim, bias=True),
        )

    def forward(self, x):
        out = self.projector(x)
        out = torch.nn.functional.normalize(out)
        return out


class BS_Net_Conv(nn.Module):
    def __init__(self, n_channel):
        super(BS_Net_Conv, self).__init__()
        self.n_channel = n_channel
        self.input_bn = nn.BatchNorm2d(n_channel, momentum=0.01, affine=True)
        self.channel_reweight_layer1 = nn.Sequential(
            nn.Conv2d(n_channel, 64, kernel_size=(3, 3)),
            nn.BatchNorm2d(64, momentum=0.01, affine=True),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )

        self.channel_reweight_layer2 = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU()
        )
        self.channel_weight_layer =  nn.Linear(128, n_channel)
        self.sigmoid = nn.Sigmoid()

        self.projector = Projector(64, 64)

        self.layer1 = nn.Sequential(
            nn.Conv2d(n_channel, 128, kernel_size=(3, 3), stride=(1, 1)),
            nn.BatchNorm2d(128, momentum=0.01, affine=True),
            nn.ReLU(),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1)),
            nn.BatchNorm2d(64, momentum=0.01, affine=True),
            nn.ReLU(),
        )

        self._initialize_weights()

    def forward(self, x):
        input_norm = self.input_bn(x)
        output = self.channel_reweight_layer1(input_norm)
        output = self.channel_reweight_layer2(output.squeeze())
        output = self.channel_weight_layer(output)
        channel_weight = self.sigmoid(output)
        channel_weight_s = channel_weight.view(-1, self.n_channel, 1, 1)
        reweight_out = channel_weight_s * input_norm
        fea = self.layer1(reweight_out)
        fea = self.layer2(fea)
        fea = fea.view(-1, 64)
        proj = self.projector(fea)
        return fea, channel_weight, proj
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data, dataset):
        self.Datalist = data
        self.Dataset = dataset

    def __getitem__(self, index):
        data_org = self.Dataset.getPatch(self.Datalist[index], windowSize=5).astype('float32')
        Data_org = torch.from_numpy(data_org.transpose((2, 0, 1)))
        Data_org = Data_org.view(Data_org.shape[0], Data_org.shape[1], Data_org.shape[2])
        return Data_org

    def __len__(self):
        return len(self.Datalist)


def eval_band_cv(Xtrain, ytrain, Xval, yval,  dataset, band_idx, times=10):
    p = Processor()
    estimator = [KNN(n_neighbors=3)]
    estimator_pre, y_test_all = [[], []], []
    for i in range(times):
        selected = np.random.permutation(len(Xtrain))[:2500]
        X_train = Xtrain[selected]
        y_train = ytrain[selected]
        selected = np.random.permutation(len(Xval))[:2500]
        X_test = Xval[selected]
        y_test = yval[selected]
        X_train = dataset.getPatches_bs(X_train, band_idx, 1)
        X_test = dataset.getPatches_bs(X_test, band_idx, 1)
        X_train = X_train.reshape(-1, len(band_idx))
        X_test = X_test.reshape(-1, len(band_idx))
        y_test_all.append(y_test)
        for c in range(len(estimator)):
            estimator[c].fit(X_train, y_train)
            y_pre = estimator[c].predict(X_test)
            estimator_pre[c].append(y_pre)
    score_dic = {'knn':{'ca':[], 'oa':[], 'aa':[], 'kappa':[]}}
    key_ = ['knn']
    for z in range(len(estimator)):
        ca, oa, aa, kappa = p.save_res_4kfolds_cv(estimator_pre[z], y_test_all, file_name=None, verbose=False)
        score_dic[key_[z]]['ca'] = ca
        score_dic[key_[z]]['oa'] = oa
        score_dic[key_[z]]['aa'] = aa
        score_dic[key_[z]]['kappa'] = kappa
    return score_dic


LEARNING_RATE = 0.0001
epochs = 40
N_BAND = 20
batch_size = 64
temperature = 0.9

def main():
    seed_everything()
    width, height, channel = 5, 5, 164
    shrub = ShrubCLSData(width, is_std=False)
    Xtrain = np.load('./data/Hyper_ShrubXtrainRatio0.99.npy', allow_pickle=True)
    ytrain = np.load('./data/Hyper_ShrubytrainRatio0.99.npy', allow_pickle=True)
    X_test = np.load('./data/Hyper_ShrubXtestRatio0.99.npy', allow_pickle=True)
    y_test = np.load('./data/Hyper_ShrubytestRatio0.99.npy', allow_pickle=True)
    Xval = np.load('./data/Hyper_ShrubXvalRatio0.99.npy', allow_pickle=True)
    yval = np.load('./data/Hyper_ShrubyvalRatio0.99.npy', allow_pickle=True)
    X = np.concatenate([Xtrain, X_test])
    y = np.concatenate([ytrain, y_test])
    print(len(X))
    print(len(X_test))
    bs_net = BS_Net_Conv(channel).cuda()
    loss = nn.MSELoss()
    patch_data = MyDataset(X, shrub)
    Patch_loader = torch.utils.data.DataLoader(dataset=patch_data, batch_size=batch_size, shuffle=True)
    bs_net_optim = torch.optim.Adam(bs_net.parameters(), lr=LEARNING_RATE)
    bs_net_scheduler = StepLR(bs_net_optim, step_size=2000, gamma=0.5)

    supConLoss = SupConLoss(temperature=temperature)
    best_acc = 0
    best_band = []
    for ep in range(epochs):
        print(ep)
        loss_total = 0
        bs_net.train()
        weight_batch = np.zeros((len(X), channel))
        for i, (data_org, data_aug) in enumerate(tqdm(Patch_loader)):
            label = torch.from_numpy(np.arange(data_org.shape[0]))
            data_org = data_org.cuda().float()
            data_aug = data_aug.cuda().float()
            fea_output, channel_weight, proj = bs_net(data_org)
            fea_output_aug, _, proj_aug = bs_net(data_aug)
            features = torch.cat([proj.unsqueeze(1), proj_aug.unsqueeze(1)], dim=1)
            loss_val = supConLoss(features, label)
            loss_val = loss_val + l1_regularization(bs_net, 0.01)
            bs_net_optim.zero_grad()
            loss_val.backward()
            bs_net_optim.step()
            bs_net_scheduler.step()
            loss_total = loss_total + loss_val.item()
            if i * batch_size + batch_size >= len(X):
                weight_batch[i*batch_size:, :] = channel_weight.detach().cpu().numpy()
            else:
                weight_batch[i*batch_size:(i+1)*batch_size, :] = channel_weight.detach().cpu().numpy()
        mean_weight = np.mean(weight_batch, axis = 0)
        band_indx = np.argsort(mean_weight)[::-1][:N_BAND]
        print('\nPatch Reconstruction Loss: {}'.format(loss_total / len(patch_data)))
        score = eval_band_cv(Xtrain, ytrain, Xval, yval, shrub, band_indx, times=20)
        print('acc=', score)
        acc = score['knn']['oa'][0]
        if acc > best_acc:
            best_acc = acc
            best_band = band_indx
            print(best_acc)
            np.save('model/SCL_weights_{}.npy'.format(N_BAND), mean_weight)
        print(band_indx)
    print('best acc: {}'.format(best_acc))
    print('best bands: {}'.format(best_band))
    np.save('model/SCL_best_bands_{}.npy'.format(N_BAND), best_band)


if __name__=='__main__':
    main()


