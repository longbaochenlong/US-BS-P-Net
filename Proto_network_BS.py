import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from datetime import datetime
from sklearn.metrics import classification_report, \
    confusion_matrix, \
    cohen_kappa_score, \
    precision_score, \
    accuracy_score
from common_usages import *
from Shrub_data_loader import ShrubCLSData

GPU = 0
LEARNING_RATE = 0.0005
n_epochs = 30
n_episodes = 100
n_classes = 2
n_way = n_classes
n_shot = 5
n_query = 15
n_test_way = n_classes
n_test_shot = 5

class Encoder(nn.Module):
    def __init__(self, n_channel):
        super(Encoder, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(n_channel, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(64, momentum=0.01, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(64, momentum=0.01, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(64, momentum=0.01, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(64, momentum=0.01, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(2))

        self._initialize_weights()

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.view(out.size(0), -1)
        return out

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


def euclidean_dist(x, y):
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)
    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)
    return torch.pow(x - y, 2).mean(2)


def test(support, query, encoder):
    sample_features = encoder(Variable(torch.from_numpy(support)).cuda(GPU))  # 5x64
    sample_features = sample_features.view(n_test_way, n_test_shot, 64)
    sample_features = torch.mean(sample_features, 1).squeeze(1)
    test_features = encoder(Variable(torch.from_numpy(query)).cuda(GPU))  # 20x64
    test_features = test_features.squeeze()
    dists = euclidean_dist(test_features, sample_features)
    log_p_y = F.log_softmax(-dists, dim=1).view(n_test_way, len(query), -1)
    _, y_hat = log_p_y.max(2)
    predict_labels = torch.argmax(-dists, dim=1)
    return predict_labels.cpu().numpy()


def main():
    seed_everything()
    width, height, channel = 21, 21, 80

    weight = np.load('model/SCL_weights_{}.npy'.format(channel))
    print(weight)
    band_idx = np.argsort(weight)[::-1][:channel]
    print(band_idx)

    shrub = ShrubCLSData(width, band_idx= band_idx, is_std=False, is_pca=False)
    Xtrain = np.load('./data/Hyper_ShrubXtrainRatio0.99.npy', allow_pickle=True)
    ytrain = np.load('./data/Hyper_ShrubytrainRatio0.99.npy', allow_pickle=True)
    Xtest = np.load('./data/Hyper_ShrubXvalRatio0.99.npy', allow_pickle=True)
    ytest = np.load('./data/Hyper_ShrubyvalRatio0.99.npy', allow_pickle=True)
    print(len(Xtrain))
    print(len(Xtest))

    X_dict = {}
    for i in range(n_classes):
        X_dict[i] = []
    tmp_count_index = 0
    for _, y_index in enumerate(ytrain):
        y_i = int(y_index)
        if y_i in X_dict:
            X_dict[y_i].append(Xtrain[tmp_count_index])
        else:
            X_dict[y_i] = []
            X_dict[y_i].append(Xtrain[tmp_count_index])
        tmp_count_index += 1
    for i in range(n_classes):
        arr = np.array(X_dict[i])
        print('{}:{}'.format(i, len(arr)))
        X_dict[i] = arr
    del Xtrain
    a = datetime.now()
    model = Encoder(channel)
    model.cuda(GPU)
    model_optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    model_scheduler = StepLR(model_optim, step_size=2000, gamma=0.5)

    save_path = './model/Hyper_window_size_{}_band_{}.pth'.format(width, channel)
    model.train()
    last_epoch_loss_avrg = 0.
    last_epoch_acc_avrg = 0.
    best_acc = 0.
    support_test = np.zeros([n_test_way, n_test_shot, width, height, channel], dtype=np.float32)
    predict_dataset = np.zeros([len(ytest)], dtype=np.int32)
    epi_classes = np.arange(n_classes)
    for i, epi_cls in enumerate(epi_classes):
        selected = np.random.permutation(len(X_dict[epi_cls]))[:n_test_shot]
        support_test[i] = shrub.getPatches(np.array(X_dict[epi_cls])[selected], width)

    support_test = support_test.transpose((0, 1, 4, 2, 3))
    support_test = np.reshape(support_test, [n_test_way * n_test_shot, channel, width, height])
    for ep in range(n_epochs):
        model.train()
        last_epoch_loss_avrg = 0.
        last_epoch_acc_avrg = 0.
        for epi in range(n_episodes):
            epi_classes = np.arange(n_way)
            # epi_classes = np.random.permutation(n_classes)[:n_way]
            samples = np.zeros([n_way, n_shot, width, height, channel], dtype=np.float32)
            batches = np.zeros([n_way, n_query, width, height, channel], dtype=np.float32)
            batch_labels = []
            for i, epi_cls in enumerate(epi_classes):
                selected = np.random.permutation(len(X_dict[epi_cls]))[:n_shot + n_query]
                samples[i] = shrub.getPatches(np.array(X_dict[epi_cls])[selected[:n_shot]], width)
                batches[i] = shrub.getPatches(np.array(X_dict[epi_cls])[selected[n_shot:]], width)
                for s in selected[n_shot:]:
                    batch_labels.append(epi_cls)

            samples = samples.transpose((0, 1, 4, 2, 3))
            batches = batches.transpose((0, 1, 4, 2, 3))

            samples = np.reshape(samples, [n_way * n_shot, channel, width, height])
            batches = np.reshape(batches, [n_way * n_query, channel, width, height])

            # calculate features
            sample_features = model(Variable(torch.from_numpy(samples)).cuda(GPU))  # 5x64
            sample_features = sample_features.view(n_way, n_shot, 64)
            sample_features = torch.mean(sample_features, 1).squeeze(1)

            test_features = model(Variable(torch.from_numpy(batches)).cuda(GPU))  # 20x64
            target_inds = torch.arange(0, n_way).view(n_way, 1, 1).expand(n_way, n_query, 1).long()
            target_inds = Variable(target_inds, requires_grad=False).cuda(GPU)
            dists = euclidean_dist(test_features, sample_features)
            log_p_y = F.log_softmax(-dists, dim=1).view(n_way, n_query, -1)
            loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
            loss_val = loss_val
            _, y_hat = log_p_y.max(2)
            acc_val = torch.eq(y_hat, target_inds.squeeze()).float().mean()
            model_optim.zero_grad()
            loss_val.backward()
            model_optim.step()
            model_scheduler.step()
            last_epoch_loss_avrg += loss_val.data
            last_epoch_acc_avrg += acc_val.data
            if (epi + 1) % 50 == 0:
                print('[epoch {}/{}, episode {}/{}] => loss: {:.5f}, acc: {:.5f}'.format(ep + 1, n_epochs, epi + 1,
                                                                                         n_episodes, loss_val.data,
                                                                                         acc_val.data))
        if (ep + 1) >= 1:
            model.eval()
            test_num = 3000
            test_count = int(len(ytest) / test_num)
            for i in range(test_count):
                query_test = shrub.getPatches(Xtest[i * test_num:(i + 1) * test_num], width).astype(np.float32)
                query_test = np.reshape(query_test, [-1, width, height, channel])
                query_test = query_test.transpose((0, 3, 1, 2))
                predict_dataset[i * test_num:(i + 1) * test_num] = test(support_test, query_test, model)
            query_test = shrub.getPatches(Xtest[test_count * test_num:], width).astype(np.float32)
            query_test = np.reshape(query_test, [-1, width, height, channel])
            query_test = query_test.transpose((0, 3, 1, 2))
            predict_dataset[test_count * test_num:] = test(support_test, query_test, model)
            overall_acc = accuracy_score(ytest, predict_dataset)
            if overall_acc > best_acc:
                best_acc = overall_acc
                print('best acc: {:.2f}'.format(overall_acc * 100))
                torch.save(model.state_dict(), save_path)
    b = datetime.now()
    durn = (b - a).seconds
    print("Training time:", durn)
    model.load_state_dict(torch.load(save_path, weights_only=True))
    print('completed')
    a = datetime.now()
    print('Testing...')
    model.eval()
    del X_dict
    Xtest = np.load('./data/Hyper_ShrubXvaltestRatio0.99-2.npy', allow_pickle=True)
    ytest = np.load('./data/Hyper_ShrubyvaltestRatio0.99-2.npy', allow_pickle=True)
    predict_dataset = np.zeros([len(ytest)], dtype=np.int32)
    test_num = 3000
    test_count = int(len(ytest) / test_num)
    for i in range(test_count):
        query_test = shrub.getPatches(Xtest[i * test_num:(i + 1) * test_num], width).astype(np.float32)
        query_test = np.reshape(query_test, [-1, width, height, channel])
        query_test = query_test.transpose((0, 3, 1, 2))
        predict_dataset[i * test_num:(i + 1) * test_num] = test(support_test, query_test, model)
    query_test = shrub.getPatches(Xtest[test_count * test_num:], width).astype(np.float32)
    query_test = np.reshape(query_test, [-1, width, height, channel])
    query_test = query_test.transpose((0, 3, 1, 2))
    del Xtest
    predict_dataset[test_count * test_num:] = test(support_test, query_test, model)
    confusion = confusion_matrix(ytest, predict_dataset)
    acc_for_each_class = precision_score(ytest, predict_dataset, average=None)
    average_accuracy = np.mean(acc_for_each_class)
    kappa = cohen_kappa_score(ytest, predict_dataset)
    overall_acc = accuracy_score(ytest, predict_dataset)
    print('best acc: {:.2f}'.format(best_acc * 100))
    print('Last loss:{:.5f}'.format(last_epoch_loss_avrg / n_episodes))
    print('Last acc:{:.2f}'.format(last_epoch_acc_avrg / n_episodes * 100))
    print('OA: {:.2f}'.format(overall_acc * 100))
    print('kappa:{:.4f}'.format(kappa))
    print('PA:')
    for i in range(len(acc_for_each_class)):
        print('{:.2f}'.format(acc_for_each_class[i] * 100))
    print('AA: {:.2f}'.format(average_accuracy * 100))


if __name__=='__main__':
    main()
