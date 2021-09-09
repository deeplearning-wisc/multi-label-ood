import os
import torch
import torch.nn as nn
import numpy as np
import time
import torch.nn.functional as F
from torch.autograd import Variable

to_np = lambda x: x.data.cpu().numpy()

def get_odin_scores(loader, model, clsfier, method, T, noise):
    ## get logits
    bceloss = nn.BCEWithLogitsLoss(reduction="none")
    for i, (images, _) in enumerate(loader):
        images = Variable(images.cuda(), requires_grad=True)
        nnOutputs = clsfier(model(images))

        # using temperature scaling
        preds = torch.sigmoid(nnOutputs / T)

        labels = torch.ones(preds.shape).cuda() * (preds >= 0.5)
        labels = Variable(labels.float())

        # input pre-processing
        loss = bceloss(nnOutputs, labels)

        if method == 'max':
            idx = torch.max(preds, dim=1)[1].unsqueeze(-1)
            loss = torch.mean(torch.gather(loss, 1, idx))
        elif method == 'sum':
            loss = torch.mean(torch.sum(loss, dim=1))

        loss.backward()
        # calculating the perturbation
        gradient = torch.ge(images.grad.data, 0)
        gradient = (gradient.float() - 0.5) * 2
        gradient.index_copy_(1, torch.LongTensor([0]).cuda(),
                             gradient.index_select(1, torch.LongTensor([0]).cuda()) / (0.229))
        gradient.index_copy_(1, torch.LongTensor([1]).cuda(),
                             gradient.index_select(1, torch.LongTensor([1]).cuda()) / (0.224))
        gradient.index_copy_(1, torch.LongTensor([2]).cuda(),
                             gradient.index_select(1, torch.LongTensor([2]).cuda()) / (0.225))
        tempInputs = torch.add(images.data, gradient, alpha=-noise)

        with torch.no_grad():
            nnOutputs = clsfier(model(Variable(tempInputs)))

            ## compute odin score
            outputs = torch.sigmoid(nnOutputs / T)

            if method == "max":
                score = np.max(to_np(outputs), axis=1)
            elif method == "sum":
                score = np.sum(to_np(outputs), axis=1)

            if i == 0:
                scores = score
            else:
                scores = np.concatenate((scores, score),axis=0)

    return scores

def sample_estimator(model, clsfier, num_classes, feature_list, train_loader):
    """
    compute sample mean and precision (inverse of covariance)
    return: sample_class_mean: list of class mean
             precision: list of precisions
    """
    import sklearn.covariance

    group_lasso = sklearn.covariance.EmpiricalCovariance(assume_centered=False)
    num_output = len(feature_list)
    num_sample_per_class = np.empty(num_classes)
    num_sample_per_class.fill(0)
    list_features = []

    # list_features = []
    # for i in range(num_output):
    #     temp_list = []
    #     for j in range(num_classes):
    #         temp_list.append(0)
    #     list_features.append(temp_list)

    for j in range(num_classes):
        list_features.append(0)

    idx = 0
    with torch.no_grad():
        for data, target in train_loader:
            idx += 1
            print(idx)
            data = Variable(data.cuda())

            target = target.cuda()

            # output, out_features = model_feature_list(model, clsfier, data)  # output = size[batch_size, num_class]
            # get hidden features
            # for i in range(num_output):
            #     out_features[i] = out_features[i].view(out_features[i].size(0), out_features[i].size(1), -1)
            #     out_features[i] = torch.mean(out_features[i].data, 2)

            out_features = model(data)
            out_features = out_features.view(out_features.size(0), out_features.size(1), -1)
            out_features = torch.mean(out_features.data, 2)

            # construct the sample matrix
            # use the training set labels(multiple) or set with the one with max prob

            for i in range(data.size(0)):
                # px = 0
                for j in range(num_classes):
                    if target[i][j] == 0:
                        continue
                    label = j
                    if num_sample_per_class[label] == 0:
                        # out_count = 0
                        # for out in out_features:
                        #     list_features[out_count][label] = out[i].view(1, -1)
                        #     out_count += 1

                        list_features[label] = out_features[i].view(1, -1)
                    else:
                        # out_count = 0
                        # for out in out_features:
                        #     list_features[out_count][label] \
                        #         = torch.cat((list_features[out_count][label], out[i].view(1, -1)), 0)
                        #     out_count += 1

                        list_features[label] = torch.cat((list_features[label],
                                                          out_features[i].view(1, -1)), 0)
                    num_sample_per_class[label] += 1

    # sample_class_mean = []
    # out_count = 0
    # for num_feature in feature_list:
    #     temp_list = torch.Tensor(num_classes, int(num_feature)).cuda()
    #     for j in range(num_classes):
    #         temp_list[j] = torch.mean(list_features[out_count][j], 0)
    #     sample_class_mean.append(temp_list)
    #     out_count += 1

    num_feature = feature_list[-1]
    temp_list = torch.Tensor(num_classes, int(num_feature)).cuda()
    for j in range(num_classes):
        temp_list[j] = torch.mean(list_features[j], 0)
    sample_class_mean = temp_list

    # precision = []
    # for k in range(num_output):
    #     X = 0
    #     for i in range(num_classes):
    #         if i == 0:
    #             X = list_features[k][i] - sample_class_mean[k][i]
    #         else:
    #             X = torch.cat((X, list_features[k][i] - sample_class_mean[k][i]), 0)
    #
    #     # find inverse
    #     group_lasso.fit(X.cpu().numpy())
    #     temp_precision = group_lasso.precision_
    #     temp_precision = torch.from_numpy(temp_precision).float().cuda()
    #     precision.append(temp_precision)

    X = 0
    for i in range(num_classes):
        if i == 0:
            X = list_features[i] - sample_class_mean[i]
        else:
            X = torch.cat((X, list_features[i] - sample_class_mean[i]), 0)
    # find inverse
    group_lasso.fit(X.cpu().numpy())
    temp_precision = group_lasso.precision_
    temp_precision = torch.from_numpy(temp_precision).float().cuda()
    precision = temp_precision

    return sample_class_mean, precision


def get_Mahalanobis_score(model, clsfier, loader, pack, noise, num_classes, method):
    '''
    Compute the proposed Mahalanobis confidence score on input dataset
    return: Mahalanobis score from layer_index
    '''
    sample_mean, precision = pack
    model.eval()
    clsfier.eval()
    Mahalanobis = []
    for i, (data, target) in enumerate(loader):
        data = Variable(data.cuda(), requires_grad=True)

        # out_features = model_penultimate_layer(model, clsfier, data)
        out_features = model(data)
        out_features = out_features.view(out_features.size(0), out_features.size(1), -1)
        out_features = torch.mean(out_features, 2) # size(batch_size, F)

        # compute Mahalanobis score
        gaussian_score = 0
        for i in range(num_classes):
            batch_sample_mean = sample_mean[i]
            zero_f = out_features.data - batch_sample_mean
            term_gau = -0.5 * torch.mm(torch.mm(zero_f, precision), zero_f.t()).diag()
            if i == 0:
                gaussian_score = term_gau.view(-1, 1)
            else:
                gaussian_score = torch.cat((gaussian_score, term_gau.view(-1, 1)), 1)

        # Input_processing
        sample_pred = gaussian_score.max(1)[1]
        batch_sample_mean = sample_mean.index_select(0, sample_pred)
        zero_f = out_features - Variable(batch_sample_mean)
        pure_gau = -0.5 * torch.mm(torch.mm(zero_f, Variable(precision)), zero_f.t()).diag()
        loss = torch.mean(-pure_gau)
        loss.backward()

        gradient = torch.ge(data.grad.data, 0)
        gradient = (gradient.float() - 0.5) * 2
        gradient.index_copy_(1, torch.LongTensor([0]).cuda(),
                             gradient.index_select(1, torch.LongTensor([0]).cuda()) / (0.229))
        gradient.index_copy_(1, torch.LongTensor([1]).cuda(),
                             gradient.index_select(1, torch.LongTensor([1]).cuda()) / (0.224))
        gradient.index_copy_(1, torch.LongTensor([2]).cuda(),
                             gradient.index_select(1, torch.LongTensor([2]).cuda()) / (0.225))
        tempInputs = torch.add(data.data, gradient, alpha=-noise)

        #noise_out_features = model.intermediate_forward(Variable(tempInputs, volatile=True), layer_index)
        with torch.no_grad():
            # noise_out_features = model_penultimate_layer(model, clsfier, Variable(tempInputs))
            noise_out_features = model(Variable(tempInputs))
            noise_out_features = noise_out_features.view(noise_out_features.size(0), noise_out_features.size(1), -1)
            noise_out_features = torch.mean(noise_out_features, 2)
            noise_gaussian_score = 0
            for i in range(num_classes):
                batch_sample_mean = sample_mean[i]
                zero_f = noise_out_features.data - batch_sample_mean
                term_gau = -0.5 * torch.mm(torch.mm(zero_f, precision), zero_f.t()).diag()
                if i == 0:
                    noise_gaussian_score = term_gau.view(-1, 1)
                else:
                    noise_gaussian_score = torch.cat((noise_gaussian_score, term_gau.view(-1, 1)), 1)
        # noise_gaussion_score size([batch_size, n_classes])

        if method == "max":
            noise_gaussian_score, _ = torch.max(noise_gaussian_score, dim=1)
        elif method == "sum":
            noise_gaussian_score = torch.sum(noise_gaussian_score, dim=1)

        Mahalanobis.extend(to_np(noise_gaussian_score))

    return Mahalanobis


def model_feature_list(model, clsfier, x, arch):
    out_list = []
    if arch == "resnet101":
        out = model.module[:4](x)
        out_list.append(out)
        out = model.module[4](out)
        out_list.append(out)
        out = model.module[5](out)
        out_list.append(out)
        out = model.module[6](out)
        out_list.append(out)
        out = model.module[7](out)
        out_list.append(out.data)
    elif arch == "densenet":
        out = model.module[:4](x)
        out_list.append(out)
        out = model.module[4:6](out)
        out_list.append(out)
        out = model.module[6:8](out)
        out_list.append(out)
        out = model.module[8:10](out)
        out_list.append(out)
        out = model.module[10:](out)
        out_list.append(out.data)
    return clsfier(out), out_list

def get_logits(loader, model, clsfier, args, k=20, name=None):
    print(args.save_path + name + ".npy", os.path.exists(args.save_path + name + ".npy"))
    if not (os.path.exists(args.save_path + name + ".npy")):
        logits_np = np.empty([0, args.n_classes])

        with torch.no_grad():
            for i, (images, labels) in enumerate(loader):

                images = Variable(images.cuda())
                nnOutputs = model(images)
                nnOutputs = clsfier(nnOutputs)

                nnOutputs_np = to_np(nnOutputs.squeeze())
                logits_np = np.vstack((logits_np, nnOutputs_np))

        os.makedirs(args.save_path, exist_ok = True)
        np.save(args.save_path + name, logits_np)

    else:
        logits_np = np.load(args.save_path + name + ".npy")

    ## Compute the Score
    logits = torch.from_numpy(logits_np).cuda()
    outputs = torch.sigmoid(logits)
    if args.ood == "logit":
        if args.method == "max": scores = np.max(logits_np, axis=1)
        if args.method == "sum": scores = np.sum(logits_np, axis=1)
    elif args.ood == "energy":
        E_f = torch.log(1+torch.exp(logits))
        if args.method == "max": scores = to_np(torch.max(E_f, dim=1)[0])
        if args.method == "sum": scores = to_np(torch.sum(E_f, dim=1))
        if args.method == "topk":
            scores = to_np(torch.sum(torch.topk(E_f, k=k, dim=1)[0], dim=1))
    elif args.ood == "prob":
        if args.method == "max": scores = np.max(to_np(outputs), axis=1)
        if args.method == "sum": scores = np.sum(to_np(outputs),axis=1)
    elif args.ood == "msp":
        outputs = F.softmax(logits, dim=1)
        scores = np.max(to_np(outputs), axis=1)
    else:
        scores = logits_np

    return scores


def get_localoutlierfactor_scores(val, test, out_scores):
    import sklearn.neighbors
    scorer = sklearn.neighbors.LocalOutlierFactor(novelty=True)
    print("fitting validation set")
    start = time.time()
    scorer.fit(val)
    end = time.time()
    print("fitting took ", end - start)
    val = np.asarray(val)
    test = np.asarray(test)
    out_scores = np.asarray(out_scores)
    print(val.shape, test.shape, out_scores.shape)
    return scorer.score_samples(np.vstack((test, out_scores)))


def get_isolationforest_scores(val, test, out_scores):
    import sklearn.ensemble
    rng = np.random.RandomState(42)
    scorer = sklearn.ensemble.IsolationForest(random_state = rng)
    print("fitting validation set")
    start = time.time()
    scorer.fit(val)
    end = time.time()
    print("fitting took ", end - start)
    val = np.asarray(val)
    test = np.asarray(test)
    out_scores = np.asarray(out_scores)
    print(val.shape, test.shape, out_scores.shape)
    return scorer.score_samples(np.vstack((test, out_scores)))



