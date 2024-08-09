import functools

print = functools.partial(print, flush=True)
import torch, os,sys
from torch import Tensor
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parameter import Parameter
import time, math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import SGDClassifier,LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from copy import deepcopy
import datetime
# from linformer import Linformer
from sklearn.metrics import confusion_matrix

from torchensemble import AdversarialTrainingClassifier
# def print_python_path():
#     print("Python sys.path:")
#     for path in sys.path:
#         print(f"  - {path}")
# print_python_path()
from Hyperparameters import args

from torchensemble import utils
from Focal_loss import FocalLoss
from typing import Any, List, Optional, Tuple

Cnames = ['Health', 'NAFLD']
from matplotlib import font_manager

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

from sklearn import tree
from sklearn.ensemble import GradientBoostingClassifier

def train(X_train, y_train, X_valid, y_valid, nclass, DNA_names,RNA_names, human_diseasename_list=None):
    X_train_gene = X_train
    X_valid_gene = X_valid
    args['maxLength'] = X_train.size(1)
    print(X_train_gene)
    print(y_train)
    print(X_valid_gene)
    print(y_valid)

    # regr = ElasticNet(random_state=0)
    # regr.fit(X_train_gene, y_train)
    regr = make_pipeline(
        # StandardScaler(),
        # SGDClassifier(loss='log_loss', penalty='elasticnet', alpha=0.0001, l1_ratio=0.15, random_state=42)
        # LogisticRegression(verbose=1)
        GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,     max_depth=1, random_state=0)
        # SGDClassifier(loss='log_loss',  alpha=0.0001, l1_ratio=0.15, random_state=42)
    )
    regr.fit(X_train, y_train)
    # importances = regr[-1].feature_importances_
    # feature_importance_pairs = list(zip(RNA_names, importances))
    #
    # # 根据重要性降序排序
    # sorted_feature_importances = sorted(feature_importance_pairs, key=lambda x: x[1], reverse=True)
    #
    # for feature, importance in sorted_feature_importances[:100]:
    #     print(f"{feature}: {importance:.4f}")


    y_pred = regr.predict_proba(X_valid_gene)

    # prob_0 = np.abs(y_pred - 1) / (np.abs(y_pred)+np.abs(y_pred - 1))
    # all_probs = np.stack([prob_0,1-prob_0]).T
    all_probs = y_pred
    print('all_probs size',all_probs.shape)
    # all_probs = torch.cat(all_probs, dim=0)
    # y_val = torch.cat(y_val, dim=0)
    # y_valid = y_val
    all_probs = torch.Tensor(all_probs)
    print('allpro ', all_probs.size())
    all_auc = []
    # fig = plt.figure(figsize=(100,100))
    rows, cols = 2, 3
    fig, ax = plt.subplots(rows, cols)
    thres = [0 for _ in range(nclass)]
    for i in range(1, nclass):
        print('====================' + human_diseasename_list[i] + ' start ===================')
        prob_y = [(prob, y) for prob, y in zip(all_probs, y_valid) if y == 0 or y == i]
        C_y_valid = [(0 if b == 0 else 1) for a, b in prob_y]
        # print(torch.stack([a for a, b in prob_y]).size())
        C_y_probs = torch.stack([a for a, b in prob_y])[:, i]
        # plt.subplot(2,2,i)
        auc, best_thres = subdraw_roc_by_proba(ax[int(i / cols)][i % cols], C_y_valid, C_y_probs,
                                   name=human_diseasename_list[i])
        print(human_diseasename_list[i] + ':', auc)

        thres[i] = best_thres
        C_y_probs_cali = Calibrate(best_thres, C_y_probs)
        print('0: ,', [p.item() for p, l in zip(C_y_probs_cali, C_y_valid) if l == 0])
        print('1: ,', [p.item() for p, l in zip(C_y_probs_cali, C_y_valid) if l == 1])
        all_auc.append(auc)
        print('######################' + human_diseasename_list[i] + ' end #####################')
    # plt.subplot(2,2,nclass)
    pan_y_probs = 1 - all_probs[:, 0]
    pan_y_valid = [(0 if b == 0 else 1) for b in y_valid]

    auc, best_thres = subdraw_roc_by_proba(ax[0][0], pan_y_valid, pan_y_probs, name='pan cancer')
    thres[0] = best_thres
    pan_y_probs_cali = Calibrate(best_thres, pan_y_probs)
    print('0: ,', [p.item() for p, l in zip(pan_y_probs_cali, pan_y_valid) if l == 0])
    print('1: ,', [p.item() for p, l in zip(pan_y_probs_cali, pan_y_valid) if l == 1])
    print('pan cancer:', auc)
    all_auc.append(auc)
    fig.tight_layout()
    fig.savefig(args['rootDir'] + '/multi_total_roc.png', bbox_inches='tight', dpi=150)
    plt.show()
    for a in all_auc:
        print(a)

    for i,b_thres in enumerate(thres):
        all_probs[:,i] = Calibrate(b_thres, all_probs[:,i])
    all_probs[:, 0] = 1 - all_probs[:,i]
    y_pred = torch.argmax(all_probs, dim=-1)
    cm = confusion_matrix(np.asarray(y_valid), np.asarray(y_pred))
    cm = cm / cm.sum(1)[:, None]
    print(cm)
    confusion_matrix_plot(cm, human_diseasename_list, filename='real_multi')

    print(human_diseasename_list)
    return regr


def draw_roc_by_proba(y_valid, gbm_y_proba, name=''):
    fig = plt.figure(figsize=(5, 5))
    gbm_auc = roc_auc_score(y_valid, gbm_y_proba)  # 计算auc
    gbm_fpr, gbm_tpr, gbm_threasholds = roc_curve(y_valid, gbm_y_proba)  # 计算ROC的值
    plt.title("roc_curve of %s(AUC=%.4f)" % (name, gbm_auc))
    plt.xlabel('1- Specificity(False Positive)')  # specificity = 1 - np.array(gbm_fpr))
    plt.ylabel('Sensitivity(True Positive)')  # sensitivity = gbm_tpr
    plt.plot(list(np.array(gbm_fpr)), gbm_tpr)
    # plt.gca().invert_xaxis()  # 将X轴反转
    fig.savefig(args['rootDir'] + name + '_roc.png', bbox_inches='tight', dpi=150)
    plt.show()
    return gbm_auc

def Calibrate(b_thres, probs):
    less_mask = probs < b_thres
    less_value = probs / b_thres * 0.5
    more_value = 1 - (1 - probs) / (1 - b_thres) * 0.5
    res = less_value * less_mask.float() + more_value * (1 - less_mask.float())
    return res

def subdraw_roc_by_proba(ax, y_valid, gbm_y_proba, name='', c='b'):
    # fig = plt.figure(figsize=(5, 5))
    gbm_auc = roc_auc_score(y_valid, gbm_y_proba)  # 计算auc
    gbm_fpr, gbm_tpr, gbm_threasholds = roc_curve(y_valid, gbm_y_proba)  # 计算ROC的值
    ax.set_title("roc_curve of %s(AUC=%.4f)" % (name, gbm_auc), fontsize=5)
    ax.set_xlabel('1- Specificity(False Positive)', fontsize=5)  # specificity = 1 - np.array(gbm_fpr))
    ax.set_ylabel('Sensitivity(True Positive)', fontsize=5)  # sensitivity = gbm_tpr
    # ax.xticks(fontsize=5)
    # ax.yticks(fontsize=5)
    print('x: ', gbm_fpr)
    print('y: ', gbm_tpr)
    ax.plot(list(np.array(gbm_fpr)), gbm_tpr, c)
    ax.fill_between(list(np.array(gbm_fpr)), y1=gbm_tpr, color=c, alpha=0.5)
    # plt.gca().invert_xaxis()  # 将X轴反转
    # fig.savefig(args['rootDir']+name + '_roc.png', bbox_inches='tight', dpi=150)
    # plt.show()
    best_threshold = None
    best_j = 0
    for i in range(len(gbm_fpr)):
        j = gbm_tpr[i] - gbm_fpr[i]
        print(j,gbm_tpr[i], gbm_fpr[i], gbm_threasholds[i])
        if j > best_j:
            best_j = j
            best_threshold = gbm_threasholds[i]
    print("best thres:" , best_threshold)
    return gbm_auc,best_threshold


def confusion_matrix_plot(cfm, human_diseasename_list=None, save_dir=None, filename=None):
    import seaborn as sns
    fig = plt.figure(figsize=(20, 20))
    ax = sns.heatmap(cfm, annot=True, fmt='.2%', cmap='Blues')

    ax.set_title('Cancer Confusion Matrix\n\n')
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values ')

    labels = human_diseasename_list if human_diseasename_list else []
    ax.xaxis.set_ticklabels(labels, rotation=45, ha='right')
    ax.yaxis.set_ticklabels(labels, rotation=0, ha='right')
    if save_dir:
        fig.savefig(save_dir, bbox_inches='tight', dpi=150)
    else:
        fig.savefig(args['rootDir'] + '/' + filename + '_cm.png', bbox_inches='tight', dpi=150)
    ## Display the visualization of the Confusion Matrix.
    plt.show()


def blind_test(model, test_rpkmt, ans_label, human_diseasename_list, separate=False):
    '''
    if separate == True then human_diseasename_list is focus_list
    else human_diseasename_list
    '''
    # print(test_rpkmt.values,type(test_rpkmt.values))
    test_data = torch.FloatTensor(test_rpkmt.values) / 100
    if separate:
        probs_on_each_model = []
        for m in model:
            # m=m.to(self.device)
            y_prob = m(test_data)
            probs_on_each_model.append(y_prob)
            m = m.to('cpu')

        probs_on_each_model = torch.stack(probs_on_each_model)[:, :, 1]
        ans = torch.argmax(probs_on_each_model, dim=0)
        # print(probs_on_each_model.size(),probs_on_each_model,ans)

        for idx, (a, al) in enumerate(zip(ans, ans_label)):
            # print(a)
            print(human_diseasename_list[a], 'Gold Label: ', al)
            for d_idx, dis in enumerate(human_diseasename_list):
                print(dis, ' : ', probs_on_each_model[d_idx, idx].item())
            print()

    else:
        y_prob = model(test_data)
        ans = torch.argmax(y_prob, dim=1)
        for a, la, yp in zip(ans, ans_label, y_prob):
            print(human_diseasename_list[a], 'Gold Label: ', la)
            for h, prob in zip(human_diseasename_list, yp):
                print(h, ' : ', prob.item())

def main():
    # 数据分割
    tsv_dir = '/home/siweideng/OxTium_cfDNA'
    dataset = ChromosomeDataset(tsv_dir)
    print(tsv_dir)
    X = dataset.processed_data
    y = dataset.labels
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
    train(X_train, y_train, X_test, y_test, 2, human_diseasename_list=['healthy','disease'])