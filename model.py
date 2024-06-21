import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parameter import Parameter
import time,math
import numpy as np

import datetime
from Hyperparameters import args
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

class deepmodel(nn.Module):
    def __init__(self):
        super(deepmodel, self).__init__()
        print("Model creation...")

        self.NLLloss = torch.nn.NLLLoss(reduction='none')
        self.CEloss = torch.nn.CrossEntropyLoss(reduction='none')

        # self.embedding = nn.Embedding(args['vocabularySize'], args['embeddingSize']).to(args['device'])

        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)
        self.logsoftmax = nn.LogSoftmax(dim=-1)

        self.x2z = nn.Linear(args['maxLength'], args['hiddenSize']).to(args['device'])
        # self.DiseaseClassifier = nn.Sequential(
        #     nn.Linear(args['hiddenSize'],2),
        #     nn.LogSoftmax(dim=-1)
        # ).to(args['device'])
        self.DiseaseClassifier =  nn.Linear(args['hiddenSize'], 2).to(args['device'])

    def build(self, x):
        '''
        :return:
        '''

        # print(x['enc_input'])
        self.encoderInputs = x['enc_input'].to(args['device']) / 100

        self.batch_size = self.encoderInputs.size()[0]
        # print(self.encoderInputs.size(), args['maxLength'])
        s_w_feature = self.x2z(self.encoderInputs)
        output = self.DiseaseClassifier(self.tanh(s_w_feature)).to(args['device'])  # batch chargenum


        return output

    def forward(self, x):
        output = self.build(x)
        output = self.logsoftmax(output)
        self.classifyLabels = x['labels'].to(args['device'])
        recon_loss = self.NLLloss(output, self.classifyLabels).to(args['device'])

        recon_loss_mean = torch.mean(recon_loss).to(args['device'])
        return recon_loss_mean

    def predict(self, x):

        output = self.build(x)
        return torch.argmax(output, dim=-1)

    def predict_proba(self, x):
        output = self.build(x)

        return self.softmax(output)

def initialize_model_(model):
    """
    Model initialization.
    :param model:
    :return:
    """
    print("Glorot init")
    for name, p in model.named_parameters():
        if name.startswith("embed") or "lagrange" in name:
            print("{:10s} {:20s} {}".format("unchanged", name, p.shape))
        elif "lstm" in name and len(p.shape) > 1:
            print("{:10s} {:20s} {}".format("xavier_n", name, p.shape))
            xavier_uniform_n_(p)
        elif len(p.shape) > 1:
            print("{:10s} {:20s} {}".format("xavier", name, p.shape))
            torch.nn.init.xavier_uniform_(p)
        elif "bias" in name:
            print("{:10s} {:20s} {}".format("zeros", name, p.shape))
            torch.nn.init.constant_(p, 0.)
        else:
            print("{:10s} {:20s} {}".format("unchanged", name, p.shape))

def getBatches(dataX,dataY, totalsize):

    batches = []

    def genNextSamples():
        """ Generator over the mini-batch training samples
        """
        for i in range(0, totalsize, args['batchSize']):
            yield dataX.iloc[i:min(i + args['batchSize'], totalsize)],dataY.iloc[i:min(i + args['batchSize'], totalsize)]

    # TODO: Should replace that by generator (better: by tf.queue)

    for index, samples in enumerate(genNextSamples()):
        # print([self.index2word[id] for id in samples[5][0]], samples[5][2])
        batch = samples
        batches.append(batch)

    return batches

def train(X_train, y_train, X_valid, y_valid , model_path= './deep.mdl', print_every=50, plot_every=10,
          learning_rate=0.001,  eps = 1e-6):


    start = time.time()
    plot_losses = []
    model = deepmodel().to(args['device'])

    optimizer = optim.Adam(model.parameters(), lr=0.0004, weight_decay=2e-6)
    initialize_model_(model)
    # if args["scheduler"] == "plateau":
    #     scheduler = ReduceLROnPlateau(
    #         G_optimizer, mode='min', factor=args["lr_decay"],
    #         patience=args["patience"],
    #         threshold=args["threshold"], threshold_mode='rel',
    #         cooldown=args["cooldown"], verbose=True, min_lr=args["min_lr"])
    # elif args["scheduler"] == "exponential":
    #     scheduler = ExponentialLR(G_optimizer, gamma=args["lr_decay"])
    # elif args["scheduler"] == "multistep":
    #     milestones = args["milestones"]
    #     print("milestones (epoch):", milestones)
    #     scheduler = MultiStepLR(
    #         G_optimizer, milestones=milestones, gamma=args["lr_decay"])
    #     scheduler_D = MultiStepLR(
    #         D_optimizer, milestones=milestones, gamma=args["lr_decay"])
    # else:
    #     raise ValueError("Unknown scheduler")

    iter = 1
    n_cases = len(X_train)
    batches = getBatches(X_train, y_train, n_cases)

    n_iters = len(batches)
    print('niters ', n_iters)
    max_f = -1

    print_loss_total = 0
    plot_loss_total = 0

    for epoch in range(args['numEpochs']):

        losses = []

        for index, (batchX, batchY) in enumerate(batches):
            # ---------------------
            #  Train Discriminator
            # ---------------------
            # for param in G_model.parameters():
            #     param.requires_grad = False
            # for param in D_model.parameters():
            #     param.requires_grad = True

            # for ind in range(index, index+5):
            #     ind = ind % n_iters

            x = {}
            x['enc_input'] = autograd.Variable(torch.FloatTensor(np.asarray(batchX))).to(args['device'])
            x['labels'] = autograd.Variable(torch.LongTensor(np.asarray(batchY))).to(args['device'])

            model.train()
            model.zero_grad()


            loss = model(x)  # batch seq_len outsize

            loss.backward(retain_graph=True)
            #
            torch.nn.utils.clip_grad_norm_(model.parameters(), args['clip'])
            #
            optimizer.step()


            print_loss_total += loss.data
            plot_loss_total += loss.data
            losses.append(loss.data)

            if iter % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0


                print('%s (%d %d%%) %.4f %.4f' % (timeSince(start, iter / (n_iters * args['numEpochs'])),
                                                  iter, iter / n_iters * 100, print_loss_avg, print_loss_avg), end='')

            if iter % plot_every == 0:
                plot_loss_avg = plot_loss_total / plot_every
                plot_losses.append(plot_loss_avg)
                plot_loss_total = 0

            iter += 1
            # print(iter, datetime.datetime.now())
        f1 = evaluate(model, X_valid, y_valid )
        if f1 > max_f or max_f == -1:
            print('F = ', f1, '>= max F1(', max_f, '), saving model...')
            torch.save(model, model_path)
            max_f = f1
    return model

def evaluate(model, X_valid, y_valid ):

    n_cases = len(X_valid)
    batches = getBatches(X_valid, y_valid, n_cases)

    TP_c = 0
    FP_c = 0
    FN_c = 0
    TN_c = 0
    for index, (batchX, batchY) in enumerate(batches):

        x = {}
        x['enc_input'] = autograd.Variable(torch.FloatTensor(np.asarray(batchX))).to(args['device'])
        x['labels'] = autograd.Variable(torch.LongTensor(np.asarray(batchY))).to(args['device'])

        y = model.predict(x)

        y = y.int().cpu().numpy()
        tp_c = ((batchY == 1) & (y == 1)).sum()  # c
        fp_c = ((batchY == 1) & (y == 0)).sum()  # c
        fn_c = ((batchY == 0) & (y == 1)).sum()  # c
        tn_c = ((batchY == 0) & (y == 0)).sum()  # c
        TP_c += tp_c
        FP_c += fp_c
        FN_c += fn_c
        TN_c += tn_c

    P_c = TP_c / (TP_c + FP_c)
    R_c = TP_c / (TP_c + FN_c)
    F_c = 2 * P_c * R_c / (P_c + R_c)

    print('p r f1: ',P_c, R_c, F_c)
    return F_c