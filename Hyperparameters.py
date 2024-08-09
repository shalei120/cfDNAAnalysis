import os
import torch

class HP:
    def __init__(self):
        self.args = self.predefined_args()

    def predefined_args(self):
        args = {}
        args['test'] = None
        args['createDataset'] = True
        args['playDataset'] = 10
        args['reset'] = True
        args['device'] = "cuda" if torch.cuda.is_available() else "cpu"
        args['rootDir'] = './Artifacts/'
        args['retrain_model'] = 'No'

        args['maxLength'] = 1000
        args['vocabularySize'] = 40000

        args['vq_embedding_dim'] = 150  # 64
        args['vq_n_embeddings'] = 512
        args['vq_beta'] = 0.25
        args['headnum'] = 4

        args['hiddenSize'] = 1000 #300
        args['numLayers'] = 2
        args['initEmbeddings'] = True
        args['embeddingSize'] = 128
        args['capsuleSize'] = 50 #300
        # args['embeddingSource'] = "GoogleNews-vectors-negative300.bin"

        args['numEpochs'] = 400
        args['saveEvery'] = 2000
        args['batchSize'] = 64
        args['learningRate'] = 0.001
        args['dropout'] = 0.3
        args['clip'] = 5.0

        args['encunit'] = 'lstm'
        args['decunit'] = 'lstm'
        args['enc_numlayer'] = 2
        args['dec_numlayer'] = 2

        args['maxLengthEnco'] = args['maxLength']
        args['maxLengthDeco'] = args['maxLength'] + 1

        args['temperature'] =1.0
        args['classify_type'] = 'multi'
        args['task'] = 'charge'
        args['scheduler'] = 'multistep'
        args["lr_decay"] = 0.97
        args["patience"] = 5
        args["threshold"] = 1e-4
        args["cooldown"] = 0
        args["min_lr"]=5e-5
        args["milestones"] = [25, 50, 75]

        return args

args = HP().args
