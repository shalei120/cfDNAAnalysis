import functools
print = functools.partial(print, flush=True)
import pandas as pd
from Hyperparameters import args
import argparse
import numpy as np
import datetime,time
import statsmodels.api as sm
from tqdm import tqdm
import json,torch

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', '-g')
parser.add_argument('--separate', '-s')
parser.add_argument('--amount', '-a')
parser.add_argument('--id', '-i')
parser.add_argument('--method', '-m')
cmdargs = parser.parse_args()

if cmdargs.gpu is None:
    args['device'] = 'cpu'
else:
    args['device'] = str(cmdargs.gpu)
    # args['device'] ='cuda:'+str(cmdargs.gpu)

if cmdargs.amount is None:
    args['amount'] = '6G'
else:
    args['amount'] = str(cmdargs.amount)
    # args['device'] ='cuda:'+str(cmdargs.gpu)
if cmdargs.separate is None:
    args['separate'] = True
    # print('seqq: 1 ', args['separate'])
else:
    args['separate'] = bool(int(cmdargs.separate))
    # print('seqq: 2 ', cmdargs.separate, args['separate'])

if cmdargs.id is None:
    args['id'] = '0'
else:
    args['id'] = cmdargs.id


if cmdargs.method is None:
    args['method'] = 'stats'
else:
    args['method'] = cmdargs.method

import sys
import os


class RNAseq_Record:
    def __init__(self):
        self.record_file = '../qc统计20240130.xlsx'
        self.cfRNA_dir = '/home/siweideng/OxTium_data/'
        self.focus_6G_list = ['健康6G', '结直肠癌6G', '胃癌6G', '肺癌6G', '肝癌6G']
        # self.focus_6G_list = ['健康6G', '胰腺癌6G']
        self.focus_1G_list = ['健康1G', '结直肠癌1G', '胃癌1G', '肺癌1G', '肝癌1G']
        self.pklfile = args['rootDir'] + '/' + args['id'] + '_real2_'+ args['amount'] +'_T.csv'
        self.Zh2En = {'健康': 'Healthy', '结直肠癌': 'Colorectal Cancer',
                      '胃癌': 'Stomach Cancer', '胰腺癌': 'Pancreatic Cancer',
                      '肺癌': 'Lung Cancer', '肝癌': 'Liver Cancer'}

    def cut4id(self, filename):
        '''
        P0230602001015200ul3min_processed.tsv
        P023080400100812_1_processed.tsv
        :param filename:
        :return:
        '''
        # if '200ul' in filename:
        #     res = filename.split('200ul')[0]
        # else:
        splitstr = filename.split('_')
        res = splitstr[0]
        if 'P' in res:
            postPstrs = res.split('P')
            preP = postPstrs[0]
            postPstr = postPstrs[1]
            postPstr = postPstr[:13]
            res = preP + 'P' + postPstr

        return res

    def read_sheet(self, name):
        dp = pd.read_excel(self.record_file, sheet_name=name)

        idlist = dp['样本ID']
        idlen = set([len(idl) for idl in idlist])
        analysis_date = dp['分析日期']
        source = dp['样本来源']
        # print(idlist)
        source_codes = pd.Categorical(source)
        region_code_map = {cat: code for code, cat in enumerate(source_codes.categories)}
        # with open(args['rootDir'] + 'region_code_map.json', 'w') as f:
        #     json.dump(region_code_map, f)
        print(name,region_code_map)

        date2filemap = {}

        fpms = []
        for id, date, hospital in zip(idlist, analysis_date, source):
            if '商业' in hospital:
                continue
            if date not in date2filemap:
                sample_path = self.cfRNA_dir + str(date) + '_run/norm/'
                filelist = os.listdir(sample_path)
                date2filemap[date] = {self.cut4id(f): f for f in filelist}
            try:
                fn = self.cfRNA_dir + str(date) + '_run/norm/' + date2filemap[date][id]
            except:
                print(date, id, name, date2filemap, idlist)
                fn = self.cfRNA_dir + str(date) + '_run/norm/' + date2filemap[date][id]

            # if region == 1 and '健康' in name:
            #     continue
            fpm = pd.read_csv(fn, delimiter='\t')
            username = fpm.columns[1]
            fpm = fpm.rename(columns={username: id})
            # df2 = {"Geneid": 'region', id: 'H'+str(region_code_map[hospital])}
            df2 = {"Geneid": 'region', id: hospital}
            # if '健康' not in name:
            # if hospital == '北大第一医院':
            #     df2 = {"Geneid": 'region', id: '浙江省邵逸夫医院'} #'浙江省邵逸夫医院'

            # if '健康' in name:
            #     df3 = {"Geneid": 'disease', id: name+'+'+hospital}
            # else:
            df3 = {"Geneid": 'disease', id: name}
            fpm = fpm.append(df2, ignore_index=True)
            fpm = fpm.append(df3, ignore_index=True)

            fpms.append(fpm)

        fpm = fpms[0]
        for i in range(1, len(fpms)):
            fpm = pd.merge(fpm, fpms[i], on="Geneid")

        fpm = fpm.rename(columns={'Geneid': 'Run'})
        fpmt = self.tranpose(fpm)
        # fpmt['disease'] = name
        print(fpmt.columns)
        return fpmt

    @classmethod
    def tranpose(cls, fpm):
        fpmt = fpm.T
        # fpmt.drop(index=0)

        fpmt = fpmt.reset_index()
        fpmt.columns = fpmt.iloc[0]
        fpmt = fpmt.drop(index=0)
        return fpmt

    def preparedata(self, focuslist):
        all_fpmt = []
        for focusdis in focuslist:
            fpmt = self.read_sheet(focusdis)
            all_fpmt.append(fpmt)
        cfRNA_fpmt_data = pd.concat(all_fpmt)
        return cfRNA_fpmt_data

    def preparedata_from_matrix(self, focuslist):
        id2dis = []
        for focusdis in focuslist:
            dp = pd.read_excel(self.record_file, sheet_name=focusdis)
            idlist = dp['样本ID']
            source = dp['样本来源']
            for id, hospital in zip(idlist, source):
                # if '北大' in hospital:
                #     region = 1
                # elif '浙江' in hospital:
                #     region = 2
                # elif '商业' in hospital:
                #     region = 3
                # else:
                #     region = 0
                id2dis.append([id, focusdis, hospital])

        id2dis_df = pd.DataFrame(id2dis, columns=['Run', 'disease', 'region'])

        fpm = pd.read_csv('../tuned_data/' + args['id'] + '/df_combined.tsv', delimiter='\t')
        fpm = fpm.rename(columns={c: self.cut4id(c) for c in fpm.columns})
        fpm.columns.values[0] = 'Run'
        fpmt = self.tranpose(fpm)
        print(id2dis_df)
        print(fpmt)

        merged_df = pd.merge(id2dis_df, fpmt, on='Run')
        return merged_df

    def run(self):
        if os.path.exists(self.pklfile):
            print('Reading CSV...')
            alldata = pd.read_csv(self.pklfile, delimiter=',')
        else:
            print('Working on data preparing..')
            if args['amount'] == '6G':
                alldata = self.preparedata(self.focus_6G_list)
                # alldata = self.preparedata_from_matrix(self.focus_6G_list)
            elif args['amount'] == '1G':
                alldata = self.preparedata(self.focus_1G_list)
            else:
                print('Fuck! wrong choice: ', args['amount'])
            # r=self.read_sheet('健康6G')
            # print(r)

            alldata.to_csv(self.pklfile, index=False)

        return alldata


def receive_any_task(rpkm_filename, info_filename, rpkmt_filename, DictInfoColumnRename, HealthyLabel, GeneColumn,
                         T, rpkm_delimiter=',', other_features=[], merge_rpkm=False, Test = False):

    rpkmt_filename = rpkmt_filename
    if merge_rpkm:
        assert os.path.isdir(rpkm_filename)
        files = os.listdir(rpkm_filename)
        fpms = []
        for f in files:
            if f[-3:] == 'tsv':
                fpm = pd.read_csv(rpkm_filename +'/'+ f, delimiter=rpkm_delimiter)
                fpms.append(fpm)
        fpm = fpms[0]
        for i in range(1, len(fpms)):
            fpm = pd.merge(fpm, fpms[i], on="Geneid")
        rpkm = fpm
        print('rpkm ', rpkm)
    else:
        print('read from rpkm file')
        rpkm = pd.read_csv(rpkm_filename, delimiter=rpkm_delimiter)

    rpkm = rpkm.rename(columns={GeneColumn: 'Run'})
    rpkmt = T(rpkm)
    print('rpkmt',rpkmt)
    merged_df = rpkmt
    if not Test:
        info = pd.read_csv(info_filename, delimiter='\t')
        info = info.rename(columns=DictInfoColumnRename)
        info['disease'] = info['disease'].replace(HealthyLabel, '健康' + args['amount'])
        print('info',info)
        merged_df = pd.merge(merged_df, info[['Run', 'disease']+other_features], on='Run')
    merged_df['region'] = '北京医院'
    merged_df.to_csv(rpkmt_filename, index=False)
    print('merged_df:',merged_df)

