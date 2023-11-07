import argparse
import os
import torch
import datetime

now = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
checkpoints = './checkpoints'
checkpoints = os.path.join(checkpoints, now)

class BaseOptions(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False
        
    def initialize(self):
        self.parser.add_argument('--input_nc',type=int, default=3, 
                                 help='input image channels')
        self.parser.add_argument('--output_nc', type=int, default=3, 
                                 help='output image channels')
        self.parser.add_argument('--checkpoints_dir', type=str, default='', 
                                 help='dir to save model')
        self.parser.add_argument('--test_checkpoints_dir', type=str, default='', 
                                 help='dir to save model')
        self.parser.add_argument('--pretrained_D', type=str, default=None, 
                                 help='name of pretrained Discrimiator model')
        self.parser.add_argument('--pretrained_G', type=str, default=None, 
                                 help='name of pretrained Generator model')
        self.parser.add_argument('--N_expr', type=int, default=6, 
                                 help='nums of expression')
        self.parser.add_argument('--gpu_ids', type=str, default='',
                                 help='0,1,2..for GPU and -1 for CPU')
        self.initialized = True
    
    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args([])
        self.opt.is_Train = self.is_Train
        
        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id>=0:
                self.opt.gpu_ids.append(id)
        if torch.cuda.is_available() and len(self.opt.gpu_ids)>0:
            torch.cuda.set_device(self.opt.gpu_ids[0])
        
        # print the options
        args = vars(self.opt)
        for k, v in sorted(args.items()):
            print('{0}: {1}'.format(str(k), str(v)))

        # save the options to the disk
        expr_dir = self.opt.checkpoints_dir
        if not os.path.exists(expr_dir):
            os.makedirs(expr_dir)
        file_name = os.path.join(expr_dir, '{}_opt.txt'.format('train' if self.opt.is_Train else 'test'))
        with open(file_name, 'wt') as f:
            f.write('-----------------Options-----------------\n')
            for k, v in sorted(args.items()):
                f.write('{0}: {1}\n'.format(str(k), str(v)))
            f.write('-------------------End-------------------\n')

        return self.opt