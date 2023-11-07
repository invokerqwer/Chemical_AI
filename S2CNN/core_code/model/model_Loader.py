import torch
from model.EPGAN import EPGAN

def CreatModel(opt):
    model = EPGAN()
    
    model.initialize(opt)
    model.init_weights()
    
    if opt.is_Train and opt.reload_epoch:
        model.reload(opt.reload_epoch)

    if len(opt.gpu_ids) and torch.cuda.is_available():
        model.Encoder.cuda()
        model.Decoder.cuda()
        model.D.cuda()

    print('model {} was created'.format(model.name()))
    return model

def CreatModel_test(opt, load_epoch):
    model = EPGAN()
    
    model.initialize(opt)
    model.init_weights()
    
    model.reload(load_epoch)

    if len(opt.gpu_ids) and torch.cuda.is_available():
        model.Encoder.cuda()
        model.Decoder.cuda()

    print('model {} was created'.format(model.name()))
    return model