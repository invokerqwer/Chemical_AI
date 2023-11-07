import Ipynb_importer
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.autograd import grad
from PIL import Image
import numpy as np
import gc
import torch.cuda
import imp

from model.Component import Decoder
from model.Component import Discriminator
from model.Component import one_hot
from model.Component import one_hot_long
from model.Component import one_hot_intensity
from model.Component import weights_init_normal
from model.Component import Tensor2Image
from model.base_model import BaseModel

class MGAN(BaseModel):#molecular Gan
    '''def name(self):
        return 'EPGAN'''''

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.opt = opt

        MainModel = imp.load_source('MainModel', '/path/to/resnet50_scratch_dims_2048.py')
        self.Encoder = MainModel.resnet50_scratch('/path/to/resnet50_scratch_dims_2048.pth')
        for param in self.Encoder.parameters():
            param.requires_grad = False

        self.Decoder = Decoder(N_expr=self.opt.N_expr)
        self.D = Discriminator()

        if self.opt.is_Train:
            self.optimizer_G = optim.Adam(self.Decoder.parameters(), lr=self.opt.lr,
                                          betas=(self.opt.beta1, self.opt.beta2), weight_decay=self.opt.lambda_reg)
            self.optimizer_D = optim.Adam(self.D.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, self.opt.beta2),
                                          weight_decay=self.opt.lambda_reg)
            self.CE_criterion = nn.CrossEntropyLoss().cuda()
            self.L1_criterion = nn.L1Loss().cuda()
            self.L2_criterion = nn.MSELoss().cuda()

    def init_weights(self):
        self.Decoder.apply(weights_init_normal)
        self.D.apply(weights_init_normal)

    def load_input_f(self, input):
        self.image_f = []
        self.expr_label = []

        for i in input[0]:
            self.image_f.append(i)
        for j in input[1]:
            self.expr_label.append(j)

    def load_input_p(self, input):
        self.image_p = []

        for i in input[0]:
            self.image_p.append(i)

    def set_input(self, input_f, input_p):
        self.load_input_f(input_f)
        self.load_input_p(input_p)

        self.image_f = torch.stack(self.image_f, dim=0)
        self.image_p = torch.stack(self.image_p, dim=0)

        self.real_expr = torch.LongTensor(self.expr_label)
        self.real_expr_one_hot = one_hot(self.real_expr, self.opt.N_expr)
        self.real_expr_long = one_hot_long(self.real_expr, self.opt.N_expr)
        self.real_expr_long_1 = self.real_expr_long[:, 0 * 5:1 * 5]
        self.real_expr_long_2 = self.real_expr_long[:, 1 * 5:2 * 5]
        self.real_expr_long_3 = self.real_expr_long[:, 2 * 5:3 * 5]
        self.real_expr_long_4 = self.real_expr_long[:, 3 * 5:4 * 5]
        self.real_expr_long_5 = self.real_expr_long[:, 4 * 5:5 * 5]
        self.real_expr_long_6 = self.real_expr_long[:, 5 * 5:6 * 5]

        # cuda
        if self.opt.gpu_ids:
            self.image_f = self.image_f.cuda()
            self.image_p = self.image_p.cuda()

            self.real_expr = self.real_expr.cuda()
            self.real_expr_one_hot = self.real_expr_one_hot.cuda()
            self.real_expr_long = self.real_expr_long.cuda()
            self.real_expr_long_1 = self.real_expr_long_1.cuda()
            self.real_expr_long_2 = self.real_expr_long_2.cuda()
            self.real_expr_long_3 = self.real_expr_long_3.cuda()
            self.real_expr_long_4 = self.real_expr_long_4.cuda()
            self.real_expr_long_5 = self.real_expr_long_5.cuda()
            self.real_expr_long_6 = self.real_expr_long_6.cuda()

        self.image_f = Variable(self.image_f)
        self.image_p = Variable(self.image_p)
        self.image_f = (self.image_f + 1) * 127.5
        self.image_p = (self.image_p + 1) * 127.5

        self.real_expr = Variable(self.real_expr)
        self.real_expr_one_hot = Variable(self.real_expr_one_hot)
        self.real_expr_long = Variable(self.real_expr_long)
        self.real_expr_long_1 = Variable(self.real_expr_long_1)
        self.real_expr_long_2 = Variable(self.real_expr_long_2)
        self.real_expr_long_3 = Variable(self.real_expr_long_3)
        self.real_expr_long_4 = Variable(self.real_expr_long_4)
        self.real_expr_long_5 = Variable(self.real_expr_long_5)
        self.real_expr_long_6 = Variable(self.real_expr_long_6)

    def forward(self, input_f, input_p):
        self.set_input(input_f, input_p)

        self.real_conv_f, self.real_id_f = self.Encoder(self.image_f)
        self.real_conv_p, self.real_id_p = self.Encoder(self.image_p)

        self.syn_image_f = self.Decoder(self.real_conv_f, self.real_expr_long)
        self.syn_image_p = self.Decoder(self.real_conv_p, self.real_expr_long)

        self.syn_conv_f, self.syn_id_f = self.Encoder(self.syn_image_f)
        self.syn_conv_p, self.syn_id_p = self.Encoder(self.syn_image_p)

        # x_adv, cats, x_e, x_n, x_m, x_f
        self.syn_adv_f, self.syn_expr_f, self.syn_eye_f, self.syn_nose_f, self.syn_mouth_f, self.syn_face_f = self.D(
            self.syn_image_f, self.real_expr_one_hot)
        self.syn_adv_p, self.syn_expr_p, self.syn_eye_p, self.syn_nose_p, self.syn_mouth_p, self.syn_face_p = self.D(
            self.syn_image_p, self.real_expr_one_hot)
        self.real_adv_f, self.real_expr_f, self.real_eye_f, self.real_nose_f, self.real_mouth_f, self.real_face_f = self.D(
            self.image_f, self.real_expr_one_hot)

        # norm for id
        #         self.real_id_f = self.real_id_f / (torch.norm(self.real_id_f, p=2, keepdim=True) + self.opt.epsilon)
        #         self.real_id_p = self.real_id_p / (torch.norm(self.real_id_p, p=2, keepdim=True) + self.opt.epsilon)
        #         self.syn_id_f = self.syn_id_f / (torch.norm(self.syn_id_f, p=2, keepdim=True) + self.opt.epsilon)
        #         self.syn_id_p = self.syn_id_p / (torch.norm(self.syn_id_p, p=2, keepdim=True) + self.opt.epsilon)

    def backward_G(self, stage):
        self.loss_G_p = torch.mean(torch.sum(torch.abs(self.image_f / 255. - self.syn_image_f / 255.), [1, 2, 3]))
        self.loss_G_id = torch.mean(0.5 * (1 - torch.cosine_similarity(self.real_id_p, self.syn_id_p)) + 0.5 * (
                    1 - torch.cosine_similarity(self.real_id_f, self.syn_id_f)))

        self.loss_G_expr = torch.mean(torch.pow((self.syn_expr_f[0] - self.real_expr_long_1), 2)) + \
                           torch.mean(torch.pow((self.syn_expr_f[1] - self.real_expr_long_2), 2)) + \
                           torch.mean(torch.pow((self.syn_expr_f[2] - self.real_expr_long_3), 2)) + \
                           torch.mean(torch.pow((self.syn_expr_f[3] - self.real_expr_long_4), 2)) + \
                           torch.mean(torch.pow((self.syn_expr_f[4] - self.real_expr_long_5), 2)) + \
                           torch.mean(torch.pow((self.syn_expr_f[5] - self.real_expr_long_6), 2))

        self.loss_G_local_fake_f = self.syn_adv_f + self.syn_eye_f + self.syn_nose_f + self.syn_mouth_f + self.syn_face_f
        self.loss_G_local_fake_p = self.syn_adv_p + self.syn_eye_p + self.syn_nose_p + self.syn_mouth_p + self.syn_face_p
        self.loss_G_adv = -torch.mean(0.5 * self.loss_G_local_fake_p + 0.5 * self.loss_G_local_fake_f) / 5

        if stage == 1:
            self.loss_G = 0.01 * self.loss_G_p
        else:
            self.loss_G = 0.01 * self.loss_G_p + 200 * self.loss_G_id + 10 * self.loss_G_expr + 1 * self.loss_G_adv

        self.loss_G.backward()

    def backward_D(self):
        ########### gradient penalty(WGAN-GP)#########
        alpha = torch.rand((self.opt.batchsize, 1, 1, 1))
        if self.opt.gpu_ids:
            alpha = alpha.cuda()

        x_hat = (1 - alpha) * self.image_f.data + alpha * self.syn_image_p.data
        x_hat.requires_grad = True

        pred_hat, _, _, _, _, _ = self.D(x_hat, self.real_expr_one_hot)
        if self.opt.gpu_ids:
            gradients = grad(outputs=pred_hat, inputs=x_hat, grad_outputs=torch.ones(pred_hat.size()).cuda(),
                             create_graph=True, retain_graph=True, only_inputs=True)[0]
        else:
            gradients = grad(outputs=pred_hat, inputs=x_hat, grad_outputs=torch.ones(pred_hat.size()),
                             create_graph=True, retain_graph=True, only_inputs=True)[0]

        self.gradient_penalty = ((gradients.view(gradients.size()[0], -1).norm(2, 1) - 1) ** 2).mean()
        ########### gradient penalty(WGAN-GP)#########

        self.loss_D_local_fake_f = self.syn_adv_f + self.syn_eye_f + self.syn_nose_f + self.syn_mouth_f + self.syn_face_f
        self.loss_D_local_fake_p = self.syn_adv_p + self.syn_eye_p + self.syn_nose_p + self.syn_mouth_p + self.syn_face_p
        self.loss_D_local_real_f = self.real_adv_f + self.real_eye_f + self.real_nose_f + self.real_mouth_f + self.real_face_f
        self.loss_D_adv = torch.mean(
            0.5 * self.loss_D_local_fake_p + 0.5 * self.loss_D_local_fake_f - self.loss_D_local_real_f) / 5

        self.loss_D_expr = torch.mean(torch.pow((self.syn_expr_f[0] - self.real_expr_long_1), 2)) + \
                           torch.mean(torch.pow((self.syn_expr_f[1] - self.real_expr_long_2), 2)) + \
                           torch.mean(torch.pow((self.syn_expr_f[2] - self.real_expr_long_3), 2)) + \
                           torch.mean(torch.pow((self.syn_expr_f[3] - self.real_expr_long_4), 2)) + \
                           torch.mean(torch.pow((self.syn_expr_f[4] - self.real_expr_long_5), 2)) + \
                           torch.mean(torch.pow((self.syn_expr_f[5] - self.real_expr_long_6), 2))

        self.loss_D = self.loss_D_adv + 10 * self.loss_D_expr + self.opt.lambda_gp * self.gradient_penalty
        self.loss_D.backward(retain_graph=True)

    def optimize_G_parameters(self, stage):
        self.optimizer_G.zero_grad()
        self.backward_G(stage)
        self.optimizer_G.step()

    def optimize_D_parameters(self):
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

    def print_current_error(self):
        print('loss G: {0} \t loss D: {1}'.format(self.loss_G.data[0], self.loss_D.data[0]))

    def save(self, epoch):
        self.save_network(self.Decoder, 'Decoder', epoch, self.opt.gpu_ids)
        self.save_network(self.D, 'D', epoch, self.opt.gpu_ids)

    def save_result(self, epoch=None):
        for i, syn_img_p in enumerate(self.syn_image_p.data):
            img_p = self.image_p.data[i]
            target_label = self.real_expr.data[i]
            filename = 'image_p_' + str(i) + '_' + str(target_label) + '.png'

            if epoch:
                filename = 'epoch{0}_{1}'.format(epoch, filename)

            path = os.path.join('/path/to/profile', filename)

            img_p = img_p / 127.5 - 1
            syn_img_p = syn_img_p / 127.5 - 1
            img_p = Tensor2Image(img_p)
            syn_img_p = Tensor2Image(syn_img_p)

            width, height = img_p.size
            result_img = Image.new(img_p.mode, (width * 2, height))
            result_img.paste(img_p, (0, 0, width, height))
            result_img.paste(syn_img_p, box=(width, 0))
            result_img.save(path, quality=95)

        for i, syn_img_f in enumerate(self.syn_image_f.data):
            img_f = self.image_f.data[i]
            target_label = self.real_expr.data[i]
            filename = 'image_f_' + str(i) + '_' + str(target_label) + '.png'

            if epoch:
                filename = 'epoch{0}_{1}'.format(epoch, filename)
            path = os.path.join('/path/to/frontal', filename)

            img_f = img_f / 127.5 - 1
            syn_img_f = syn_img_f / 127.5 - 1
            img_f = Tensor2Image(img_f)
            syn_img_f = Tensor2Image(syn_img_f)

            width, height = img_f.size
            result_img = Image.new(img_f.mode, (width * 2, height))
            result_img.paste(img_f, (0, 0, width, height))
            result_img.paste(syn_img_f, box=(width, 0))
            result_img.save(path, quality=95)