import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torchvision import transforms
import numpy as np

def Tensor2Image(img):
    img = img.cpu()
    img = img * 0.5 + 0.5
    img = transforms.ToPILImage()(img)
    return img

def one_hot(label, depth):
    out_tensor = torch.zeros(len(label), depth)
    
    for i, index in enumerate(label):
        out_tensor[i][index] = 1.0
    return out_tensor

def one_hot_long_without_intensity(label, depth):
    out_tensor = -1*torch.ones(len(label), depth*5)
    
    for i, index in enumerate(label):
        out_tensor[i][index*5: (index+1)*5] *= -1
    return out_tensor

def one_hot_long(label, depth, set_intensity=None):
    out_tensor = torch.zeros(len(label), depth*5)
    
    for i, index in enumerate(label):
        if set_intensity:
            onehot = torch.Tensor(5).uniform_(0.1*set_intensity, 0.1*set_intensity)
        else:
            onehot = torch.Tensor(5).uniform_(-1, 1)
        out_tensor[i] = (-1*torch.abs(onehot)).repeat(depth)
        out_tensor[i][index*5: (index+1)*5] = torch.abs(onehot)
    return out_tensor

def one_hot_intensity(label, depth):
    out_tensor = torch.zeros(len(label), depth)
    intensity = [0.1, 0.2, 0.3, 0.4, 0.5 ,0.6, 0.7, 0.8, 0.9, 1.0]
    
    for i, index in enumerate(label):
        out_tensor[i][index] = intensity[np.random.randint(9)]
    return out_tensor

def concat_label(feature_map, label, duplicate=1):
    feature_shape = feature_map.shape
    if duplicate<1:
        return feature_map
    
    label = label.repeat(1, duplicate)
    label_shape = label.shape
    
    if len(feature_shape) == 2:# FC
        return torch.cat((feature_map, label), 1)
    elif len(feature_shape) == 4:# Conv or DConv
        label = label.view(feature_shape[0], label_shape[-1], 1, 1)
        return torch.cat((feature_map, label*torch.ones((feature_shape[0], label_shape[-1], feature_shape[2], feature_shape[3])).cuda()), 1)

def weights_init_normal(m):
    if isinstance(m, nn.ConvTranspose2d):
        init.normal_(m.weight.data, 0.0, 0.02)
    elif isinstance(m, nn.Conv2d):
        init.normal_(m.weight.data, 0.0, 0.02)
    elif isinstance(m, nn.Linear):
        init.normal_(m.weight.data, 0.0, 0.02)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

class Decoder(nn.Module):
    def __init__(self, N_expr=6, N_long=5, N_z=2048):
        super(Decoder, self).__init__()
        self.N_expr = N_expr
        self.N_long = N_long
        self.N_z = N_z
        self.duplicate = int(self.N_z//self.N_expr)# 341
        
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()
#         self.bn = nn.BatchNorm2d(2048)
        
#         # fc1   B*(2048+5*6*341)x1x1-->B*2048x7x7
#         self.fc1 = nn.Linear(self.N_z+self.N_long*self.N_expr*self.duplicate, 2048*7*7)
        
        # conv1   B*(2048+5*6*341)*7*7-->B*512*7*7           
        conv1 = [nn.Conv2d(2048+self.N_long*self.N_expr*self.duplicate, 512, kernel_size=1, stride=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True)]
        self.conv1 = nn.Sequential(*conv1)
        
        # res1_1   B*512*7*7-->B*512*7*7
        res1_1 = [nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
               nn.BatchNorm2d(512),
               nn.ReLU(inplace=True),
               nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
               nn.BatchNorm2d(512)]
        self.res1_1 = nn.Sequential(*res1_1)
        
        # res1_2   B*512*7*7-->B*512*7*7     
        res1_2 = [nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
               nn.BatchNorm2d(512),
               nn.ReLU(inplace=True),
               nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
               nn.BatchNorm2d(512)]
        self.res1_2 = nn.Sequential(*res1_2)
        
        # res1_3   B*512*7*7-->B*512*7*7   
        res1_3 = [nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
               nn.BatchNorm2d(512),
               nn.ReLU(inplace=True),
               nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
               nn.BatchNorm2d(512)]
        self.res1_3 = nn.Sequential(*res1_3)
        
        # res1_4   B*512*7*7-->B*512*7*7      
        res1_4 = [nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
               nn.BatchNorm2d(512),
               nn.ReLU(inplace=True),
               nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
               nn.BatchNorm2d(512)]
        self.res1_4 = nn.Sequential(*res1_4)
        
        # dconv2   B*(512+5*6)*7*7-->B*256*14*14      
        dconv2 = [nn.ReLU(inplace=True),
               nn.ConvTranspose2d(512+self.N_long*self.N_expr, 256, kernel_size=4, stride=2, padding=1),
               nn.BatchNorm2d(256),
               nn.ReLU(inplace=True)]
        self.dconv2 = nn.Sequential(*dconv2)
        
        # res2   B*256*14*14-->B*256*14*14
        res2 = [nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(256)]
        self.res2 = nn.Sequential(*res2)
        
        # dconv3   B*(256+5*6)*14*14-->B*128*28*28
        dconv3 = [nn.ReLU(inplace=True),
               nn.ConvTranspose2d(256+self.N_long*self.N_expr, 128, kernel_size=4, stride=2, padding=1),
               nn.BatchNorm2d(128),
               nn.ReLU(inplace=True)]
        self.dconv3 = nn.Sequential(*dconv3)
        
        # res3   B*128*28*28-->B*128*28*28    
        res3 = [nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(128)]
        self.res3 = nn.Sequential(*res3)
        
        # dconv4   B*(128+5*6)*28*28-->B*64*56*56   
        dconv4 = [nn.ReLU(inplace=True),
               nn.ConvTranspose2d(128+self.N_long*self.N_expr, 64, kernel_size=4, stride=2, padding=1),
               nn.BatchNorm2d(64),
               nn.ReLU(inplace=True)]
        self.dconv4 = nn.Sequential(*dconv4)
                
        # res4   B*64*56*56-->B*64*56*56   
        res4 = [nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64)]
        self.res4 = nn.Sequential(*res4)
        
        # dconv5   B*(64+5*6)*56*56-->B*32*112*112
        dconv5 = [nn.ReLU(inplace=True),
               nn.ConvTranspose2d(64+self.N_long*self.N_expr, 32, kernel_size=4, stride=2, padding=1),
               nn.BatchNorm2d(32),
               nn.ReLU(inplace=True)]
        self.dconv5 = nn.Sequential(*dconv5)
        
        # res5   B*32*112*112->B*32*112*112
        res5 = [nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(32)]
        self.res5 = nn.Sequential(*res5)
        
        # dconv6   B*(32+5*6)*112*112-->B*32*224*224 
        dconv6 = [nn.ReLU(inplace=True),
               nn.ConvTranspose2d(32+self.N_long*self.N_expr, 32, kernel_size=4, stride=2, padding=1),
               nn.BatchNorm2d(32),
               nn.ReLU(inplace=True)]
        self.dconv6 = nn.Sequential(*dconv6)
        
        # res6   B*32*224*224-->B*32*224*224
        res6 = [nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(32)]
        self.res6 = nn.Sequential(*res6)
        
        # cw_conv B*(32+5*6)*224*224-->3*224*224
        cw_conv = [nn.ReLU(inplace=True),
               nn.Conv2d(32+self.N_long*self.N_expr, 3, kernel_size=1, stride=1),
               nn.Tanh()]
        self.cw_conv = nn.Sequential(*cw_conv)
    
    def forward(self, input, expr):
        x = concat_label(input, expr, self.duplicate)
        x = self.conv1(x)
        y = self.res1_1(x)
        x = x+y
        x = self.relu(x)
        y = self.res1_2(x)
        x = x+y
        x = self.relu(x)
        y = self.res1_3(x)
        x = x+y 
        x = self.relu(x)
        y = self.res1_4(x)
        x = x+y
        
        x = concat_label(x, expr)
        x = self.dconv2(x)
        y = self.res2(x)
        x = x+y 
        
        x = concat_label(x, expr)
        x = self.dconv3(x)
        y = self.res3(x)
        x = x+y 
        
        x = concat_label(x, expr)
        x = self.dconv4(x)
        y = self.res4(x)
        x = x+y 
        
        x = concat_label(x, expr)
        x = self.dconv5(x)
        y = self.res5(x)
        x = x+y
        
        x = concat_label(x, expr)
        x = self.dconv6(x)
        y = self.res6(x)
        x = x+y
        
        x = concat_label(x, expr)
        x_final = self.cw_conv(x)
        
        return (x_final + 1) * 127.5
    
class Discriminator(nn.Module):
    def __init__(self, N_expr=6, N_long=5):
        super(Discriminator, self).__init__()
        self.N_expr = N_expr
        self.N_long = N_long
        self.lrelu = nn.LeakyReLU(inplace=True)
        self.tanh = nn.Tanh()
        
        #########adv##########
        self.d_conv1 = nn.Conv2d(3+N_expr, 32, kernel_size=4, stride=2, padding=1)
        
        d_conv1 = [nn.Conv2d(3+N_expr, 32, kernel_size=4, stride=2, padding=1),
                 nn.LeakyReLU(inplace=True)
                ]
        self.d_conv1 = nn.Sequential(*d_conv1)
        
        d_conv2 = [nn.Conv2d(32+N_expr, 64, kernel_size=4, stride=2, padding=1),
                 nn.LayerNorm(56),
                 nn.LeakyReLU(inplace=True)
                ]
        self.d_conv2 = nn.Sequential(*d_conv2)

        d_conv3 = [nn.Conv2d(64+N_expr, 128, kernel_size=4, stride=2, padding=1),
                 nn.LayerNorm(28),
                 nn.LeakyReLU(inplace=True)
                ]
        self.d_conv3 = nn.Sequential(*d_conv3)

        d_conv4 = [nn.Conv2d(128+N_expr, 256, kernel_size=4, stride=2, padding=1),
                 nn.LayerNorm(14),
                 nn.LeakyReLU(inplace=True)
                ]
        self.d_conv4 = nn.Sequential(*d_conv4)
        
        d_conv5 = [nn.Conv2d(256+N_expr, 256, kernel_size=4, stride=2, padding=1),
                 nn.LayerNorm(7),
                 nn.LeakyReLU(inplace=True)
                ]
        self.d_conv5 = nn.Sequential(*d_conv5)
                
        d_fc1 = [nn.Linear((256+N_expr)*7*7, 1024),
               nn.LeakyReLU(inplace=True)
              ]
        self.d_fc1 = nn.Sequential(*d_fc1)
        
        ##########adv############
        self.d_fc_adv = nn.Linear(1024+N_expr, 1)

        ##########expr############
        q_fc_shared = [nn.Linear(1024+N_expr, 128),
                       nn.LeakyReLU(inplace=True)
                      ]
        self.q_fc_shared = nn.Sequential(*q_fc_shared)
                
        q_fc_1 = [nn.Linear(128, 64),
                  nn.LeakyReLU(inplace=True),
                  nn.Linear(64, N_long),
                  nn.Tanh()
                 ]
        self.q_fc_1 = nn.Sequential(*q_fc_1)

        q_fc_2 = [nn.Linear(128, 64),
                  nn.LeakyReLU(inplace=True),
                  nn.Linear(64, N_long),
                  nn.Tanh()
                 ]
        self.q_fc_2 = nn.Sequential(*q_fc_2)
 
        q_fc_3 = [nn.Linear(128, 64),
                  nn.LeakyReLU(inplace=True),
                  nn.Linear(64, N_long),
                  nn.Tanh()
                 ]
        self.q_fc_3 = nn.Sequential(*q_fc_3)

        q_fc_4 = [nn.Linear(128, 64),
                  nn.LeakyReLU(inplace=True),
                  nn.Linear(64, N_long),
                  nn.Tanh()
                 ]
        self.q_fc_4 = nn.Sequential(*q_fc_4)

        q_fc_5 = [nn.Linear(128, 64),
                  nn.LeakyReLU(inplace=True),
                  nn.Linear(64, N_long),
                  nn.Tanh()
                 ]
        self.q_fc_5 = nn.Sequential(*q_fc_5)

        q_fc_6 = [nn.Linear(128, 64),
                  nn.LeakyReLU(inplace=True),
                  nn.Linear(64, N_long),
                  nn.Tanh()
                 ]
        self.q_fc_6 = nn.Sequential(*q_fc_6)
        
        
        ##########local###############        
        ###############eyes###################65.180
        conv_eye = [nn.Conv2d(3+N_expr, 32, kernel_size=3, stride=2, padding=1),
                   nn.LeakyReLU(inplace=True),
                   nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                   nn.LayerNorm([17,45]),
                   nn.LeakyReLU(inplace=True),
                   nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                   nn.LayerNorm([9, 23]),
                   nn.LeakyReLU(inplace=True),
                   nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
                   nn.LayerNorm([5, 12]),
                   nn.LeakyReLU(inplace=True)]
        
        self.conv_eye = nn.Sequential(*conv_eye)
        
        self.fc_eye = nn.Linear(256*5*12, 1)
        
        ###############nose###################70.80
        conv_nose = [nn.Conv2d(3+N_expr, 32, kernel_size=3, stride=2, padding=1),
                   nn.LeakyReLU(inplace=True),
                   nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                   nn.LayerNorm([18, 20]),
                   nn.LeakyReLU(inplace=True),
                   nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                   nn.LayerNorm([9, 10]),
                   nn.LeakyReLU(inplace=True),
                   nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
                   nn.LayerNorm([5, 5]),
                   nn.LeakyReLU(inplace=True)]
        
        self.conv_nose = nn.Sequential(*conv_nose)
        
        self.fc_nose = nn.Linear(256*5*5, 1)
        
        ###############mouth###################80.120        
        conv_mouth = [nn.Conv2d(3+N_expr, 32, kernel_size=3, stride=2, padding=1),
                   nn.LeakyReLU(inplace=True),
                   nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                   nn.LayerNorm([20, 30]),
                   nn.LeakyReLU(inplace=True),
                   nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                   nn.LayerNorm([10, 15]),
                   nn.LeakyReLU(inplace=True),
                   nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
                   nn.LayerNorm([5, 8]),
                   nn.LeakyReLU(inplace=True)]
        
        self.conv_mouth = nn.Sequential(*conv_mouth)
        
        self.fc_mouth = nn.Linear(256*5*8, 1)
        
        ###############face###################170.180
        conv_face = [nn.Conv2d(3+N_expr, 32, kernel_size=3, stride=2, padding=1),
                   nn.LeakyReLU(inplace=True),
                   nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                   nn.LayerNorm([43, 45]),
                   nn.LeakyReLU(inplace=True),
                   nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                   nn.LayerNorm([22, 23]),
                   nn.LeakyReLU(inplace=True),
                   nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
                   nn.LayerNorm([11, 12]),
                   nn.LeakyReLU(inplace=True)]
        
        self.conv_face = nn.Sequential(*conv_face)
        
        self.fc_face = nn.Linear(256*11*12, 1)
       
    def forward(self, input, expr):
        images = input / 127.5 - 1
        images = concat_label(images, expr)
        ########local###########
        image = images
        
        eye = images[:, :, 45:45+65, 20:20+180]
        nose = images[:, :, 75:75+70, 70:70+80]
        mouth = images[:, :, 130:130+80, 60:60+120]
        face = images[:, :, 40:40+170, 25:25+180]

        #############eye##################
        x = self.conv_eye(eye)
        x = x.view(-1, 256*5*12)
        x_e = self.fc_eye(x)

        ############nose################
        x = self.conv_nose(nose)
        x = x.view(-1, 256*5*5)
        x_n = self.fc_nose(x)
        
        ############mouth################
        x = self.conv_mouth(mouth)
        x = x.view(-1, 256*5*8)
        x_m = self.fc_mouth(x)
        
        ###########face#####################
        x = self.conv_face(face)
        x = x.view(-1, 256*11*12)
        x_f = self.fc_face(x)
        
        #########adv############
        x = self.d_conv1(image)
        x = concat_label(x, expr)
        
        x = self.d_conv2(x)
        x = concat_label(x, expr)

        x = self.d_conv3(x)
        x = concat_label(x, expr)

        x = self.d_conv4(x)
        x = concat_label(x, expr)

        x = self.d_conv5(x)
        x = concat_label(x, expr)
        x = x.view(-1, (256+self.N_expr)*7*7)
        x = self.d_fc1(x)
        x_1024 = concat_label(x, expr)
        
        x_adv = self.d_fc_adv(x_1024)
        
        q_shared = self.q_fc_shared(x_1024)
        
        cat1 = self.q_fc_1(q_shared)
        cat2 = self.q_fc_2(q_shared)
        cat3 = self.q_fc_3(q_shared)
        cat4 = self.q_fc_4(q_shared)
        cat5 = self.q_fc_5(q_shared)
        cat6 = self.q_fc_6(q_shared)
        
        cats = []
        cats.append(cat1)
        cats.append(cat2) 
        cats.append(cat3) 
        cats.append(cat4)
        cats.append(cat5) 
        cats.append(cat6) 
                
        return x_adv, cats, x_e, x_n, x_m, x_f
