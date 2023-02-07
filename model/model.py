import logging
from collections import OrderedDict
import itertools
import torch
import torch.nn as nn
import os, glob
import model.networks as networks
from .base_model import BaseModel
logger = logging.getLogger('base')
from . import metrics as Metrics
from util.image_pool import ImagePool

class DARM(BaseModel):
    def __init__(self, opt):
        super(DARM, self).__init__(opt)
        # define network and load pretrained models
        self.netG = self.set_device(networks.define_G(opt))
        self.netD_s = self.set_device(networks.define_D(opt))
        self.netD_a = self.set_device(networks.define_D(opt))
        self.schedule_phase = None
        self.centered = opt['datasets']['train']['centered']

        # set loss and load resume state
        self.set_loss()
        self.set_new_noise_schedule(opt['model']['beta_schedule']['train'], schedule_phase='train')
        self.load_network()
        if self.opt['phase'] == 'train':
            self.netG.train()
            self.fake_I_pool = ImagePool(50)
            self.fake_L_pool = ImagePool(50)
            # find the parameters to optimize
            if opt['model']['finetune_norm']:
                optim_params = []
                for k, v in self.netG.named_parameters():
                    v.requires_grad = False
                    if k.find('transformer') >= 0:
                        v.requires_grad = True
                        v.data.zero_()
                        optim_params.append(v)
                        logger.info(
                            'Params [{:s}] initialized to 0 and will optimize.'.format(k))
            else:
                optim_params = list(self.netG.parameters())

            self.optG = torch.optim.Adam(optim_params, lr=opt['train']["optimizer"]["lr"], betas=(0.5, 0.999))
            self.optD = torch.optim.Adam(itertools.chain(self.netD_s.parameters(), self.netD_a.parameters()),
                                         lr=opt['train']["optimizer"]["lr"], betas=(0.5, 0.999))

            self.load_opt()
            self.log_dict = OrderedDict()
        self.print_network(self.netG)
        self.print_network(self.netD_s)

    def feed_data(self, data):
        self.data = self.set_device(data)

    def backward_D_basic(self, netD, real, fake):
        # Real
        pred_real = netD(real)
        loss_D_real = self.netG.loss_gan(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.netG.loss_gan(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def optimize_parameters(self):
        h_alpha = 0.2
        h_beta = 5.0
        self.optG.zero_grad()
        output, [l_dif, l_cyc] = self.netG(self.data)

        self.A_noisy, self.A_latent, self.B_noisy, self.B_latent, self.segm_V, self.synt_A, self.recn_F = output
        l_cyc = l_cyc * h_beta
        l_adv_Gs = self.netG.loss_gan(self.netD_s(self.segm_V), True) * h_alpha
        l_adv_Ga = self.netG.loss_gan(self.netD_a(self.synt_A), True) * h_alpha
        l_tot = l_dif + l_cyc + l_adv_Gs + l_adv_Ga
        l_tot.backward()
        self.optG.step()

        self.optD.zero_grad()  # set D_A and D_B's gradients to zero
        segm_V = self.fake_L_pool.query(self.segm_V)
        l_adv_Ds = self.backward_D_basic(self.netD_s, self.data['F'], segm_V) * h_alpha
        synt_A = self.fake_I_pool.query(self.synt_A)
        l_adv_Da = self.backward_D_basic(self.netD_a, self.data['A'], synt_A) * h_alpha
        self.optD.step()

        # set log
        self.log_dict['l_tot'] = l_tot.item()
        self.log_dict['l_dif'] = l_dif.item()
        self.log_dict['l_cyc'] = l_cyc.item()
        self.log_dict['l_adv_Gs'] = l_adv_Gs.item()
        self.log_dict['l_adv_Ga'] = l_adv_Ga.item()
        self.log_dict['l_adv_Ds'] = l_adv_Ds.item()
        self.log_dict['l_adv_Da'] = l_adv_Da.item()

    def test(self, continous=False):
        self.netG.eval()
        if isinstance(self.netG, nn.DataParallel):
            self.sample, self.test_A, self.test_V = self.netG.module.sample(self.data, continous)
        else:
            self.sample, self.test_A, self.test_V = self.netG.sample(self.data, continous)
        self.netG.train()

    def test_segment(self):
        self.netG.eval()
        if isinstance(self.netG, nn.DataParallel):
            self.test_V = self.netG.module.segment(self.data)
        else:
            self.test_V = self.netG.segment(self.data)
        self.netG.train()

    def set_loss(self):
        if isinstance(self.netG, nn.DataParallel):
            self.netG.module.set_loss(self.device)
        else:
            self.netG.set_loss(self.device)

    def set_new_noise_schedule(self, schedule_opt, schedule_phase='train'):
        if self.schedule_phase is None or self.schedule_phase != schedule_phase:
            self.schedule_phase = schedule_phase
            if isinstance(self.netG, nn.DataParallel):
                self.netG.module.set_new_noise_schedule(
                    schedule_opt, self.device)
            else:
                self.netG.set_new_noise_schedule(schedule_opt, self.device)

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, isTrain=True):
        out_dict = OrderedDict()
        if self.centered:
            min_max = (-1, 1)
        else:
            min_max = (0, 1)

        out_dict['dataA'] = Metrics.tensor2im(self.data['A'][0].detach().float().cpu(), min_max=min_max)
        out_dict['dataB'] = Metrics.tensor2im(self.data['B'][0].detach().float().cpu(), min_max=min_max)
        out_dict['dataF'] = Metrics.tensor2im(self.data['F'][0].detach().float().cpu(), min_max=min_max)
        out_dict['A_noisy'] = Metrics.tensor2im(self.A_noisy[0].detach().float().cpu(), min_max=min_max)
        out_dict['A_latent'] = Metrics.tensor2im(self.A_latent[0].detach().float().cpu(), min_max=min_max)
        out_dict['B_noisy'] = Metrics.tensor2im(self.B_noisy[0].detach().float().cpu(), min_max=min_max)
        out_dict['B_latent'] = Metrics.tensor2im(self.B_latent[0].detach().float().cpu(), min_max=min_max)
        out_dict['segm_V'] = Metrics.tensor2im(self.segm_V[0].detach().float().cpu(), min_max=min_max)
        out_dict['synt_A'] = Metrics.tensor2im(self.synt_A[0].detach().float().cpu(), min_max=min_max)
        out_dict['recn_F'] = Metrics.tensor2im(self.recn_F[0].detach().float().cpu(), min_max=min_max)

        if not isTrain:
            out_dict['SAM'] = Metrics.tensor2im(self.sample[0].detach().float().cpu(), min_max=min_max)
            out_dict['test_A'] = Metrics.tensor2im(self.test_A[0].detach().float().cpu(), min_max=min_max)
            out_dict['test_V'] = Metrics.tensor2im(self.test_V[0].detach().float().cpu(), min_max=min_max)
        return out_dict

    def get_current_sample(self):
        out_dict = OrderedDict()
        out_dict['SAM'] = self.sample.detach().float().cpu()
        out_dict['test_A'] = self.test_A.detach().float().cpu()
        out_dict['test_V'] = self.test_V.detach().float().cpu()
        return out_dict

    def get_current_segment(self):
        out_dict = OrderedDict()
        out_dict['test_V'] = self.test_V.detach().float().cpu()
        return out_dict

    def print_network(self, net):
        s, n = self.get_network_description(net)
        if isinstance(net, nn.DataParallel):
            net_struc_str = '{} - {}'.format(net.__class__.__name__,
                                             net.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(net.__class__.__name__)

        logger.info(
            'Network structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
        logger.info(s)

    def save_network(self, epoch, iter_step, seg_save=False, dice=0):
        if not seg_save:
            G_path = os.path.join(self.opt['path']['checkpoint'], 'I{}_E{}_G.pth'.format(iter_step, epoch))
            Ds_path = os.path.join(self.opt['path']['checkpoint'], 'I{}_E{}_Ds.pth'.format(iter_step, epoch))
            Da_path = os.path.join(self.opt['path']['checkpoint'], 'I{}_E{}_Da.pth'.format(iter_step, epoch))
            optG_path = os.path.join(self.opt['path']['checkpoint'], 'I{}_E{}_optG.pth'.format(iter_step, epoch))
            optD_path = os.path.join(self.opt['path']['checkpoint'], 'I{}_E{}_optD.pth'.format(iter_step, epoch))
        else:
            segPath = glob.glob(os.path.join(self.opt['path']['checkpoint'], 'D*'))
            for idx in range(len(segPath)):
                os.remove(segPath[idx])
            G_path = os.path.join(self.opt['path']['checkpoint'], 'D{}_I{}_E{}_G.pth'.format(dice, iter_step, epoch))
            Ds_path = os.path.join(self.opt['path']['checkpoint'], 'D{}_I{}_E{}_Ds.pth'.format(dice, iter_step, epoch))
            Da_path = os.path.join(self.opt['path']['checkpoint'], 'D{}_I{}_E{}_Da.pth'.format(dice, iter_step, epoch))
            optG_path = os.path.join(self.opt['path']['checkpoint'], 'D{}_I{}_E{}_optG.pth'.format(dice, iter_step, epoch))
            optD_path = os.path.join(self.opt['path']['checkpoint'], 'D{}_I{}_E{}_optD.pth'.format(dice, iter_step, epoch))

        # G
        network = self.netG
        if isinstance(self.netG, nn.DataParallel):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, G_path, _use_new_zipfile_serialization=False)
        # D_s
        network = self.netD_s
        if isinstance(self.netD_s, nn.DataParallel):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, Ds_path, _use_new_zipfile_serialization=False)
        # D_a
        network = self.netD_a
        if isinstance(self.netD_a, nn.DataParallel):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, Da_path, _use_new_zipfile_serialization=False)

        # opt
        opt_state = {'epoch': epoch, 'iter': iter_step, 'scheduler': None, 'optimizer': None}
        opt_state['optimizer'] = self.optG.state_dict()
        torch.save(opt_state, optG_path)
        opt_state = {'epoch': epoch, 'iter': iter_step, 'scheduler': None, 'optimizer': None}
        opt_state['optimizer'] = self.optD.state_dict()
        torch.save(opt_state, optD_path)

        logger.info('Saved model in [{:s}] ...'.format(G_path))

    def load_network(self):
        load_path = self.opt['path']['resume_state']
        if load_path is not None:
            logger.info('Loading pretrained model for G [{:s}] ...'.format(load_path))
            G_path = '{}_G.pth'.format(load_path)

            network = self.netG
            if isinstance(self.netG, nn.DataParallel):
                network = network.module
            network.load_state_dict(torch.load(G_path), strict=(not self.opt['model']['finetune_norm']))

            if self.opt['phase'] == 'train':
                Ds_path = '{}_Ds.pth'.format(load_path)
                Da_path = '{}_Da.pth'.format(load_path)
                network = self.netD_s
                if isinstance(self.netD_s, nn.DataParallel):
                    network = network.module
                network.load_state_dict(torch.load(Ds_path), strict=(not self.opt['model']['finetune_norm']))
                network = self.netD_a
                if isinstance(self.netD_a, nn.DataParallel):
                    network = network.module
                network.load_state_dict(torch.load(Da_path), strict=(not self.opt['model']['finetune_norm']))

    def load_opt(self):
        load_path = self.opt['path']['resume_state']
        if load_path is not None:
            logger.info(
                'Loading pretrained model for G [{:s}] ...'.format(load_path))
            optG_path = '{}_optG.pth'.format(load_path)
            optD_path = '{}_optD.pth'.format(load_path)

            # optimizer
            optG = torch.load(optG_path)
            self.optG.load_state_dict(optG['optimizer'])
            self.begin_step = optG['iter']
            self.begin_epoch = optG['epoch']
            optD = torch.load(optD_path)
            self.optD.load_state_dict(optD['optimizer'])
            self.begin_step = optD['iter']
            self.begin_epoch = optD['epoch']
