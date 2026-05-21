"""The base model definition.

This module defines the abstract meta model class and base model class. In the base model,
 we define the basic model functions, like get_loader, build_network, and run_train, etc.
 The api of the base model is run_train and run_test, they are used in `opengait/main.py`.

Typical usage:

BaseModel.run_train(model)
BaseModel.run_test(model)
"""
import torch
import numpy as np
import os.path as osp
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as tordata

from tqdm import tqdm
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler
from abc import ABCMeta
from abc import abstractmethod

from . import backbones
from .loss_aggregator import LossAggregator
from data.transform import get_transform
from data.collate_fn import CollateFn
from data.dataset import DataSet
import data.sampler as Samplers
from utils import Odict, mkdir, ddp_all_gather
from utils import get_valid_args, is_list, is_dict, np2var, ts2np, list2var, get_attr_from
from evaluation import evaluator as eval_functions
from utils import NoOp
from utils import get_msg_mgr




# add path
import spikingjelly.activation_based.functional as functional

__all__ = ['BaseModel']


class MetaModel(metaclass=ABCMeta):
    """The necessary functions for the base model.

    This class defines the necessary functions for the base model, in the base model, we have implemented them.
    """
    @abstractmethod
    def get_loader(self, data_cfg):
        """Based on the given data_cfg, we get the data loader."""
        raise NotImplementedError

    @abstractmethod
    def build_network(self, model_cfg):
        """Build your network here."""
        raise NotImplementedError

    @abstractmethod
    def init_parameters(self):
        """Initialize the parameters of your network."""
        raise NotImplementedError

    @abstractmethod
    def get_optimizer(self, optimizer_cfg):
        """Based on the given optimizer_cfg, we get the optimizer."""
        raise NotImplementedError

    @abstractmethod
    def get_scheduler(self, scheduler_cfg):
        """Based on the given scheduler_cfg, we get the scheduler."""
        raise NotImplementedError

    @abstractmethod
    def save_ckpt(self, iteration):
        """Save the checkpoint, including model parameter, optimizer and scheduler."""
        raise NotImplementedError

    @abstractmethod
    def resume_ckpt(self, restore_hint):
        """Resume the model from the checkpoint, including model parameter, optimizer and scheduler."""
        raise NotImplementedError

    @abstractmethod
    def inputs_pretreament(self, inputs):
        """Transform the input data based on transform setting."""
        raise NotImplementedError

    @abstractmethod
    def train_step(self, loss_num) -> bool:
        """Do one training step."""
        raise NotImplementedError

    @abstractmethod
    def inference(self):
        """Do inference (calculate features.)."""
        raise NotImplementedError

    @abstractmethod
    def run_train(model):
        """Run a whole train schedule."""
        raise NotImplementedError

    @abstractmethod
    def run_test(model):
        """Run a whole test schedule."""
        raise NotImplementedError


class BaseModel(MetaModel, nn.Module):
    """Base model.

    This class inherites the MetaModel class, and implements the basic model functions, like get_loader, build_network, etc.

    Attributes:
        msg_mgr: the massage manager.
        cfgs: the configs.
        iteration: the current iteration of the model.
        engine_cfg: the configs of the engine(train or test).
        save_path: the path to save the checkpoints.
    """

    def __init__(self, cfgs, training):
        """Initialize the base model.

        Complete the model initialization, including the data loader, the network, the optimizer, the scheduler, the loss.

        Args:
        cfgs:
            All of the configs.
        training:
            Whether the model is in training mode.
        """

        super(BaseModel, self).__init__()
        self.msg_mgr = get_msg_mgr()
        self.cfgs = cfgs
        self.iteration = 0
        self.engine_cfg = cfgs['trainer_cfg'] if training else cfgs['evaluator_cfg']


        if self.engine_cfg is None:
            raise Exception("Initialize a model without -Engine-Cfgs-")

        if training and self.engine_cfg['enable_float16']:
            self.Scaler = GradScaler()
            
        self.save_path = osp.join('output/', cfgs['data_cfg']['dataset_name'],
                                    cfgs['model_cfg']['model'], self.engine_cfg['save_name'])

        self.build_network(cfgs['model_cfg'])       # 如果yaml里有backbone-cfg字样，则调用backbones.py里的backbone赋值给self.Backbone
        self.init_parameters()
        self.trainer_trfs = get_transform(cfgs['trainer_cfg']['transform'])
        self.evtrainer_trfs = get_transform(cfgs['trainer_cfg']['evtransform']) if 'evtransform' in cfgs['trainer_cfg'] else None
        self.rgbtrainer_trfs = get_transform(cfgs['trainer_cfg']['rgbtransform']) if 'rgbtransform' in cfgs['trainer_cfg'] else None

        self.msg_mgr.log_info(cfgs['data_cfg'])
        if training:
            self.train_loader = self.get_loader(
                cfgs['data_cfg'], train=True)
        if not training or self.engine_cfg['with_test']:
            self.test_loader = self.get_loader(
                cfgs['data_cfg'], train=False)
            self.evaluator_trfs = get_transform(
                cfgs['evaluator_cfg']['transform'])
            self.evevaluator_trfs = get_transform(cfgs['evaluator_cfg']['evtransform']) if 'evtransform' in cfgs['evaluator_cfg'] else None
            self.rgbevaluator_trfs = get_transform(cfgs['evaluator_cfg']['rgbtransform']) if 'rgbtransform' in cfgs['evaluator_cfg'] else None

        self.device = torch.distributed.get_rank()
        torch.cuda.set_device(self.device)
        self.to(device=torch.device(
            "cuda", self.device))

        if training:
            self.loss_aggregator = LossAggregator(cfgs['loss_cfg'])
            self.optimizer = self.get_optimizer(self.cfgs['optimizer_cfg'])
            self.scheduler = self.get_scheduler(cfgs['scheduler_cfg'])
        self.train(training)
        restore_hint = self.engine_cfg['restore_hint']      # 断点恢复
        if restore_hint != 0:
            self.resume_ckpt(restore_hint)

    def get_backbone(self, backbone_cfg):
        """Get the backbone of the model."""
        if is_dict(backbone_cfg):
            Backbone = get_attr_from([backbones], backbone_cfg['type'])
            valid_args = get_valid_args(Backbone, backbone_cfg, ['type'])
            return Backbone(**valid_args)
        if is_list(backbone_cfg):
            Backbone = nn.ModuleList([self.get_backbone(cfg)
                                      for cfg in backbone_cfg])
            return Backbone
        raise ValueError(
            "Error type for -Backbone-Cfg-, supported: (A list of) dict.")

    def check_nan_inf(self):
        for name, p in self.named_parameters():
            if p.grad is not None and torch.isnan(p.grad).any():
                print(f"NaN first found in grad of {name}")
                return True
        return False

    def build_network(self, model_cfg):
        if 'backbone_cfg' in model_cfg.keys():
            self.Backbone = self.get_backbone(model_cfg['backbone_cfg'])

    def init_parameters(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Conv2d, nn.Conv1d)):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, (nn.BatchNorm3d, nn.BatchNorm2d, nn.BatchNorm1d)):
                if m.affine:
                    nn.init.normal_(m.weight.data, 1.0, 0.02)
                    nn.init.constant_(m.bias.data, 0.0)

    def get_loader(self, data_cfg, train=True):
        sampler_cfg = self.cfgs['trainer_cfg']['sampler'] if train else self.cfgs['evaluator_cfg']['sampler']
        dataset = DataSet(data_cfg, train)

        Sampler = get_attr_from([Samplers], sampler_cfg['type'])
        vaild_args = get_valid_args(Sampler, sampler_cfg, free_keys=[
            'sample_type', 'type'])
        sampler = Sampler(dataset, **vaild_args)

        loader = tordata.DataLoader(
            dataset=dataset,
            batch_sampler=sampler,
            collate_fn=CollateFn(dataset.label_set, sampler_cfg),
            pin_memory=True,
            num_workers=data_cfg['num_workers'])
        return loader

    def get_optimizer(self, optimizer_cfg):
        self.msg_mgr.log_info(optimizer_cfg)
        optimizer = get_attr_from([optim], optimizer_cfg['solver'])
        valid_arg = get_valid_args(optimizer, optimizer_cfg, ['solver'])
        optimizer = optimizer(
            filter(lambda p: p.requires_grad, self.parameters()), **valid_arg)
        return optimizer

    def get_scheduler(self, scheduler_cfg):
        self.msg_mgr.log_info(scheduler_cfg)
        Scheduler = get_attr_from(
            [optim.lr_scheduler], scheduler_cfg['scheduler'])
        valid_arg = get_valid_args(Scheduler, scheduler_cfg, ['scheduler'])
        scheduler = Scheduler(self.optimizer, **valid_arg)
        return scheduler

    def save_ckpt(self, iteration):
        if torch.distributed.get_rank() == 0:
            mkdir(osp.join(self.save_path, "checkpoints/"))
            save_name = self.engine_cfg['save_name']
            checkpoint = {
                'model': self.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict(),
                'iteration': iteration}
            torch.save(checkpoint,
                       osp.join(self.save_path, 'checkpoints/{}-{:0>5}.pt'.format(save_name, iteration)))
            print(f'model save to {self.save_path}!')

    def _load_ckpt(self, save_name):
        load_ckpt_strict = self.engine_cfg['restore_ckpt_strict']

        checkpoint = torch.load(save_name, map_location=torch.device(
            "cuda", self.device))
        model_state_dict = checkpoint['model']

        if not load_ckpt_strict:
            self.msg_mgr.log_info("-------- Restored Params List --------")
            self.msg_mgr.log_info(sorted(set(model_state_dict.keys()).intersection(
                set(self.state_dict().keys()))))
        try:
            self.load_state_dict(model_state_dict, strict=load_ckpt_strict)
            # 如果参数对不上，把SeparateBNNecks层的参数去掉再加载
        except RuntimeError as e:
            self.msg_mgr.log_warning(str(e))
            self.msg_mgr.log_info("Try to load model without SeparateBNNecks parameters...")
            self.init_parameters()
            for name, param in model_state_dict.items():
                if 'teacher' in name:
                    pass        # 不加载teacher的参数，因为后续会直接load teacher的checkpoint
                elif 'classifier.7' in name:        # 这里只针对evgait
                    pass        # 不加载classifier的参数，因为类别数可能不一样
                elif 'BNNecks' in name:
                    pass        # 不加载SeparateBNNecks的参数
                else:
                    if name in dict(self.named_parameters()).keys():
                        dict(self.named_parameters())[name].data.copy_(param.data)
                    # print(f'loaded {name} from checkpoint')
            del checkpoint['optimizer'], checkpoint['scheduler'], checkpoint['iteration']
        if self.training:
            if not self.engine_cfg["optimizer_reset"] and 'optimizer' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            else:
                self.msg_mgr.log_warning(
                    "Restore NO Optimizer from %s !!!" % save_name)
            if not self.engine_cfg["scheduler_reset"] and 'scheduler' in checkpoint:
                self.scheduler.load_state_dict(
                    checkpoint['scheduler'])
            else:
                self.msg_mgr.log_warning(
                    "Restore NO Scheduler from %s !!!" % save_name)
        self.msg_mgr.log_info("Restore Parameters from %s !!!" % save_name)
        # 如果self对象有init_from_ckpt函数，则调用它
        if hasattr(self, 'init_from_ckpt'):
            _part_ckpt = self.init_from_ckpt(checkpoint)
            self.msg_mgr.log_info(f'init {_part_ckpt} from ckpt')

    def resume_ckpt(self, restore_hint):
        if isinstance(restore_hint, int):
            save_name = self.engine_cfg['save_name']
            save_name = osp.join(
                '.'+self.save_path, 'checkpoints/{}-{:0>5}.pt'.format(save_name, restore_hint))
            self.iteration = restore_hint
        elif isinstance(restore_hint, str):
            save_name = restore_hint
            self.iteration = 0
        else:
            raise ValueError(
                "Error type for -Restore_Hint-, supported: int or string.")
        self._load_ckpt(save_name)

    def fix_BN(self):
        for module in self.modules():
            classname = module.__class__.__name__
            if classname.find('BatchNorm') != -1:
                module.eval()

    def inputs_pretreament(self, inputs):
        """Conduct transforms on input data.

        Args:
            inputs: the input data.
        Returns:
            tuple: training data including inputs, labels, and some meta data.
        """
        # print(np.asarray(inputs[0]).shape)   # (1, p * k, l, 2, 128, 128, 3)
        # print(np.asarray(inputs[1]).shape)   # (p*k)
        # print(np.asarray(inputs[2]).shape)   # (p*k)
        # print(np.asarray(inputs[3]).shape)   # (p*k)
        # print(np.asarray(inputs[4]).shape)   # ()
        # print(np.asarray(inputs[5]).shape)   # (1, p * k, l, 6, 128, 128, 3)
        events_batch = None
        rgb_batch = None
        if len(inputs) == 7:        # two features
            seqs_batch, labs_batch, typs_batch, vies_batch, seqL_batch, events_batch, rgb_batch = inputs      # [(1, p*k, l, 2, 128, 128, 3), (p*k), (p*k), (p*k)]
            if events_batch is None or rgb_batch is None:
                print('there is a bug')
        elif len(inputs) == 6:
            rgb_in_use = self.cfgs['data_cfg'].get('rgb_in_use')
            if rgb_in_use is not None:
                seqs_batch, labs_batch, typs_batch, vies_batch, seqL_batch, rgb_batch = inputs
            else:
                seqs_batch, labs_batch, typs_batch, vies_batch, seqL_batch, events_batch = inputs
        else:
            seqs_batch, labs_batch, typs_batch, vies_batch, seqL_batch = inputs     
        seq_trfs = self.trainer_trfs if self.training else self.evaluator_trfs
        if len(seqs_batch) != len(seq_trfs):       
            raise ValueError(
                "The number of types of input data and transform should be same. But got {} and {}".format(len(seqs_batch), len(seq_trfs)))
        requires_grad = bool(self.training)
        # print(np.asarray(seqs_batch).shape)     # (1, p * k, l, 128, 128, 3)
        seqs_batch = np.asarray(seqs_batch)
        if len(seqs_batch.shape) == 7:
            assert seqs_batch.shape[3] == 2, "The 4th dimension of seqs_batch should be 2."
            rgb_batch = seqs_batch[:, :, :, 1]
            seqs_batch = seqs_batch[:, :, :, 0]
        seqs = [np2var(np.asarray([trf(fra) for fra in seq]), requires_grad=requires_grad).float()
                for trf, seq in zip(seq_trfs, seqs_batch)]
        
        typs = typs_batch
        vies = vies_batch

        labs = list2var(labs_batch).long()

        if seqL_batch is not None:
            seqL_batch = np2var(seqL_batch).int()
        seqL = seqL_batch

        if seqL is not None:
            seqL_sum = int(seqL.sum().data.cpu().numpy())
            ipts = [_[:, :seqL_sum] for _ in seqs]
        else:
            ipts = seqs
        del seqs

        if events_batch is not None:
            evseq_trfs = self.evtrainer_trfs if self.training else self.evevaluator_trfs
            assert evseq_trfs is not None, "The events transform should be set in the config file."
            # print(np.asarray(events_batch).shape)   # 1, p * k, l, chunk_size, 128, 128, 3
            events = [np2var(np.asarray([trf(fra) for fra in seq]), requires_grad=requires_grad).float()
                      for trf, seq in zip(evseq_trfs, events_batch)]
            if seqL is not None:
                seqL_sum = int(seqL.sum().data.cpu().numpy())
                events = [_[:, :seqL_sum] for _ in events]
            ipts.append(events[0])
            del events

        if rgb_batch is not None:
            rgb_trfs = self.rgbtrainer_trfs if self.training else self.rgbevaluator_trfs
            assert rgb_trfs is not None, "The rgb transform should be set in the config"
            rgbs = [np2var(np.asarray([trf(fra) for fra in seq]), requires_grad=False).float() 
                    for trf, seq in zip(rgb_trfs, rgb_batch)]
            if seqL is not None:
                seqL_sum = int(seqL.sum().data.cpu().numpy())
                rgbs = [_[:, :seqL_sum] for _ in rgbs]
            ipts.append(rgbs[0])
            del rgbs
        # print("=============")
        # print(ipts[0].shape)    # torch.size([p*k, l, 3, 128, 128])
        # print(ipts[1].shape)    # torch.size([p*k, l * 6, 3, 128, 128])
        # print(ipts[2].shape)    # torch.size([p*k, l, 3, 128, 128])
        return ipts, labs, typs, vies, seqL

    def train_step(self, loss_sum) -> bool:
        """Conduct loss_sum.backward(), self.optimizer.step() and self.scheduler.step().

        Args:
            loss_sum:The loss of the current batch.
        Returns:
            bool: True if the training is finished, False otherwise.
        """

        self.optimizer.zero_grad()
        # print(loss_sum)
        if loss_sum <= 1e-9:
            self.msg_mgr.log_warning(
                "Find the loss sum less than 1e-9 but the training process will continue!")

        if self.engine_cfg['enable_float16']:
            self.Scaler.scale(loss_sum).backward()
            self.Scaler.step(self.optimizer)
            scale = self.Scaler.get_scale()
            self.Scaler.update()
            # Warning caused by optimizer skip when NaN
            # https://discuss.pytorch.org/t/optimizer-step-before-lr-scheduler-step-error-using-gradscaler/92930/5
            if scale != self.Scaler.get_scale():
                self.msg_mgr.log_debug("Training step skip. Expected the former scale equals to the present, got {} and {}".format(
                    scale, self.Scaler.get_scale()))
                return False
        else:
            loss_sum.backward()
            self.optimizer.step()


        self.iteration += 1
        self.scheduler.step()
        return True

    def inference(self, rank):
        """Inference all the test data.

        Args:
            rank: the rank of the current process.Transform
        Returns:
            Odict: contains the inference results.
        """
        total_size = len(self.test_loader)
        if rank == 0:
            pbar = tqdm(total=total_size, desc='Transforming')
        else:
            pbar = NoOp()
        batch_size = self.test_loader.batch_sampler.batch_size
        rest_size = total_size
        info_dict = Odict()
        for inputs in self.test_loader:
            ipts = self.inputs_pretreament(inputs)
            with autocast(enabled=self.engine_cfg['enable_float16']):
                retval = self.forward(ipts)
                inference_feat = retval['inference_feat']
                for k, v in inference_feat.items():
                    inference_feat[k] = ddp_all_gather(v, requires_grad=False)
                del retval
            for k, v in inference_feat.items():
                inference_feat[k] = ts2np(v)
            info_dict.append(inference_feat)
            rest_size -= batch_size
            if rest_size >= 0:
                update_size = batch_size
            else:
                update_size = total_size % batch_size
            pbar.update(update_size)
        pbar.close()
        for k, v in info_dict.items():
            v = np.concatenate(v)[:total_size]
            info_dict[k] = v
        return info_dict
    
    @ staticmethod
    def run_train(model):
        """Accept the instance object(model) here, and then run the train loop."""
        for inputs in model.train_loader:   # [(1, 1, 768, 128, 128, 4), (128), (128), (128)]
            # print([len(data) for data in inputs if data is not None])
            # check lock file
            ipts = model.inputs_pretreament(inputs)
            with autocast(enabled=model.engine_cfg['enable_float16']):
                retval = model(ipts)
                training_feat, visual_summary = retval['training_feat'], retval['visual_summary']
                del retval
            loss_sum, loss_info = model.loss_aggregator(training_feat)      # 计算loss
            ok = model.train_step(loss_sum)
            functional.reset_net(model)

            if not ok:
                continue

            visual_summary.update(loss_info)
            visual_summary['scalar/learning_rate'] = model.optimizer.param_groups[0]['lr']

            model.msg_mgr.train_step(loss_info, visual_summary)
            if model.iteration % model.engine_cfg['save_iter'] == 0:
                # save the checkpoint
                model.save_ckpt(model.iteration)

                # run test if with_test = true
                if model.engine_cfg['with_test']:
                    model.msg_mgr.log_info("Running test...")
                    model.eval()
                    result_dict = BaseModel.run_test(model)
                    model.train()
                    if model.cfgs['trainer_cfg']['fix_BN']:
                        model.fix_BN()
                    if result_dict:
                        model.msg_mgr.write_to_tensorboard(result_dict)
                    model.msg_mgr.reset_time()
            if model.iteration >= model.engine_cfg['total_iter']:
                break


    @ staticmethod
    def run_test(model):
        """Accept the instance object(model) here, and then run the test loop."""
        evaluator_cfg = model.cfgs['evaluator_cfg']
        if torch.distributed.get_world_size() != evaluator_cfg['sampler']['batch_size']:
            raise ValueError("The batch size ({}) must be equal to the number of GPUs ({}) in testing mode!".format(
                evaluator_cfg['sampler']['batch_size'], torch.distributed.get_world_size()))
        rank = torch.distributed.get_rank()
        with torch.no_grad():
            info_dict = model.inference(rank)
        if rank == 0:
            loader = model.test_loader
            label_list = loader.dataset.label_list
            types_list = loader.dataset.types_list
            views_list = loader.dataset.views_list
            info_dict.update({
                'labels': label_list, 'types': types_list, 'views': views_list})

            if 'eval_func' in evaluator_cfg.keys():
                eval_func = evaluator_cfg["eval_func"]
            else:
                eval_func = 'identification'
            eval_func = getattr(eval_functions, eval_func)
            valid_args = get_valid_args(
                eval_func, evaluator_cfg, ['metric'])
            try:
                dataset_name = model.cfgs['data_cfg']['test_dataset_name']
            except:
                dataset_name = model.cfgs['data_cfg']['dataset_name']
            return eval_func(info_dict, dataset_name, **valid_args)
