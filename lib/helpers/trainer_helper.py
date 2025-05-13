import os
import tqdm

import torch
import numpy as np
import torch.nn as nn

from lib.helpers.save_helper import get_checkpoint_state
from lib.helpers.save_helper import load_checkpoint
from lib.helpers.save_helper import save_checkpoint

from utils import misc


class Trainer(object):
    def __init__(self,
                cfg,
                model,
                optimizer,
                train_loader,
                test_loader,
                lr_scheduler,
                warmup_lr_scheduler,
                logger,
                loss,
                model_name,
                teacher_model=None):
        self.cfg = cfg
        self.model = model
        self.teacher_model = teacher_model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.lr_scheduler = lr_scheduler
        self.warmup_lr_scheduler = warmup_lr_scheduler
        self.logger = logger
        self.epoch = 0
        self.best_result = 0
        self.best_epoch = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.detr_loss = loss
        self.model_name = model_name
        self.output_dir = os.path.join('./' + cfg['save_path'], model_name)
        self.tester = None

        if cfg.get('pretrain_model'):
            assert os.path.exists(cfg['pretrain_model'])
            load_checkpoint(model=self.model,
                            optimizer=None,
                            filename=cfg['pretrain_model'],
                            map_location=self.device,
                            logger=self.logger)

        if cfg.get('resume_model', None):
            resume_model_path = os.path.join(self.output_dir, "checkpoint.pth")
            assert os.path.exists(resume_model_path)
            self.epoch, self.best_result, self.best_epoch = load_checkpoint(
                model=self.model.to(self.device),
                optimizer=self.optimizer,
                filename=resume_model_path,
                map_location=self.device,
                logger=self.logger)
            self.lr_scheduler.last_epoch = self.epoch - 1
            self.logger.info("Loading Checkpoint... Best Result:{}, Best Epoch:{}".format(self.best_result, self.best_epoch))

    def train(self):
        start_epoch = self.epoch

        progress_bar = tqdm.tqdm(range(start_epoch, self.cfg['max_epoch']), dynamic_ncols=True, leave=True, desc='epochs')
        best_result = self.best_result
        best_epoch = self.best_epoch
        for epoch in range(start_epoch, self.cfg['max_epoch']):
            np.random.seed(np.random.get_state()[1][0] + epoch)
            self.train_one_epoch(epoch)
            self.epoch += 1

            if self.warmup_lr_scheduler is not None and epoch < 5:
                self.warmup_lr_scheduler.step()
            else:
                self.lr_scheduler.step()

            if (self.epoch % self.cfg['save_frequency']) == 0:
                os.makedirs(self.output_dir, exist_ok=True)
                if self.cfg['save_all']:
                    ckpt_name = os.path.join(self.output_dir, 'checkpoint_epoch_%d' % self.epoch)
                else:
                    ckpt_name = os.path.join(self.output_dir, 'checkpoint')

                save_checkpoint(
                    get_checkpoint_state(self.model, self.optimizer, self.epoch, best_result, best_epoch),
                    ckpt_name)

                if self.tester is not None:
                    self.logger.info("Test Epoch {}".format(self.epoch))
                    self.tester.inference()
                    cur_result = self.tester.evaluate()
                    if cur_result > best_result:
                        best_result = cur_result
                        best_epoch = self.epoch
                        ckpt_name = os.path.join(self.output_dir, 'checkpoint_best')
                        save_checkpoint(
                            get_checkpoint_state(self.model, self.optimizer, self.epoch, best_result, best_epoch),
                            ckpt_name)
                    self.logger.info("Best Result:{}, epoch:{}".format(best_result, best_epoch))

            progress_bar.update()

        self.logger.info("Best Result:{}, epoch:{}".format(best_result, best_epoch))
        return None

    def train_one_epoch(self, epoch):
        torch.set_grad_enabled(True)
        self.model.train()
        print(">>>>>>> Epoch:", str(epoch) + ":")

        progress_bar = tqdm.tqdm(total=len(self.train_loader), leave=(self.epoch+1 == self.cfg['max_epoch']), desc='iters')
        for batch_idx, (inputs, calibs, targets, info) in enumerate(self.train_loader):
            imgs_clean, imgs_foggy = inputs  # inputs를 분리
            imgs_clean = imgs_clean.to(self.device)
            imgs_foggy = imgs_foggy.to(self.device)
            calibs = calibs.to(self.device)
            img_sizes = targets['img_size'].to(self.device)

            targets_list = self.prepare_targets(targets, imgs_clean.shape[0])
            for t in targets_list:
                for k in t.keys():
                    t[k] = t[k].to(self.device)

            dn_args = None
            if self.cfg.get("use_dn", False) and self.model.use_dn:  # 모델의 use_dn도 확인
                dn_args = (targets_list, self.cfg['scalar'], self.cfg['label_noise_scale'], 
                           self.cfg['box_noise_scale'], self.cfg['num_patterns'])

            self.optimizer.zero_grad()
            
            # 1. Teacher (clean 이미지) - frozen
            with torch.no_grad():
                teacher_outputs = self.teacher_model(imgs_clean, calibs, targets_list, img_sizes)
            
            # 2. Student (foggy 이미지) - training
            student_outputs = self.model(imgs_foggy, calibs, targets_list, img_sizes, dn_args=dn_args)
            
            # 3. Losses
            # 3.1. Task loss (detection)
            detr_losses_dict = self.detr_loss(student_outputs, targets_list)
            weight_dict = self.detr_loss.weight_dict
            detr_losses = sum([detr_losses_dict[k] * weight_dict[k] for k in detr_losses_dict if k in weight_dict])
            
            # 3.2. Distillation loss - student 출력을 teacher 크기에 맞춤
            student_logits = student_outputs["pred_logits"].transpose(1, 2)  # [B, C, 550]
            student_logits = torch.nn.functional.adaptive_avg_pool1d(student_logits, 50)  # [B, C, 50]
            student_logits = student_logits.transpose(1, 2)  # [B, 50, C]
            
            distill_loss = torch.nn.functional.mse_loss(
                student_logits,  # 크기 조정된 student 출력 
                teacher_outputs["pred_logits"]  # [B, 50, C]
            )
            lambda_distill = self.cfg.get('lambda_distill', 1.0)
            
            # 3.3. Total loss
            total_loss = detr_losses + lambda_distill * distill_loss

            total_loss.backward()
            self.optimizer.step()
            progress_bar.update()
        progress_bar.close()

    def prepare_targets(self, targets, batch_size):
        targets_list = []
        mask = targets['mask_2d']
        key_list = ['labels', 'boxes', 'calibs', 'depth', 'size_3d', 'heading_bin', 'heading_res', 'boxes_3d']
        for bz in range(batch_size):
            target_dict = {}
            for key, val in targets.items():
                if key in key_list:
                    target_dict[key] = val[bz][mask[bz]]
            targets_list.append(target_dict)
        return targets_list