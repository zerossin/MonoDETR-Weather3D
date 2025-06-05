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
                    self.tester.inference(eval_mode='both')  # Clean과 Foggy 둘 다 평가
                    results = self.tester.evaluate(eval_mode='both')
                    
                    # Foggy 결과를 main metric으로 사용 (개선 목표이므로)
                    cur_result = results['foggy']
                    
                    self.logger.info("Clean Result: {}, Foggy Result: {}".format(
                        results['clean'], results['foggy']))
                    
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
        self.logger.info(">>>>>>> Epoch: {}".format(epoch))

        progress_bar = tqdm.tqdm(total=len(self.train_loader), leave=(self.epoch+1 == self.cfg['max_epoch']), desc='iters')
        for batch_idx, (inputs, calibs, targets, info) in enumerate(self.train_loader):
            
            # 1. 입력 데이터 처리 - 모드별 대응
            if isinstance(inputs, (tuple, list)) and len(inputs) == 2:
                imgs_clean, imgs_foggy = inputs
            else:
                imgs_clean = inputs
                imgs_foggy = None
            
            imgs_clean = imgs_clean.to(self.device)
            if imgs_foggy is not None:
                imgs_foggy = imgs_foggy.to(self.device)
            
            calibs = calibs.to(self.device)
            img_sizes = targets['img_size'].to(self.device)

            targets_list = self.prepare_targets(targets, imgs_clean.shape[0])
            for t in targets_list:
                for k in t.keys():
                    t[k] = t[k].to(self.device)

            dn_args = None
            if self.cfg.get("use_dn", False):
                dn_args = (targets_list, self.cfg['scalar'], self.cfg['label_noise_scale'], 
                          self.cfg['box_noise_scale'], self.cfg['num_patterns'])

            self.optimizer.zero_grad()
            
            # 2. 모드별 설정 확인
            use_teacher = (self.teacher_model is not None and 
                          self.cfg.get('lambda_distill', 0.0) > 0)
            use_foggy = imgs_foggy is not None  # foggy 이미지가 있으면 사용
            
            lambda_distill = self.cfg.get('lambda_distill', 0.0)
            
            # 자동 가중치 조정 - 원래 균형 유지
            if use_teacher and use_foggy:
                # Teacher-Student + Multi-domain: 균형 유지
                alpha = self.cfg.get('clean_weight', 0.5)
                beta = self.cfg.get('foggy_weight', 0.5)
            elif use_foggy:
                # Multi-domain only: 균형
                alpha = self.cfg.get('clean_weight', 0.5)
                beta = self.cfg.get('foggy_weight', 0.5)
            else:
                # Clean only
                alpha = self.cfg.get('clean_weight', 1.0)
                beta = self.cfg.get('foggy_weight', 0.0)
            
            # 3. Teacher 모델 실행 (조건부)
            teacher_outputs = None
            if use_teacher:
                with torch.no_grad():
                    teacher_outputs = self.teacher_model(imgs_clean, calibs, targets_list, img_sizes, dn_args=dn_args)
            
            # 4. Student 모델 실행
            student_outputs_clean = self.model(imgs_clean, calibs, targets_list, img_sizes, dn_args=dn_args)
            
            student_outputs_foggy = None
            if use_foggy:
                student_outputs_foggy = self.model(imgs_foggy, calibs, targets_list, img_sizes, dn_args=dn_args)
            
            # 5. Loss 계산
            # 5.1. Clean domain detection loss
            clean_losses_dict = self.detr_loss(student_outputs_clean, targets_list)
            weight_dict = self.detr_loss.weight_dict
            clean_losses = sum([clean_losses_dict[k] * weight_dict[k] for k in clean_losses_dict if k in weight_dict])
            
            # 5.2. Foggy domain detection loss (조건부)
            foggy_losses = 0
            if use_foggy and student_outputs_foggy is not None:
                foggy_losses_dict = self.detr_loss(student_outputs_foggy, targets_list)
                foggy_losses = sum([foggy_losses_dict[k] * weight_dict[k] for k in foggy_losses_dict if k in weight_dict])
            
            # 5.3. Distillation loss (조건부)
            distill_loss = 0
            if use_teacher and teacher_outputs is not None:
                student_outputs = student_outputs_foggy if use_foggy else student_outputs_clean
                
                # Auxiliary layers distillation
                if 'aux_outputs' in teacher_outputs and 'aux_outputs' in student_outputs:
                    teacher_aux = teacher_outputs['aux_outputs']
                    student_aux = student_outputs['aux_outputs']
                    min_layers = min(len(teacher_aux), len(student_aux))
                    
                    for i in range(min_layers):
                        if 'pred_logits' in teacher_aux[i] and 'pred_logits' in student_aux[i]:
                            t_logits = teacher_aux[i]['pred_logits']
                            s_logits = student_aux[i]['pred_logits']
                            
                            if t_logits.shape[1] != s_logits.shape[1]:
                                min_queries = min(t_logits.shape[1], s_logits.shape[1])
                                t_logits = t_logits[:, :min_queries, :]
                                s_logits = s_logits[:, :min_queries, :]
                            
                            distill_loss += torch.nn.functional.mse_loss(s_logits, t_logits)
                        
                        if 'pred_boxes' in teacher_aux[i] and 'pred_boxes' in student_aux[i]:
                            t_boxes = teacher_aux[i]['pred_boxes']
                            s_boxes = student_aux[i]['pred_boxes']
                            
                            if t_boxes.shape[1] != s_boxes.shape[1]:
                                min_queries = min(t_boxes.shape[1], s_boxes.shape[1])
                                t_boxes = t_boxes[:, :min_queries, :]
                                s_boxes = s_boxes[:, :min_queries, :]
                            
                            distill_loss += torch.nn.functional.mse_loss(s_boxes, t_boxes)
                
                # Final layer distillation
                student_logits = student_outputs["pred_logits"]
                if isinstance(student_logits, list):
                    student_logits = student_logits[-1]
                    
                teacher_logits = teacher_outputs["pred_logits"]
                
                if teacher_logits.shape[1] != student_logits.shape[1]:
                    min_queries = min(teacher_logits.shape[1], student_logits.shape[1])
                    teacher_logits = teacher_logits[:, :min_queries, :]
                    student_logits = student_logits[:, :min_queries, :]
                
                distill_loss += torch.nn.functional.mse_loss(student_logits, teacher_logits)
                
                # Box distillation
                if 'pred_boxes' in teacher_outputs and 'pred_boxes' in student_outputs:
                    teacher_boxes = teacher_outputs['pred_boxes']
                    student_boxes = student_outputs['pred_boxes']
                    
                    if isinstance(student_boxes, list):
                        student_boxes = student_boxes[-1]
                    
                    if teacher_boxes.shape[1] != student_boxes.shape[1]:
                        min_queries = min(teacher_boxes.shape[1], student_boxes.shape[1])
                        teacher_boxes = teacher_boxes[:, :min_queries, :]
                        student_boxes = student_boxes[:, :min_queries, :]
                    
                    distill_loss += torch.nn.functional.mse_loss(student_boxes, teacher_boxes)

            # 6. Total loss
            total_loss = (alpha * clean_losses + 
                        beta * foggy_losses + 
                        lambda_distill * distill_loss)

            total_loss.backward()
            self.optimizer.step()

            # 7. 로깅
            if batch_idx == 0:
                mode_str = []
                if use_teacher: mode_str.append("Teacher-Student")
                if use_foggy: mode_str.append("Multi-domain")
                if not use_teacher and not use_foggy: mode_str.append("Clean-only")
                
                self.logger.info(f"🔍 Mode: {' + '.join(mode_str) if mode_str else 'Clean-only'}")
                self.logger.info(f"🔍 Use teacher: {use_teacher}")
                self.logger.info(f"🔍 Use foggy: {use_foggy}")
                self.logger.info(f"🔍 Clean weight: {alpha}")
                self.logger.info(f"🔍 Foggy weight: {beta}")
                self.logger.info(f"🔍 Lambda distill: {lambda_distill}")
                
                self.logger.info(f"🔍 Clean detection loss: {clean_losses.item():.4f}")
                if use_foggy:
                    self.logger.info(f"🔍 Foggy detection loss: {foggy_losses.item():.4f}")
                if use_teacher:
                    self.logger.info(f"🔍 Distillation loss: {distill_loss.item():.4f}")
                self.logger.info(f"🔍 Total loss: {total_loss.item():.4f}")
            
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