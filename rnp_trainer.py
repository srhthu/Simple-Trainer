import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.optimization import get_scheduler

from .trainer import (
    SimpleTrainingArguments,
    BasicTrainer,
    FreezeGrad
)
from .utils import (
    cal_continuity_loss,
    cal_sparsity_loss
)

class RNP_Trainer(BasicTrainer):
    """
    RNP model trainer.

    Args:
        optimizer - (gen_opt, pred_opt)
        scheduler - (gen_sche, pred_sche)
    """

    def _get_learning_rate(self):
        last_lr = [sche.get_last_lr()[0] for sche in self.lr_scheduler]
        return last_lr

    def create_scheduler(self, num_training_steps: int, optimizer=None):
        gen_opt, pred_opt = optimizer
        if self.lr_scheduler is None:
            gen_sche = get_scheduler(
                self.args.lr_scheduler_type,
                optimizer = gen_opt,
                num_warmup_steps = self.args.get_warmup_steps(num_training_steps),
                num_training_steps = num_training_steps,
            )
            pred_sche = get_scheduler(
                self.args.lr_scheduler_type,
                optimizer = pred_opt,
                num_warmup_steps = self.args.get_warmup_steps(num_training_steps),
                num_training_steps = num_training_steps,
            )
            self.lr_scheduler = [gen_sche, pred_sche]
        return self.lr_scheduler
    def training_step(self, model, inputs):
        args = self.args

        if self.state.global_step % 4 == 0:
            freeze_params = self.model.predictor_trainable_variables()
        else:
            freeze_params = self.model.generator_trainable_variables()
        
        labels = inputs.pop('labels')

        with FreezeGrad(freeze_params):
            rationale, pred_logits = model(**inputs)

            pred_loss = F.cross_entropy(pred_logits, labels)
            
            sparsity_loss = args.sparsity_lambda * cal_sparsity_loss(
                    rationale[:, :, 1], inputs['attention_mask'], args.sparsity_percentage)
            continuity_loss = args.continuity_lambda * cal_continuity_loss(
                    rationale[:, :, 1])
            gen_loss = pred_loss + sparsity_loss + continuity_loss

        gen_opt, pred_opt = self.optimizer
        gen_sche, pred_sche = self.lr_scheduler
        
        if self.state.global_step % 4 == 0:
            gen_opt.zero_grad()
            gen_loss.backward()
            gen_opt.step()
            gen_sche.step()
        else:
            pred_opt.zero_grad()
            pred_loss.backward()
            pred_opt.step()
            pred_sche.step()
        
        pred_acc = (pred_logits.argmax(dim = -1) == labels).float().mean().item()
        # torch mean() not accept bool type. but sum() can. should change to float first.

        return {'pred_loss': pred_loss.detach().cpu().item(), 'pred_acc': pred_acc}

    def prediction_step(self, model, inputs):
        labels = inputs.pop('labels')

        _, pred_logits = model(**inputs)

        return pred_logits, labels