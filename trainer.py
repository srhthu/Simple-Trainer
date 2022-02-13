"""
All you need to train a neural network.

Adapted from transformers.Trainer

Function:
    Train
    Evaluate
    Log
    Save
"""

from distutils.sysconfig import PREFIX
from optparse import Option
import os
from re import L
import sys
import json
import logging
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
import math
import numpy as np
from tqdm import tqdm

from transformers import DataCollator, TrainingArguments
from transformers.data.data_collator import default_data_collator

def get_logger():
    logger = logging.getLogger(__name__)
    if len(logger.handlers) == 0:
        #_fmt = logging.Formatter(fmt = "%(name)s %(asctime)s %(message)s", datefmt = "%Y-%m-%d %H:%M:%S")
        _hdlr = logging.StreamHandler()
        _hdlr.setFormatter(logging.Formatter(fmt = ""))
        logger.addHandler(_hdlr)
    logger.setLevel(logging.DEBUG)
    return logger

logger = get_logger()

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from transformers.trainer import Trainer
from transformers.trainer_utils import (
    set_seed
)
from transformers.optimization import AdamW, get_scheduler

from .utils import (
    cal_continuity_loss, cal_sparsity_loss, to_cuda, cuda,
    nest_batch
)

PREFIX_CHECKPOINT_DIR = 'checkpoint'

@dataclass
class SimpleTrainingArguments:
    """
    Training Arguments.

    Parameters:
        output_dir (str, optional):
            If None, do not save and log
        do_eval (bool):
            Whether do evaluation during training
        evaluation_strategy (str):
            possible values:
                "no": No evaluation
                "steps": eval is done every `eval_steps`
                "epoch": eval is done at every epoch.
    """
    output_dir: Optional[str] = None
    do_eval: bool = False

    evaluation_strategy: str = "no"
    eval_steps: Optional[int] = None

    logging_steps: Optional[int] = 100

    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 16

    learning_rate: float = 5e-5
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0

    num_train_epochs: float = 3.0
    max_steps: int = -1

    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8

    lr_scheduler_type: str = "linear"
    warmup_steps: int = 0
    warmup_ratio: float = 0.0    

    no_cuda: bool = False
    dataloader_num_workers: int = 0
    metric_for_best_model: Optional[str] = None
    greater_is_better: bool = True

    seed: int = 26

    early_stopping_patience = 3

    _n_gpu: int = field(init=False, repr = False, default = -1)

    def _setup_devices(self):
        if self.no_cuda:
            device = torch.device('cpu')
            self._n_gpu = 0
        else:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self._n_gpu = torch.cuda.device_count()

    def get_warmup_steps(self, num_training_steps: int):
        """
        Get number of steps used for a linear warmup.
        """
        warmup_steps = (
            self.warmup_steps if self.warmup_steps > 0 else math.ceil(num_training_steps * self.warmup_ratio)
        )
        return warmup_steps

    @property
    def n_gpu(self):
        _ = self._setup_devices
        return self._n_gpu

    @property
    def train_batch_size(self):
        dev_n = self.n_gpu if self.n_gpu > 0 else 1
        return dev_n * self.per_device_train_batch_size
    
    @ property
    def eval_batch_size(self):
        dev_n = self.n_gpu if self.n_gpu > 0 else 1
        return dev_n * self.per_device_eval_batch_size
    
@dataclass
class TrainerState:
    """
    Trainer state variables
    """
    epoch: Optional[float] = None
    global_step: int = 0
    max_steps: int = 0
    best_metric = None
    best_step = 0
    best_model = None
    early_stopping_patience_conter = 0

@dataclass
class TrainerControl:
    should_save: bool = False
    should_log: bool = False
    should_evaluate: bool = False
    should_training_stop: bool = False

class MetricHolder:
    """
    Record metrics, e.g., loss.

    Attributes:
        metric_interval: metrics in one interval
            Dict[str, List[float]]    
    """
    def __init__(self):
        self._reset()
    
    def _reset(self):
        self.metric_interval = {}

    def record(self, metrics):
        for met_name, met_value in metrics.items():
            if met_name not in self.metric_interval:
                self.metric_interval[met_name] = []
            self.metric_interval[met_name].append(met_value)

    def retrieve(self):
        if len(self.metric_interval) == 0:
            return {}
        interval_ave = {
            name: np.mean(values) for name, values in self.metric_interval.items()
        }

        self._reset()
        return interval_ave


class BasicTrainer:
    """
    Simple and customized trainer.

    You can customize:
        log information
        feed data
        loss function
        parameter update
    
    Args:

    """
    def __init__(
        self,
        model,
        args: SimpleTrainingArguments = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        compute_metrics: Optional[Callable] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None)
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.compute_metrics = compute_metrics

        if args is None:
            logger.info('Use default training arguments')
            args = SimpleTrainingArguments()
        self.args = args

        set_seed(self.args.seed)

        self.device = self.args._setup_devices()

        self.data_collator = data_collator if data_collator else default_data_collator

        # will change during training.
        self.model_wrapped = model

        self.optimizer, self.lr_scheduler = optimizers

        self.state = TrainerState()
        self.control = TrainerControl()
        self.loss_recoder = MetricHolder() # you can record more than loss

        # output to file and logger
        if args.output_dir is not None:
            if not os.path.exists(args.output_dir):
                #print('create output dir')
                os.makedirs(args.output_dir)
            # remove duplicate file handlers
            logger.handlers = logger.handlers[:1]

            log_file = os.path.join(args.output_dir, 'log.txt')
            file_hdlr = logging.FileHandler(log_file, mode = 'a')
            fmt = logging.Formatter(fmt = "%(name)s %(asctime)s %(message)s", datefmt = "%Y-%m-%d %H:%M:%S")
            file_hdlr.setFormatter(fmt)
            logger.addHandler(file_hdlr)
    
    def get_dataloader(self, dataset, batch_size, shuffle = False):
        dl = DataLoader(
            dataset,
            batch_size = batch_size,
            collate_fn = self.data_collator,
            shuffle = shuffle)
        
        return dl

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """
        If exists, do nothing. Else, a reasonable default.
        """
        self.create_optimizer()
        self.create_scheduler(num_training_steps = num_training_steps, optimizer = self.optimizer)
    
    def create_optimizer(self):
        """
        Setup the optimizer.

        Some parameters, e.g., nn.LayerNorm do not need decay. Treat them separately.
        """
        if self.optimizer is None:
            # we use AdamW here
            opt_kwargs = {
                'lr': self.args.learning_rate,
                'betas': (self.args.adam_beta1, self.args.adam_beta2),
                'eps': self.args.adam_epsilon
            }
            self.optimizer = AdamW(self.model.parameters(), **opt_kwargs)
        
        return self.optimizer

    def create_scheduler(self, num_training_steps: int, optimizer = None):
        """
        Setup the scheduler.
        """
        if self.lr_scheduler is None:
            self.lr_scheduler = get_scheduler(
                self.args.lr_scheduler_type,
                optimizer=self.optimizer if optimizer is None else optimizer,
                num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                num_training_steps=num_training_steps,
            )
        return self.lr_scheduler

    def _wrap_model(self, model, training = True):
        if self.args.n_gpu > 0:
            model.cuda()
        # multi-gpu training
        if self.args.n_gpu > 1:
            model = nn.DataParallel(model)

        if training:
            model.train()
        else:
            model.eval()
        
        return model

    def train(self, train_dataset = None, eval_dataset = None):
        train_dataset = train_dataset if train_dataset else self.train_dataset
        eval_dataset = eval_dataset if eval_dataset else self.eval_dataset
        self.eval_dataset = eval_dataset # will be used in evaluate()

        train_dataloader = self.get_dataloader(
            train_dataset, self.args.train_batch_size, True)

        model = self._wrap_model(self.model)  # ! the original arg is `self.model_wrapped`. I think the current one is proper.
        if model is not self.model:
            self.model_wrapped = model

        # setup max_steps and num_train_epochs
        steps_per_epoch = len(train_dataloader)
        if self.args.max_steps > 0:
            max_steps = self.args.max_steps
            num_train_epochs = max_steps // steps_per_epoch + int(
                max_steps % steps_per_epoch > 0
            )
        else:
            max_steps = math.ceil(self.args.num_train_epochs * steps_per_epoch)
            num_train_epochs = math.ceil(self.args.num_train_epochs)
        
        self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        logger.info("***** Running training *****")
        logger.info(f"  Num Epochs = {num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {self.args.train_batch_size}")
        logger.info(f"  Total optimization steps = {max_steps}")
        
        self.state = TrainerState(
            epoch=0,
            max_steps = max_steps
        )
        self.control = TrainerControl()
        self.loss_recoder = MetricHolder()

        model.zero_grad()
        self.on_train_begin()
        progress_bar = tqdm(total=max_steps)
        for epoch in range(num_train_epochs):

            steps_in_epoch = len(train_dataloader)
            self.on_epoch_begin()

            for step, inputs in enumerate(train_dataloader):
                model.train()
                inputs = self._prepare_inputs(inputs)
                
                # Training step
                # Do: feed data, calculate loss, loss backward
                # Return: a dict of measurements (float) that need to be loged
                losses = self.training_step(model, inputs)
                
                self.loss_recoder.record(losses)

                self.state.global_step += 1
                self.state.epoch = epoch + (step + 1) / steps_in_epoch

                self.on_step_end(self.args, self.state, self.control)
                progress_bar.update(1)

                self._maybe_log_save_evaluate()

                if self.control.should_training_stop:
                    break
                # train step end
            
            self.on_epoch_end(self.args, self.state, self.control)
            self._maybe_log_save_evaluate()

            if self.control.should_training_stop:
                break
        
            # Train loop end
        
        self.on_train_end()
        # save best ckpt
        # create link to the checkpoint
        if self.args.output_dir is not None:
            # ver_1, save model
            """
            best_path = os.path.join(self.args.output_dir, "best_model.bin")
            logger.info(f'Save best model to {best_path}')
            torch.save(
                self.state.best_model,
                best_path)
            """
            # ver_2, save step
            pass

    def training_step(self, model, inputs):
        """
        Perform a training step on a batch of inputs, including feedforward, compute loss, 
        loss backward, optimizer and scheduler update.

        Args:
            model (nn.Module):
                The model to train
            inputs Dict[str, Union[torch.Tensor, Any]]:
                inputs and targets of the model.
        Return:
            `Dict[str, float]`: metrics that need to be logged, e.g. loss.
        """
        

        outputs = model(**inputs)
        loss = outputs['loss'] if isinstance(outputs, dict) else outputs[0]
        
        if self.args.n_gpu > 1:
            loss = loss.mean() # mean() to average on multi-gpu parallel training.

        model.zero_grad()
        loss.backward()

        # Gradient clipping
        if self.args.max_grad_norm is not None and self.args.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)
        
        self.optimizer.step()
        self.lr_scheduler.step()

        return {'loss': loss.detach().cpu().item()}

    def evaluate(self, model = None, eval_dataset = None):
        """
        Predict and calculate eval metrics. Print metrics and update trainer state.
        """
        model = self.model if model is None else model
        eval_dataset = self.eval_dataset if eval_dataset is None else eval_dataset

        eval_dataloader = self.get_dataloader(eval_dataset, self.args.eval_batch_size)

        all_logits, all_labels = self.evaluation_loop(eval_dataloader, model)

        metrics = self.compute_metrics(all_logits, all_labels) if self.compute_metrics else {}

        self.log(metrics)

        self.on_evaluate(self.args, self.state, self.control, metrics)

        return metrics

    
    def evaluation_loop(self, dataloader, model):
        """
        Get all predictions.
        """
        model = self._wrap_model(self.model, training = False)
        logger.info(f"***** Running evaluation *****")
        logger.info(f"  Num examples = {len(dataloader)}")
        logger.info(f"  Batch size = {dataloader.batch_size}")

        # Initialize containers
        all_logits = None
        all_labels = None
        with torch.no_grad():
            for step, inputs in enumerate(dataloader):
                inputs = self._prepare_inputs(inputs)
                logits, labels = self.prediction_step(model, inputs)
                logits = logits.detach().cpu().numpy()
                labels = labels.detach().cpu().numpy()

                all_logits = logits if all_logits is None else nest_batch(all_logits, logits)
                all_labels = labels if all_labels is None else nest_batch(all_labels, labels)
        
        return all_logits, all_labels

            
    def prediction_step(self, model, inputs):
        """
        Predict step. Return logits and labels.
        """

        labels = inputs.pop('labels')
        outputs = model(**inputs)

        logits = outputs['logits'] if isinstance(outputs, dict) else outputs[1]

        return logits, labels

    def _prepare_inputs(self, inputs: dict):
        # to cuda
        if self.args.n_gpu > 0:
            return cuda(inputs)
        else:
            return inputs

    def _maybe_log_save_evaluate(self):
        if self.control.should_log:
            logs = self.loss_recoder.retrieve()
            logs["learning_rate"] = self._get_learning_rate()
            self.log(logs)
        
        if self.control.should_evaluate:
            metrics = self.evaluate()
        else:
            metrics = None
        
        if self.control.should_save:
            self._save_checkpoint(self.model, metrics = metrics)

    def _get_learning_rate(self):
        last_lr = self.lr_scheduler.get_last_lr()[0]
        return last_lr
    
    def _save_checkpoint(self, model, metrics = None):
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
        run_dir = self.args.output_dir

        if run_dir is None:
            return None
        output_dir = os.path.join(run_dir, checkpoint_folder)
        os.makedirs(output_dir, exist_ok= True)
        logger.info(f"Saving model checkpoint to {output_dir}")

        # save model
        torch.save(self.model.state_dict, os.path.join(output_dir, "pytorch_model.bin"))

        # save args
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

        # save metrics
        if metrics is not None:
            with open(os.path.join(output_dir, "eval_metrics.json"), 'w') as f:
                json.dump(metrics, f, indent = 4, ensure_ascii=False)

        self.control.should_save = False

    def log(self, logs: Dict[str, float]) -> None:
        step_info = {"step": self.state.global_step}
        if self.state.epoch is not None:
            step_info["epoch"] = round(self.state.epoch, 2)

        output = {**logs, **step_info}
        
        self.control.should_log = False
        logger.info(repr(output))
        
    
    def on_train_begin(self):
        """
        Subclass and override for customized service
        """
        pass
        
    def on_epoch_begin(self):
        """
        Subclass and override for customized service
        """
        pass
    
    def on_step_end(self, args: SimpleTrainingArguments, state: TrainerState, 
            control: TrainerControl, **kwargs):
        """
        Subclass and override for customized service
        """
        # Log
        if (args.logging_steps
            and args.logging_steps > 0
            and state.global_step % args.logging_steps == 0):
            control.should_log = True
        
        # Evaluate
        if args.eval_steps and (args.eval_steps > 0 and state.global_step % args.eval_steps == 0):
            if self.eval_dataset is not None:
                control.should_evaluate = True
            control.should_save = True
        
        # End training
        if state.global_step >= state.max_steps:
            control.should_training_stop = True

        return control        

    def on_epoch_end(self, args: SimpleTrainingArguments, state: TrainerState, 
            control: TrainerControl, **kwargs):
        """
        Subclass and override for customized service
        """
        # Evaluate
        if (
            args.evaluation_strategy == 'epoch'
        ):
            if self.eval_dataset is not None:
                control.should_evaluate = True
            control.should_save = True # save at each epoch.

        return control
    
    def on_train_end(self):
        """
        Subclass and override for customized service
        """
        pass

    def on_evaluate(self, args: SimpleTrainingArguments, state: TrainerState, 
            control: TrainerControl, metrics):
        """
        Perform early stop
        """
        
        control.should_evaluate = False
        # Early stop
        metric_to_check = args.metric_for_best_model
        if not metric_to_check:
            if len(metrics) == 1:
                metric_to_check = list(metrics.keys())[0]
            else:
                raise ValueError(f"did not find {metric_to_check}")
        if not metric_to_check.startswith("eval_"):
            metric_to_check = f"eval_{metric_to_check}"
        metric_value = metrics.get(metric_to_check)

        if metric_value is None:
            logger.warning(
                f"early stopping required metric_for_best_model, but did not find {metric_to_check} so early stopping is disabled"
            )
            return
        
        operator = np.greater if args.greater_is_better else np.less
        if state.best_metric is None or operator(metric_value, state.best_metric):
            state.best_metric = metric_value
            state.best_step = state.global_step
            #state.best_model = self.model.state_dict()
            state.early_stopping_patience_conter = 0
            
            logger.info(f'Best step {self.state.best_step}')
            if self.args.output_dir is not None:
                with open(os.path.join(self.args.output_dir, 'best_step.txt'), 'w') as f:
                    f.write(f'{self.state.best_step}')
        else:
            state.early_stopping_patience_conter += 1
        
        if state.early_stopping_patience_conter >= args.early_stopping_patience:
            logger.info(f'No improvement after {args.early_stopping_patience} eval steps. Early stop.')
            control.should_training_stop = True
        
"""
Old version
"""

class FreezeGrad:
    """
    Set requires_grad to False when entering and restore after exit
    """
    def __init__(self, params):
        self.params = params
    
    def __enter__(self):
        for param in self.params:
            param.requires_grad = False
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        for param in self.params:
            param.requires_grad = True


def train_loop_rnp(train_data, model, opts, args):
    """
    Train RNP for one epoch
    """
    # optimizer
    gen_opt, pred_opt = opts
    step = 0
    # log information
    num_sample = 0
    pred_acc = 0.
    pred_losses = 0.

    model.train()

    for batch in train_data:
        batch = to_cuda(batch)

        if step % 4 == 0:
            froze_params = model.predictor_trainable_variables()
        else:
            froze_params = model.generator_trainable_variables()
        
        with FreezeGrad(froze_params):
            rationale, pred_logits = model(
                batch['text_ids'],
                batch['mask'],
                batch['env']
            )
            pred_loss = F.cross_entropy(
                input = F.softmax(pred_logits, dim = -1), target = batch['label'],
                reduction = 'mean'
            )
            sparsity_loss = args.sparsity_lambda * cal_sparsity_loss(
                    rationale[:, :, 1], batch['mask'], args.sparsity_percentage)
            continuity_loss = args.continuity_lambda * cal_continuity_loss(
                    rationale[:, :, 1])
            gen_loss = pred_loss + sparsity_loss + continuity_loss
        
        if step % 4 == 0:
            # update generator
            gen_opt.zero_grad()
            gen_loss.backward()
            gen_opt.step()
            
        else:
            # update env inv predictor
            pred_opt.zero_grad()
            pred_loss.backward()
            pred_opt.step()
        
        step += 1

        pred_acc += (pred_logits.argmax(dim = -1) == batch['label']).sum().item()
        pred_losses += pred_loss.detach().item()
        num_sample += batch['label'].shape[0]
    
    return pred_acc / num_sample, pred_losses / num_sample

def evaluate_RNP(eval_data, model):
    """
    Evaluate RNP model.
    
    Return
        eval_acc
    """

    # log information
    num_sample = 0
    pred_acc = 0.

    model.eval()
    for batch in eval_data:
        batch = to_cuda(batch)
        with torch.no_grad():
            rationale, pred_logits = model(
                batch['text_ids'],
                batch['mask'],
                batch['env']
            )
        
        pred_acc += (pred_logits.argmax(dim = -1) == batch['label']).sum().item()
        num_sample += batch['label'].shape[0]
    
    model.train()
    return pred_acc / num_sample

def train_RNP(num_epoch, train_data, dev_data, model, opts, 
        save_path, args, test_data = None):
    """
    Train RNP model with early stop
    """
    best_metric = 0
    best_epoch = -1
    best_model_state = None
    for epoch in range(1, num_epoch + 1):
        train_results = train_loop_rnp(train_data, model, opts, args)
        dev_acc = evaluate_RNP(dev_data, model)
        
        if test_data is not None:
            test_results = evaluate_RNP(test_data, model)
        else:
            test_results = None

        print(
            f'Epoch {epoch}\n'
            f'Train  acc:    {train_results[0]:.4f}, loss: {train_results[1]:.4f}'
            )
        print(
            f'Dev    acc:   {dev_acc:.4f} '
        )
        if test_results is not None:
            print(
                f'Test  acc:  {test_results:.4f}'
            )
        if dev_acc > best_metric:
            best_metric = dev_acc
            best_epoch = epoch
            best_model_state = model.state_dict()
        elif epoch - best_epoch >=5:
            # early stop
            print('No improvement after 3 epoch\nEarly stop.')
            print(f'Best epoch: {best_epoch}, inv acc: {best_metric:.4f}')
            break
    if save_path is not None:
        torch.save(best_model_state, os.path.join(save_path, 'best_model.pth'))
    
