#!/usr/bin/env python

import os
from copy import deepcopy
from datetime import timedelta

from sympy.ntheory.factor_ import smoothness

from models import TangentSpaceModel
from models.CompressionAEModel import load_encoding_model
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.tensorboard import SummaryWriter
import ShapeNetDataLoader
from Dataset import generate_datasets, load_dataset
import json
from types import SimpleNamespace
import torch.nn.init as init


def weights_init_orthogonal(module):
    # classname = m.__class__.__name__
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
        init.orthogonal_(module.weight.data, gain=1)
    elif isinstance(module, nn.Conv3d) or isinstance(module, nn.ConvTranspose3d):
        init.orthogonal_(module.weight.data, gain=1)
    elif isinstance(module, nn.Linear):
        init.orthogonal_(module.weight.data, gain=1)
    elif isinstance(module, nn.BatchNorm2d):
        init.normal_(module.weight.data, 1.0, 0.02)
        init.constant_(module.bias.data, 0.0)

def setup_ddp(parallel):
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    init_process_group(backend='nccl', timeout=timedelta(seconds=10*3600))

    return local_rank


def prepare_training_objects(datasets_dict,
                             n_output_vectors,
                             enable_bn,
                             train_batch_size,
                             val_batch_size,
                             n_cpus,
                             n_epochs,
                             lr,
                             momentum,
                             weight_decay,
                             parallel=1):

    encoding_model = load_encoding_model()
    model = TangentSpaceModel.Model(n_output_vectors, enable_bn)
    optimizer = torch.optim.Adam(params=model.parameters(), #filter(lambda p: p.requires_grad, model.parameters())
                                 lr=lr,
                                 weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

    train_loader = ShapeNetDataLoader.get_train_loader(train_dataset=datasets_dict['train'],
                                                       batch_size=train_batch_size,
                                                       n_cpus=n_cpus,
                                                       parallel=parallel)

    val_loader = ShapeNetDataLoader.get_val_loader(val_dataset=datasets_dict['val'],
                                                   parallel=parallel,
                                                   batch_size=val_batch_size,
                                                   n_cpus=n_cpus)

    return encoding_model, model, optimizer, scheduler, train_loader, val_loader


class Trainer:
    def __init__(self,
                 model: torch.nn.Module,
                 encoding_model: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 scheduler: torch.optim.lr_scheduler,
                 train_dataloader: torch.utils.data.DataLoader,
                 val_dataloader: torch.utils.data.DataLoader,
                 parallel: bool,
                 save_every: int,
                 print_every: int,
                 snapshot_dir: str,
                 snapshot_path: str = None, ):
        if parallel:
            self.gpu_id = int(os.environ['LOCAL_RANK'])
            self.device = torch.device(f'cuda:{self.gpu_id}')
            self.encoding_model = encoding_model.to(self.device)
            self.model = model.to(self.gpu_id)
            self.model = DDP(self.model, device_ids=[self.gpu_id], output_device=self.gpu_id)
        else:
            self.device = torch.device('cpu')
            self.gpu_id = 0
            self.model = model
            self.encoding_model = encoding_model

        self.optimizer = optimizer
        self.lr = self.optimizer
        self.scheduler = scheduler
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.unit_criterion = nn.MSELoss()
        self.orth_criterion = nn.MSELoss()
        self.save_every = save_every
        self.print_every = print_every
        self.epochs_run = 0
        self.snapshot_dir = snapshot_dir
        self.writer = SummaryWriter()
        if not os.path.exists(self.snapshot_dir):
            os.makedirs(self.snapshot_dir)

        if os.path.exists(snapshot_path) and snapshot_path.endswith('.pth'):
            print("Loading snapshot")
            self._load_snapshot(snapshot_path)
        else:
            print("Initializing weights")
            self.model.apply(weights_init_orthogonal)

    def _criterion(self, z_dot, basis_vectors1, basis_vectors2=None): # derivatives_shape (batch, 128*9, 6) # z_dot_shape (batch, 128, 3, 3)
        vectorized_z_dot = z_dot.view(z_dot.shape[0], -1)
        vectorized_basis_vectors1 = basis_vectors1.view(basis_vectors1.shape[0], -1, basis_vectors1.shape[-1])
        basis_vectors_norms1 = torch.norm(vectorized_basis_vectors1, dim=1, keepdim=True)
        normalized_basis_vectors1 = vectorized_basis_vectors1 / basis_vectors_norms1
        z_dot_norm = torch.norm(vectorized_z_dot, dim=1, keepdim=True)
        z_unit = vectorized_z_dot / z_dot_norm
        # z_unit[z_dot_norm.squeeze() == 0, :] = 0

        linear_fit1 = torch.linalg.lstsq(normalized_basis_vectors1, z_unit.unsqueeze(2)) # A.X = B
        residuals1 = torch.bmm(normalized_basis_vectors1, linear_fit1.solution) - z_unit.unsqueeze(2)
        sse1 = torch.sum(residuals1 ** 2, dim=1)
        span_loss1 = torch.mean(sse1)
        span_loss = span_loss1
        pairwise_multiplication = torch.bmm(normalized_basis_vectors1.permute([0, 2, 1]),
                                            normalized_basis_vectors1)
        batch_identity = torch.eye(pairwise_multiplication.shape[1],
                                   pairwise_multiplication.shape[2], device=pairwise_multiplication.device).unsqueeze(0)
        orthogonality_loss = torch.mean((batch_identity.expand(pairwise_multiplication.shape[0], -1, -1) - pairwise_multiplication)**2)
        smoothness_loss = torch.zeros_like(span_loss)

        if basis_vectors2 is not None:
            vectorized_basis_vectors2 = basis_vectors2.view(basis_vectors2.shape[0], -1, basis_vectors2.shape[-1])
            basis_vectors_norms2 = torch.norm(vectorized_basis_vectors2, dim=1, keepdim=True)
            normalized_basis_vectors2 = basis_vectors2 / basis_vectors_norms2
            basis_vectors_diff = (normalized_basis_vectors1 - normalized_basis_vectors2) # alt: measure the difference between the spaces that these vectors span
            smoothness_loss = torch.mean(torch.sum(basis_vectors_diff ** 2, dim=2))
            linear_fit2 = torch.linalg.lstsq(vectorized_basis_vectors2, -1 * z_unit.unsqueeze(2))
            residuals2 = torch.bmm(basis_vectors2, linear_fit2.solution) - (-1 * z_unit.unsqueeze(2))
            sse2 = torch.sum(residuals2 ** 2, dim=1)
            span_loss2 = torch.mean(sse2)
            span_loss = (span_loss1 + span_loss2) / 2

        # orthogonality_loss = torch.zeros_like(span_loss)
        norm_loss = torch.zeros_like(span_loss)

        return norm_loss, orthogonality_loss, span_loss, smoothness_loss

    def _load_snapshot(self, snapshot_path):
        snapshot = torch.load(snapshot_path, map_location=self.device, weights_only=True)
        model_dict = self.model.state_dict()
        new_state_dict = deepcopy(snapshot['state_dict'])
        for key in model_dict.keys():
            if f'module.{key}' in snapshot['state_dict'].keys():
                new_state_dict[key] = snapshot['state_dict'][f'module.{key}']
                del new_state_dict[f'module.{key}']

        print(self.model.load_state_dict(new_state_dict))
        if 'epochs_run' in snapshot.keys():
            self.epochs_run = snapshot["epochs_run"]

        print(f"Resuming training from snapshot at Epoch {self.epochs_run + 1}")

    def _save_snapshot(self, epoch):
        snapshot = {
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epochs_run": self.epochs_run
        }
        torch.save(snapshot, os.path.join(self.snapshot_dir, f"snapshot_{epoch}.pth"))

    def _run_epoch(self, epoch):
        self.model.train()
        cum_norm_loss = 0.0
        cum_orth_loss = 0.0
        cum_span_loss = 0.0
        cum_smooth_loss = 0.0
        torch.autograd.set_detect_anomaly(True)
        print(f'run {epoch} in progress')
        for i_batch, (_, viewpoint1, _, viewpoint2) in enumerate(self.train_dataloader):
            if params.PARALLEL:
                viewpoint1 = viewpoint1.to(self.gpu_id)
                viewpoint2 = viewpoint2.to(self.gpu_id)
            with torch.no_grad():
                z1 = self.encoding_model(viewpoint1)
                z2 = self.encoding_model(viewpoint2)

            basis_vectors1 = self.model(z1)
            # basis_vectors2 = self.model(z2)

            norm_loss, orthogonality_loss, span_loss, smoothness_loss = self._criterion(z2 - z1, basis_vectors1, basis_vectors2=None)
            # loss = .95 * span_loss + .05 * smoothness_loss
            loss = span_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            cum_norm_loss += norm_loss.item()
            cum_orth_loss += orthogonality_loss.item()
            cum_span_loss += span_loss.item()
            cum_smooth_loss += smoothness_loss.item()

            if i_batch % self.print_every == 0 and self.gpu_id == 0:
                self.writer.add_scalar("Loss/train-norm", cum_norm_loss / (i_batch + 1),
                                       epoch * len(self.train_dataloader) + i_batch)
                self.writer.add_scalar("Loss/train-orth", cum_orth_loss / (i_batch + 1),
                                       epoch * len(self.train_dataloader) + i_batch)
                self.writer.add_scalar("Loss/train-span", cum_span_loss / (i_batch + 1),
                                       epoch * len(self.train_dataloader) + i_batch)
                print(
                    f"TRN Epoch {epoch} | Batch {i_batch} / {len(self.train_dataloader)} | "
                    f"Loss (span, ,smooth, orth, norm) "
                    f"{cum_span_loss / (i_batch + 1)} | "
                    f"{cum_smooth_loss / (i_batch + 1)} | "
                    f"{cum_orth_loss / (i_batch + 1)} | "
                    f"{cum_norm_loss / (i_batch + 1)}")

        self.scheduler.step()

        if epoch % self.save_every == 0 and self.gpu_id == 0:
            self._save_snapshot(epoch)

    def train(self, n_epochs, do_validate=False):
        for epoch in range(self.epochs_run, n_epochs):
            if params.PARALLEL:
                self.train_dataloader.sampler.set_epoch(epoch)
            self._run_epoch(epoch)
            if do_validate:
                self.validate(epoch)
            self.epochs_run += 1

        # syn for logging
        # torch.cuda.synchronize()

    def validate(self, epoch: int):
        with torch.no_grad():
            self.model.eval()
            loss_sum = 0.0
            for i_batch, (_, viewpoint1, _, viewpoint2) in enumerate(self.val_dataloader):
                if params.PARALLEL:
                    viewpoint1 = viewpoint1.to(self.gpu_id)
                    viewpoint2 = viewpoint2.to(self.gpu_id)
                with torch.no_grad():
                    z1 = self.encoding_model(viewpoint1)
                    z2 = self.encoding_model(viewpoint2)

                basis_vectors1 = self.model(z1)
                basis_vectors2 = self.model(z2)

                norm_loss, orthogonality_loss, span_loss, smoothness_loss = self._criterion(z2 - z1,
                                                                                            basis_vectors1,
                                                                                            basis_vectors2)
                loss = .95 * span_loss + .05 * smoothness_loss
                loss_sum += loss.item()

            if i_batch % self.print_every == 0 and self.gpu_id == 0:
                self.writer.add_scalar("Loss/val", loss_sum / len(self.val_dataloader), self.epochs_run)
                print(
                    f"VAL Epoch {epoch} | Batch {i_batch} / {len(self.val_dataloader)} | Loss {loss_sum / (i_batch + 1)}")


if __name__ == "__main__":
    with open('./Parameters.json', 'r') as json_file:
        params = json.load(json_file,
                           object_hook=lambda d: SimpleNamespace(**d))
    object_category = '02876657'
    if params.PARALLEL:
        rank = setup_ddp(parallel=params.PARALLEL)
    else:
        rank = 0

    if rank == 0:
        datasets_dict = generate_datasets(dataset_path=params.DATASET_PATH,
                                          object_category=object_category, # bottle
                                          rotation_sample_num=50,
                                          use_prev_indices=True,
                                          test=False)
        print('dataset ready')
        if params.PARALLEL:
            torch.distributed.barrier()

    else:
        if params.PARALLEL:
            torch.distributed.barrier()
        datasets_dict = {'train': None, 'val': None, 'test': None}
        for split_name in ['train', 'val']:
            datasets_dict[split_name] = load_dataset(split_name=split_name,
                                                     object_category=object_category,
                                                     dataset_path=params.DATASET_PATH)
    print('All nodes in sync')

    # parallel = 1
    encoding_model, model, optimizer, scheduler, train_dataloader, val_dataloader = prepare_training_objects(
        datasets_dict=datasets_dict,
        n_output_vectors=int(params.N_OUTPUT_VECTORS),
        enable_bn=True,
        train_batch_size=int(params.TRAIN_BATCH_SIZE),
        val_batch_size=int(params.VAL_BATCH_SIZE),
        n_cpus=int(params.NUM_WORKERS),
        n_epochs=int(params.N_EPOCHS),
        lr=params.LEARNING_RATE,
        momentum=params.MOMENTUM,
        weight_decay=params.WEIGHT_DECAY,
        parallel=params.PARALLEL)

    print('training objects are ready')
    trainer = Trainer(model=model,
                      encoding_model=encoding_model,
                      optimizer=optimizer,
                      scheduler=scheduler,
                      train_dataloader=train_dataloader,
                      val_dataloader=val_dataloader,
                      parallel=params.PARALLEL,
                      save_every=1,
                      print_every=100,
                      snapshot_dir=params.SNAPSHOT_DIR,
                      snapshot_path=os.path.join(params.SNAPSHOT_DIR, ''))

    trainer.train(n_epochs=int(params.N_EPOCHS), do_validate=params.DO_VALIDATE)
    trainer.writer.close()
    destroy_process_group()
