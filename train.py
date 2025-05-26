#!/usr/bin/env python

import os
from copy import deepcopy
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
    init_process_group(backend='nccl')

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
    optimizer = torch.optim.SGD(
        params=filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    # split_datasets_dict = generate_datasets(dataset_path=params.DATASET_PATH,
    #                                         portions = {'train': .5, 'val': .1, 'test': .1})

    train_loader = ShapeNetDataLoader.get_train_loader(train_dataset=datasets_dict['train'],
                                                       batch_size=train_batch_size,
                                                       n_cpus=n_cpus,
                                                       parallel=parallel)

    val_loader = ShapeNetDataLoader.get_val_loader(val_dataset=datasets_dict['val'],
                                                   parallel=parallel,
                                                   batch_size=val_batch_size,
                                                   n_cpus=n_cpus)
    # criterion = nn.MSELoss()

    return encoding_model, model, optimizer, scheduler, train_loader, val_loader #, criterion


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
            self.model = DDP(self.model, device_ids=[self.gpu_id])
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
        self.span_criterion = nn.MSELoss()
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

    def _criterion(self, z_dot, derivatives): # derivatives_shape (batch, 512, 6) # z_dot_shape (batch, 512)
        derivatives = derivatives.view(derivatives.shape[0], -1, derivatives.shape[-1])
        z_dot = z_dot.view(z_dot.shape[0], -1)
        derivatives_norms = torch.norm(derivatives, dim=1)
        # derivatives_unit = derivatives / derivatives_norms.unsqueeze(1)
        norm_loss = self.unit_criterion(derivatives_norms, torch.ones_like(derivatives_norms))
        z_unit = z_dot / torch.norm(z_dot, dim=1, keepdim=True)
        linear_fit = torch.linalg.lstsq(derivatives, z_unit.unsqueeze(2)) # might need to put z_unit.unsqueeze(2)
        residuals = torch.bmm(derivatives, linear_fit.solution) - z_unit.unsqueeze(2)
        se = residuals ** 2
        sse = se.sum(dim=1)
        span_loss = self.span_criterion(sse, torch.zeros_like(sse))

        return norm_loss + span_loss

    def _load_snapshot(self, snapshot_path):
        snapshot = torch.load(snapshot_path, map_location=self.device)
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
        loss_sum = 0.0
        for i_batch, (viewpoint1, viewpoint2) in enumerate(self.train_dataloader):
            if params.PARALLEL:
                viewpoint1 = viewpoint1.to(self.gpu_id)
                viewpoint2 = viewpoint2.to(self.gpu_id)
            with torch.no_grad():
                z1 = self.encoding_model(viewpoint1)
                z2 = self.encoding_model(viewpoint2)

            derivatives = self.model(z1)

            loss = self._criterion(z2 - z1, derivatives)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            loss_sum += loss.item()

            if i_batch % self.print_every == 0 and self.gpu_id == 0:
                self.writer.add_scalar("Loss/train", loss_sum / (i_batch + 1),
                                       epoch * len(self.train_dataloader) + i_batch)
                print(
                    f"TRN Epoch {epoch} | Batch {i_batch} / {len(self.train_dataloader)} | Loss {loss_sum / (i_batch + 1)}")

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
            for i_batch, (viewpoint1, viewpoint2) in enumerate(self.val_dataloader):
                if params.PARALLEL:
                    viewpoint1 = viewpoint1.to(self.gpu_id)
                    viewpoint2 = viewpoint2.to(self.gpu_id)
                with torch.no_grad():
                    z1 = self.encoding_model(viewpoint1)
                    z2 = self.encoding_model(viewpoint2)

                derivatives = self.model(z1)

                loss = self._criterion(z2 - z1, derivatives)
                loss_sum += loss.item()

            if i_batch % self.print_every == 0 and self.gpu_id == 0:
                self.writer.add_scalar("Loss/val", loss_sum / len(self.val_dataloader), self.epochs_run)
                print(
                    f"VAL Epoch {epoch} | Batch {i_batch} / {len(self.val_dataloader)} | Loss {loss_sum / (i_batch + 1)}")


if __name__ == "__main__":
    with open('./Parameters.json', 'r') as json_file:
        params = json.load(json_file,
                           object_hook=lambda d: SimpleNamespace(**d))
    if params.PARALLEL:
        rank = setup_ddp(parallel=params.PARALLEL)
    else:
        rank = 0

    if rank == 0:
        datasets_dict = generate_datasets(dataset_path=params.DATASET_PATH, test=False)
        if params.PARALLEL:
            torch.distributed.barrier()

    else:
        if params.PARALLEL:
            torch.distributed.barrier()
        datasets_dict = {'train': None, 'val': None, 'test': None}
        for split_name in ['train', 'val']:
            datasets_dict[split_name] = load_dataset(split_name=split_name)
    print('All nodes in sync, starting training...')

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

    trainer = Trainer(model=model,
                      encoding_model=encoding_model,
                      optimizer=optimizer,
                      scheduler=scheduler,
                      train_dataloader=train_dataloader,
                      val_dataloader=val_dataloader,
                      parallel=params.PARALLEL,
                      save_every=1,
                      print_every=1000,
                      snapshot_dir=params.SNAPSHOT_DIR,
                      snapshot_path=os.path.join(params.SNAPSHOT_DIR, ''))

    trainer.train(n_epochs=int(params.N_EPOCHS), do_validate=params.DO_VALIDATE)
    trainer.writer.close()
    destroy_process_group()
