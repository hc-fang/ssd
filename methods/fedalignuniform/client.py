from typing import Any, Optional
import copy
import numpy as np
import torch
import torch.nn
import torch.optim
from torch.utils.data import Dataset
from tqdm import tqdm
from fedbox.typing import SizedIterable

from ..loss import align_loss, uniform_loss, subspace_loss, distillation_loss, mse_loss


class FedAlignUniformClient:
    def __init__(
        self,
        *,
        id: int,
        encoder: torch.nn.Module,
        projector: torch.nn.Module,
        aug_train_loader: SizedIterable[tuple[torch.Tensor, torch.Tensor]],
        train_set: Dataset[tuple[torch.Tensor, torch.Tensor]],
        # --- config ---
        local_epochs: int,
        lr: float,
        momentum: float = 0.9,
        weight_decay: float,
        temperature: float,
        uniform_coeff: float,
        subspace_coeff: float,
        # subspace,
        subspace_norm: bool,
        subspace_detach: bool,
        # distillation
        distill_coeff: float,
        distill_method: str,
        device: torch.device,
        use_autocast: bool = True,
    ):
        self.id = id
        self.encoder = encoder
        self.projector = projector
        self.aug_train_loader = aug_train_loader
        self.train_set = train_set
        self.local_epochs = local_epochs
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.temperature = temperature
        self.device = device
        self.uniform_coeff = uniform_coeff
        self.subspace_coeff = subspace_coeff
        # subspace 
        self.subspace_norm = subspace_norm
        self.subspace_detach = subspace_detach
        # distill
        self.distill_coeff = distill_coeff
        self.distill_method = distill_method

        self.use_autocast = use_autocast

    def fit(
        self,
        global_encoder: Optional[torch.nn.Module] = None,
        global_projector: Optional[torch.nn.Module] = None,
        lr: Optional[float] = None,
        current_round: int = None,
        subspace = None
    ) -> dict[str, Any]:

        self.encoder.load_state_dict(global_encoder.state_dict())
        self.projector.load_state_dict(global_projector.state_dict())

        self.optimizer = self.configure_optimizer()

        for m in (self.encoder, self.projector, global_encoder, global_projector):
            m.to(self.device)
            m.train()

        losses, alignment_losses, uniformity_losses = [], [], []
        subspace_losses, dis_losses = [], []

        encoder_params_global = copy.deepcopy(list(self.encoder.parameters()))

        scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.local_epochs): #, desc=f'client {self.id}', leave=False):
            for x1, x2 in self.aug_train_loader: #, desc=f'epoch {epoch}', leave=False):
                if x1.shape[0] == 1:
                    continue
                x1, x2 = x1.to(self.device), x2.to(self.device)
                self.subspace = subspace.to(self.device) if subspace is not None else None

                with torch.cuda.amp.autocast(enabled=self.use_autocast):
                    z1_enc, z2_enc = self.encoder(x1), self.encoder(x2)
                    z1_pro, z2_pro = self.projector(z1_enc), self.projector(z2_enc)

                    # operate on unit hypersphere
                    z1_norm = torch.nn.functional.normalize(z1_pro, dim=1)
                    z2_norm = torch.nn.functional.normalize(z2_pro, dim=1)

                    # align positive pairs
                    alignment_loss = align_loss(z1_norm, z2_norm)

                    # make the distribution uniformly distributed on the hypersphere
                    uniformity_loss = (uniform_loss(z1_norm) + uniform_loss(z2_norm)) / 2

                    loss = alignment_loss + self.uniform_coeff * uniformity_loss

                    # regularize the features in a subspace 
                    if self.subspace_coeff > 0:
                        ss_loss = (subspace_loss(z1_norm, self.subspace, self.subspace_detach, self.subspace_norm) + subspace_loss(z2_norm, self.subspace, self.subspace_detach, self.subspace_norm)) / 2
                        loss += self.subspace_coeff * ss_loss
                        subspace_losses.append(ss_loss.item())                    
                    
                    # distill from the projector to the encoder
                    if self.distill_method == 'KL':
                        dis_loss = (distillation_loss(z1_enc, z1_pro) + distillation_loss(z2_enc, z2_pro)) / 2
                    elif self.distill_method == 'MSE':
                        dis_loss = (mse_loss(z1_enc, z1_pro) + mse_loss(z2_enc, z2_pro)) / 2

                    if self.distill_coeff > 0:
                        loss += self.distill_coeff * dis_loss
                        
                    dis_losses.append(dis_loss.item())  
                    
                self.optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
                
                losses.append(loss.item())
                alignment_losses.append(alignment_loss.item())
                uniformity_losses.append(uniformity_loss.item())

        for m in (self.encoder, self.projector):
            m.cpu()


        return {
            "id": self.id,
            'train_loss': np.mean(losses),
            "encoder": self.encoder,
            "projector": self.projector,
            "alignment_loss": np.mean(alignment_losses),
            "uniformity_loss": np.mean(uniformity_losses),
            "subspace_loss": np.mean(subspace_losses) if subspace_losses else None,
            "distillation_loss": np.mean(dis_losses) if dis_losses else None,
        }

    def gen_feat(self):
        for m in (self.encoder, self.projector):
            m.to(self.device)
            m.eval()

        feats = []

        with torch.no_grad():
            for x1, _ in self.aug_train_loader:
                if x1.shape[0] == 1:
                    continue
                x1 = x1.to(self.device)
                z1 = torch.nn.functional.normalize(self.projector(self.encoder(x1)), dim=1)
                feats.append(z1.cpu())
        
        feats = np.concatenate(feats)
        return feats

    def schedule_lr(self, lr: float):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
    def configure_optimizer(self):
        return torch.optim.SGD(
            [*self.encoder.parameters(), *self.projector.parameters()],
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay
        )

    def make_checkpoint(self) -> dict[str, Any]:
        return {
            'encoder': self.encoder.state_dict(),
            'projector': self.projector.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }

    def load_checkpoint(self, checkpoint: dict[str, Any]):
        self.encoder.load_state_dict(checkpoint['encoder'])
        self.projector.load_state_dict(checkpoint['projector'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
