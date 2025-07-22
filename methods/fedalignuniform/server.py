import random
import os
import numpy as np
import torch
import wandb
import json
from typing import Any, Optional
from torch.utils.data import Dataset, ConcatDataset, DataLoader
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import concurrent.futures
import torch.multiprocessing as mp

from fedbox.utils.functional import model_average
from fedbox.utils.training import EarlyStopper as Recorder
from .client import FedAlignUniformClient
from utils.evaluate import knn_evaluate, calc_uniform
from utils.optim import cosine_learning_rates



class FedAlignUniformServer:
    def __init__(
        self,
        *,
        clients: list[FedAlignUniformClient],
        encoder: torch.nn.Module,
        projector: torch.nn.Module,
        test_set: Dataset[tuple[torch.Tensor, torch.Tensor]],
        global_rounds: int,
        join_ratio: float = 1.0,
        device: torch.device,
        checkpoint_dir: Optional[str] = None,

        # subspace 
        subspace_weight_dim: int = None,
        subspace_weight_range: list = None,
        subspace_mode: str = None,
    ):
        self.clients = clients
        self.encoder = encoder
        self.projector = projector
        self.test_set = test_set
        self.current_round = 0
        self.global_rounds = global_rounds
        self.join_ratio = join_ratio
        self.device = device
        self.checkpoint_dir = checkpoint_dir

        self.best_acc = 0

        self.subspace_weight_dim = subspace_weight_dim
        self.subspace_weight_range = subspace_weight_range
        self.subspace_mode = subspace_mode
        self.dim = 512

    @staticmethod
    def extract_buffers(model, state_dict):
        param_keys = {name for name, _ in model.named_parameters()}
        buffers = {
            name: tensor
            for name, tensor in state_dict.items()
            if name not in param_keys
        }
        return buffers
    
    def client_worker(self, rank, client, global_encoder, global_projector, lr, current_round, return_dict):
        """
        Runs `client.fit` on a specific GPU (determined by `rank`).
        """
        # Assign each process to a specific GPU
        device = torch.device(f'cuda:{rank}')
        global_encoder = global_encoder.to(device)
        global_projector = global_projector.to(device)

        # Run the client on the assigned device
        response = client.fit(
            global_encoder=global_encoder,
            global_projector=global_projector,
            lr=lr,
            current_round=current_round
        )
        
        return_dict[rank] = response

    
    def generate_weighted_subspaces(self, subspaces, basis_indices, weights):

        weighted_subspaces = []

        for subspace, client_indices, client_weight in zip(subspaces, basis_indices, weights):
            subspace[:, client_indices] *= torch.sqrt(torch.tensor(client_weight))
            weighted_subspaces.append(subspace)
        
        return weighted_subspaces

    def fit(self):
        learning_rates = cosine_learning_rates(self.clients[0].lr, self.global_rounds)

        if self.subspace_mode == 'fixed':
            print("Generate subspaces...")
            # generate dimensions to be weighted

            indices = np.arange(self.dim)
            np.random.shuffle(indices)
            n_clients = len(self.clients)
             
            subspace_indices = [
                indices[i * self.subspace_weight_dim: (i+1) * self.subspace_weight_dim]
                for i in range(n_clients)
            ] 

            # generate subspace weight
            subspace_weights = [np.random.uniform(low=self.subspace_weight_range[0], high=self.subspace_weight_range[1]) for _ in range(n_clients)]

            # generate random orthogonal basis
            random_matrix = torch.randn(self.dim, self.dim)
            orthonormal_basis, _ = torch.linalg.qr(random_matrix)  # QR decomposition
            random_subspaces = [orthonormal_basis.clone() for _ in range(n_clients)]

            subspaces = self.generate_weighted_subspaces(
                random_subspaces,
                subspace_indices,
                subspace_weights
            )
        else:
            subspaces = None


        for self.current_round in tqdm(range(self.current_round, self.global_rounds)):
            selected_clients: list[FedAlignUniformClient] = self.select_clients()
            client_weights = [
                len(client.aug_train_loader) for client in selected_clients
            ]

            responses = []

            for i, client in enumerate(tqdm(
                selected_clients, desc=f"round {self.current_round}, number of clients: {len(selected_clients)}", leave=False
            )):
                response = client.fit(
                    global_encoder=self.encoder,
                    global_projector=self.projector,
                    lr=learning_rates[self.current_round],
                    current_round=self.current_round,
                    subspace=subspaces[i] if subspaces else None
                )
                responses.append(response)

            self.encoder.load_state_dict(
                model_average(
                    [response["encoder"] for response in responses], client_weights
                )
            )

            self.projector.load_state_dict(
                model_average(
                    [response["projector"] for response in responses],
                    client_weights,
                )
            )

            log_info = {}

            for response in responses:
                for key, val in response.items():
                    if "loss" in key and val is not None:
                        if key not in log_info:
                            log_info[key] = []
                        log_info[key].append(val)
            
            for key in log_info:
                log_info[key] = np.mean(log_info[key])

            # evaluation
            self.gen_feat()
            acc = self.knn_test()
            

            if acc > self.best_acc:
                self.best_acc = acc

                if self.checkpoint_dir is not None:
                    torch.save(self.make_checkpoint(), f'{self.checkpoint_dir}/checkpoint_best.pth')
                
            # calculate uniformity 
            # uni_train_client, uni_train_all, uni_test = calc_uniform(self.train_feat, self.test_feat)
            # uni_train_client_p, uni_train_all_p, uni_test_p = calc_uniform(self.train_feat_p, self.test_feat_p)
                    
            # with open(f'{self.checkpoint_dir}/uniform_score.jsonl', 'a') as f:
            #     uniform_score = {
            #         "train": uni_train_all,
            #         "train_p": uni_train_all_p,
            #         "test": uni_test,
            #         "test_p": uni_test_p,
            #         "train_client": uni_train_client,
            #         "train_client_p": uni_train_client_p
            #     }
                # f.write(json.dumps(uniform_score) + '\n')                

            print(
                f"round {self.current_round}, knn acc: {acc:.4f}, loss: {log_info['train_loss']:.4g}"
            )

            log_info.update({
                "acc": acc,
                "best_knn_acc": self.best_acc,
                "lr": learning_rates[self.current_round],
                "train_encoder_feature_norm_mean": np.mean(self.train_encoder_feature_norm),
                "train_projector_feature_norm_mean": np.mean(self.train_projector_feature_norm),
                "test_encoder_feature_norm_mean": np.mean(self.test_encoder_feature_norm),
                "test_projector_feature_norm_mean": np.mean(self.test_projector_feature_norm) 
            })

            wandb.log(log_info)            

        if self.checkpoint_dir is not None:
            torch.save(self.make_checkpoint(), f'{self.checkpoint_dir}/checkpoint_last.pth')


    def gen_feat(self):
        for m in (self.encoder, self.projector):
            m.to(self.device)
            m.eval()

        train_loaders = [
            DataLoader(
                client.train_set,
                batch_size=512,
                shuffle=False,
                num_workers=4
            ) 
            for client in self.clients
        ]
        test_loader = DataLoader(self.test_set, batch_size=512, shuffle=False)

        # reinitialize 
        self.train_feat = [[] for _ in range(len(train_loaders))]
        self.train_feat_p = [[] for _ in range(len(train_loaders))]
        self.train_label = []
        self.test_feat = []
        self.test_feat_p = []
        self.test_label = []

        self.train_encoder_feature_norm = []
        self.train_projector_feature_norm = []
        self.test_encoder_feature_norm = []
        self.test_projector_feature_norm = []

        with torch.no_grad():
            for i, train_loader in enumerate(train_loaders):
                for x, y in train_loader:
                    x = x.to(self.device)
                    z = self.encoder(x)
                    z_p = self.projector(z)

                    self.train_encoder_feature_norm.extend(torch.linalg.norm(z, dim=1).cpu().tolist())
                    self.train_projector_feature_norm.extend(torch.linalg.norm(z_p, dim=1).cpu().tolist())

                    self.train_feat[i].append(torch.nn.functional.normalize(z, dim=1).cpu())
                    self.train_feat_p[i].append(torch.nn.functional.normalize(z_p, dim=1).cpu())
                    self.train_label.append(y)


            for x, y in test_loader:
                x = x.to(self.device)
                z = self.encoder(x)
                z_p = self.projector(z)

                self.test_encoder_feature_norm.extend(torch.linalg.norm(z, dim=1).cpu().tolist())
                self.test_projector_feature_norm.extend(torch.linalg.norm(z_p, dim=1).cpu().tolist())

                self.test_feat.append(torch.nn.functional.normalize(z, dim=1).cpu())
                self.test_feat_p.append(torch.nn.functional.normalize(z_p, dim=1).cpu())
                self.test_label.append(y)

        for i in range(len(self.train_feat)):
            self.train_feat[i] = torch.concat(self.train_feat[i])
            self.train_feat_p[i] = torch.concat(self.train_feat_p[i])

        self.train_label = torch.concat(self.train_label)
        self.test_feat = torch.concat(self.test_feat)
        self.test_feat_p = torch.concat(self.test_feat_p)
        self.test_label = torch.concat(self.test_label)


    def knn_test(self) -> float:
        # train_set = ConcatDataset([client.train_set for client in self.clients])
        acc = knn_evaluate(
            # encoder=self.encoder,
            # train_set=train_set,
            # test_set=self.test_set,
            train_z=torch.concat(self.train_feat),
            train_y=self.train_label,
            test_z=self.test_feat,
            test_y=self.test_label,
            device=self.device,
        )
        return acc

    def select_clients(self):
        return (
            self.clients
            if self.join_ratio == 1.0
            else random.sample(
                self.clients, int(round(len(self.clients) * self.join_ratio))
            )
        )

    def make_checkpoint(self, include_clients: bool = False) -> dict[str, Any]:
        checkpoint = {
            "current_round": self.current_round,
            "encoder": self.encoder.state_dict(),
            "projector": self.projector.state_dict(),
        }
        if include_clients:
            checkpoint["clients"] = [
                client.make_checkpoint() for client in self.clients
            ]
        return checkpoint

    def load_checkpoint(self, checkpoint: dict[str, Any]):
        self.current_round = checkpoint["current_round"]
        self.encoder.load_state_dict(checkpoint["encoder"])
        self.projector.load_state_dict(checkpoint["projector"])
        if "clients" in checkpoint:
            for client, client_checkpoint in zip(self.clients, checkpoint["clients"]):
                client.load_checkpoint(client_checkpoint)
