"""
Standalone CODE-ADV training module extracted from this project.

Drop this file into another repo and import `train_code_adv` to pre-train
the domain-specific encoders (DSNBasisAE) with adversarial alignment.

Example:
    from standalone_code_adv import train_code_adv, get_device
    encoder, histories = train_code_adv(
        s_dataloaders=(s_train_loader, s_val_loader),
        t_dataloaders=(t_train_loader, t_val_loader),
        input_dim=542,
        latent_dim=128,
        encoder_hidden_dims=[512, 256, 128],
        classifier_hidden_dims=[256, 128],
        dop=0.1,
        num_geo_layer=1,
        norm_flag=True,
        lr=1e-3,
        pretrain_num_epochs=50,
        train_num_epochs=1000,
        retrain_flag=True,
        es_flag=False,
        model_save_folder="./code_adv_norm",
        device=get_device(),
    )
"""

from collections import defaultdict
from itertools import chain
from typing import Any, List, Tuple, Optional
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd


# -----------------------------------------------------------------------------#
# Utility
# -----------------------------------------------------------------------------#

def get_device() -> torch.device:
    """Return CUDA device if available, else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------------------------------------------------------#
# Gradient Reversal Layer
# -----------------------------------------------------------------------------#

class RevGradFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_: torch.Tensor, alpha_: torch.Tensor):
        ctx.save_for_backward(alpha_)
        return input_

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (alpha_,) = ctx.saved_tensors
        grad_input = -grad_output * alpha_ if ctx.needs_input_grad[0] else None
        return grad_input, None


def revgrad(input_: torch.Tensor, alpha_: torch.Tensor) -> torch.Tensor:
    return RevGradFn.apply(input_, alpha_)


class RevGrad(nn.Module):
    """Gradient reversal layer used in adversarial training."""

    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self._alpha = torch.tensor(alpha, requires_grad=False)

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        return revgrad(input_, self._alpha)


# -----------------------------------------------------------------------------#
# Base AE + MLP
# -----------------------------------------------------------------------------#

class BaseAE(nn.Module):
    """Minimal base class for autoencoders."""

    def __init__(self) -> None:
        super().__init__()

    def encode(self, input: torch.Tensor) -> List[torch.Tensor]:
        raise NotImplementedError

    def decode(self, input: torch.Tensor) -> List[torch.Tensor]:
        raise NotImplementedError

    def sample(self, batch_size: int, current_device: int, **kwargs) -> torch.Tensor:
        raise RuntimeWarning()

    def generate(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def loss_function(self, *inputs: Any, **kwargs) -> torch.Tensor:
        raise NotImplementedError


class MLP(nn.Module):
    """Simple MLP with optional gradient reversal head."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Optional[List[int]] = None,
        dop: float = 0.1,
        act_fn=nn.SELU,
        out_fn=None,
        gr_flag: bool = False,
    ) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.dop = dop

        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        modules: List[nn.Module] = []
        if gr_flag:
            modules.append(RevGrad())

        modules.append(
            nn.Sequential(
                nn.Linear(input_dim, hidden_dims[0], bias=True),
                act_fn(),
                nn.Dropout(self.dop),
            )
        )

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.Linear(hidden_dims[i], hidden_dims[i + 1], bias=True),
                    act_fn(),
                    nn.Dropout(self.dop),
                )
            )

        self.module = nn.Sequential(*modules)

        if out_fn is None:
            self.output_layer = nn.Linear(hidden_dims[-1], output_dim, bias=True)
        else:
            self.output_layer = nn.Sequential(
                nn.Linear(hidden_dims[-1], output_dim, bias=True),
                out_fn(),
            )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        embed = self.module(input)
        output = self.output_layer(embed)
        return output


# -----------------------------------------------------------------------------#
# DSN Basis Autoencoder
# -----------------------------------------------------------------------------#

class DSNBasisAE(BaseAE):
    """
    Domain separation network autoencoder with shared/private encoders and
    orthogonality regularization.
    """

    def __init__(
        self,
        shared_encoder: nn.Module,
        decoder: nn.Module,
        num_geo_layer: int,
        input_dim: int,
        latent_dim: int,
        alpha: float = 1.0,
        beta: float = 1.0,
        gamma: float = 1.0,
        eta: float = 1.0,
        basis_weight: torch.Tensor = torch.ones(1),
        hidden_dims: Optional[List[int]] = None,
        dop: float = 0.1,
        noise_flag: bool = False,
        norm_flag: bool = False,
        cns_basis_label_loss: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.eta = eta
        self.basis_weight = basis_weight
        self.noise_flag = noise_flag
        self.dop = dop
        self.norm_flag = norm_flag

        self.cns_basis_label_loss = cns_basis_label_loss
        self.num_geo_layer = num_geo_layer

        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        self.shared_encoder = shared_encoder
        self.decoder = decoder

        self.private_encoder = MLP(
            input_dim=input_dim,
            output_dim=latent_dim,
            hidden_dims=hidden_dims,
            dop=self.dop,
        )

        self.softmax = nn.Softmax(dim=-1)

    def p_encode(self, fet_input: torch.Tensor) -> torch.Tensor:
        if self.noise_flag and self.training:
            latent_code = self.private_encoder(fet_input + torch.randn_like(fet_input) * 0.1)
        else:
            latent_code = self.private_encoder(fet_input)

        return F.normalize(latent_code, p=2, dim=1) if self.norm_flag else latent_code

    def s_encode(self, fet_input: torch.Tensor) -> torch.Tensor:
        if self.noise_flag and self.training:
            latent_code = self.shared_encoder(fet_input + torch.randn_like(fet_input) * 0.1)
        else:
            latent_code = self.shared_encoder(fet_input)
        return F.normalize(latent_code, p=2, dim=1) if self.norm_flag else latent_code

    def encode(self, input: torch.Tensor) -> torch.Tensor:
        p_latent_code = self.p_encode(input)
        s_latent_code = self.s_encode(input)
        return torch.cat((p_latent_code, s_latent_code), dim=1)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, input: torch.Tensor, **kwargs) -> List[torch.Tensor]:
        z = self.encode(input)
        return [input, self.decode(z), z]

    def loss_function(self, *args, **kwargs) -> dict:
        input_ = args[0]
        recons = args[1]
        z = args[2]

        p_z = z[:, : z.shape[1] // 2]
        s_z = z[:, z.shape[1] // 2 :]

        recons_loss = F.mse_loss(input_, recons)

        s_l2_norm = torch.norm(s_z, p=2, dim=1, keepdim=True).detach()
        s_l2 = s_z.div(s_l2_norm.expand_as(s_z) + 1e-6)

        p_l2_norm = torch.norm(p_z, p=2, dim=1, keepdim=True).detach()
        p_l2 = p_z.div(p_l2_norm.expand_as(p_z) + 1e-6)

        ortho_loss = torch.mean((s_l2.t().mm(p_l2)).pow(2))
        loss = recons_loss + self.alpha * ortho_loss
        return {"loss": loss, "recons_loss": recons_loss, "ortho_loss": ortho_loss}

    def sample(self, num_samples: int, current_device: int, **kwargs) -> torch.Tensor:
        z = torch.randn(num_samples, self.latent_dim).to(current_device)
        return self.decode(z)

    def generate(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.forward(x)[1]


# -----------------------------------------------------------------------------#
# Eval helpers
# -----------------------------------------------------------------------------#

def model_save_check(history: dict, metric_name: str, tolerance_count: int = 5, reset_count: int = 1):
    save_flag = False
    stop_flag = False
    if "best_index" not in history:
        history["best_index"] = 0
    if metric_name.endswith("loss"):
        if history[metric_name][-1] <= history[metric_name][history["best_index"]]:
            save_flag = True
            history["best_index"] = len(history[metric_name]) - 1
    else:
        if history[metric_name][-1] >= history[metric_name][history["best_index"]]:
            save_flag = True
            history["best_index"] = len(history[metric_name]) - 1

    if len(history[metric_name]) - history["best_index"] > tolerance_count * reset_count and history["best_index"] > 0:
        stop_flag = True

    return save_flag, stop_flag


def eval_basis_dsnae_epoch(model: DSNBasisAE, data_loader, device: torch.device, history: dict):
    model.eval()
    avg_loss_dict = defaultdict(float)
    for x_batch in data_loader:
        x_batch = x_batch[0].to(device)
        with torch.no_grad():
            loss_dict = model.loss_function(*(model(x_batch)))
            for k, v in loss_dict.items():
                avg_loss_dict[k] += v.cpu().detach().item() / len(data_loader)

    for k, v in avg_loss_dict.items():
        history[k].append(v)
    return history


def basis_dsn_ae_train_step(
    s_dsnae: DSNBasisAE,
    t_dsnae: DSNBasisAE,
    s_batch,
    t_batch,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    history: dict,
    scheduler=None,
):
    s_dsnae.zero_grad()
    t_dsnae.zero_grad()
    s_dsnae.train()
    t_dsnae.train()

    s_x = s_batch[0].to(device)
    t_x = t_batch[0].to(device)

    s_loss_dict = s_dsnae.loss_function(*s_dsnae(s_x))
    t_loss_dict = t_dsnae.loss_function(*t_dsnae(t_x))

    optimizer.zero_grad()
    loss = s_loss_dict["loss"] + t_loss_dict["loss"]
    loss.backward()

    optimizer.step()
    if scheduler is not None:
        scheduler.step()
    loss_dict = {k: v.cpu().detach().item() + t_loss_dict[k].cpu().detach().item() for k, v in s_loss_dict.items()}

    for k, v in loss_dict.items():
        history[k].append(v)

    return history


# -----------------------------------------------------------------------------#
# Adversarial training steps
# -----------------------------------------------------------------------------#

def compute_gradient_penalty(critic, real_samples, fake_samples, device):
    """Calculates the gradient penalty loss for WGAN-GP."""
    alpha = torch.rand((real_samples.shape[0], 1), device=device)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    critic_interpolates = critic(interpolates)
    fakes = torch.ones((real_samples.shape[0], 1), device=device)
    gradients = autograd.grad(
        outputs=critic_interpolates,
        inputs=interpolates,
        grad_outputs=fakes,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def critic_dsn_train_step(
    critic,
    s_dsnae: DSNBasisAE,
    t_dsnae: DSNBasisAE,
    s_batch,
    t_batch,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    history: dict,
    scheduler=None,
    clip=None,
    gp: Optional[float] = None,
):
    critic.zero_grad()
    s_dsnae.zero_grad()
    t_dsnae.zero_grad()
    s_dsnae.eval()
    t_dsnae.eval()
    critic.train()

    s_x = s_batch[0].to(device)
    t_x = t_batch[0].to(device)

    s_code = s_dsnae.encode(s_x)
    t_code = t_dsnae.encode(t_x)
    loss = torch.mean(critic(t_code)) - torch.mean(critic(s_code))

    if gp is not None:
        gradient_penalty = compute_gradient_penalty(critic, real_samples=s_code, fake_samples=t_code, device=device)
        loss = loss + gp * gradient_penalty

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if clip is not None:
        for p in critic.parameters():
            p.data.clamp_(-clip, clip)
    if scheduler is not None:
        scheduler.step()

    history["critic_loss"].append(loss.cpu().detach().item())
    return history


def gan_dsn_gen_train_step(
    critic,
    s_dsnae: DSNBasisAE,
    t_dsnae: DSNBasisAE,
    s_batch,
    t_batch,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    alpha: float,
    history: dict,
    scheduler=None,
):
    critic.zero_grad()
    s_dsnae.zero_grad()
    t_dsnae.zero_grad()
    critic.eval()
    s_dsnae.train()
    t_dsnae.train()

    s_x = s_batch[0].to(device)
    t_x = t_batch[0].to(device)

    t_code = t_dsnae.encode(t_x)

    optimizer.zero_grad()
    gen_loss = -torch.mean(critic(t_code))
    s_loss_dict = s_dsnae.loss_function(*s_dsnae(s_x))
    t_loss_dict = t_dsnae.loss_function(*t_dsnae(t_x))
    recons_loss = s_loss_dict["loss"] + t_loss_dict["loss"]
    loss = recons_loss + alpha * gen_loss

    loss.backward()
    optimizer.step()
    if scheduler is not None:
        scheduler.step()

    loss_dict = {k: v.cpu().detach().item() + t_loss_dict[k].cpu().detach().item() for k, v in s_loss_dict.items()}
    for k, v in loss_dict.items():
        history[k].append(v)
    history["gen_loss"].append(gen_loss.cpu().detach().item())

    return history


# -----------------------------------------------------------------------------#
# Public API
# -----------------------------------------------------------------------------#

def train_code_adv(
    s_dataloaders: Tuple,
    t_dataloaders: Tuple,
    **kwargs,
):
    """
    Adversarial CODE-AE training.

    Args (kwargs):
        input_dim (int): feature dimension of inputs.
        latent_dim (int): latent code dimension per private/shared encoder.
        encoder_hidden_dims (list[int]): hidden layer sizes for encoders/decoder.
        classifier_hidden_dims (list[int]): hidden sizes for confounding classifier.
        dop (float): dropout rate.
        num_geo_layer (int): kept for parity with original signature.
        norm_flag (bool): whether to L2-normalize latent codes.
        lr (float): learning rate for optimizers.
        pretrain_num_epochs (int): epochs for reconstruction pretraining.
        train_num_epochs (int): epochs for WGAN alignment.
        retrain_flag (bool): if False, will load encoder weights from `model_save_folder`.
        es_flag (bool): enable early-stop/save during pretraining.
        model_save_folder (str): path to save encoder weights.
        device (torch.device): computation device.
    Returns:
        shared_encoder (nn.Module), histories (tuple of training logs).
    """

    s_train_dataloader, s_test_dataloader = s_dataloaders
    t_train_dataloader, t_test_dataloader = t_dataloaders

    device = kwargs["device"]

    shared_encoder = MLP(
        input_dim=kwargs["input_dim"],
        output_dim=kwargs["latent_dim"],
        hidden_dims=kwargs["encoder_hidden_dims"],
        dop=kwargs["dop"],
    ).to(device)

    shared_decoder = MLP(
        input_dim=2 * kwargs["latent_dim"],
        output_dim=kwargs["input_dim"],
        hidden_dims=kwargs["encoder_hidden_dims"][::-1],
        dop=kwargs["dop"],
    ).to(device)

    s_dsnae = DSNBasisAE(
        shared_encoder=shared_encoder,
        decoder=shared_decoder,
        input_dim=kwargs["input_dim"],
        latent_dim=kwargs["latent_dim"],
        hidden_dims=kwargs["encoder_hidden_dims"],
        dop=kwargs["dop"],
        num_geo_layer=kwargs["num_geo_layer"],
        cns_basis_label_loss=False,
        norm_flag=kwargs["norm_flag"],
    ).to(device)

    t_dsnae = DSNBasisAE(
        shared_encoder=shared_encoder,
        decoder=shared_decoder,
        input_dim=kwargs["input_dim"],
        latent_dim=kwargs["latent_dim"],
        hidden_dims=kwargs["encoder_hidden_dims"],
        dop=kwargs["dop"],
        cns_basis_label_loss=False,
        num_geo_layer=kwargs["num_geo_layer"],
        norm_flag=kwargs["norm_flag"],
    ).to(device)

    confounding_classifier = MLP(
        input_dim=kwargs["latent_dim"] * 2,
        output_dim=1,
        hidden_dims=kwargs["classifier_hidden_dims"],
        dop=kwargs["dop"],
    ).to(device)

    ae_params = [
        t_dsnae.private_encoder.parameters(),
        s_dsnae.private_encoder.parameters(),
        shared_decoder.parameters(),
        shared_encoder.parameters(),
    ]
    t_ae_params = [
        t_dsnae.private_encoder.parameters(),
        s_dsnae.private_encoder.parameters(),
        shared_decoder.parameters(),
        shared_encoder.parameters(),
    ]

    ae_optimizer = torch.optim.AdamW(chain(*ae_params), lr=kwargs["lr"])
    classifier_optimizer = torch.optim.RMSprop(confounding_classifier.parameters(), lr=kwargs["lr"])
    t_ae_optimizer = torch.optim.RMSprop(chain(*t_ae_params), lr=kwargs["lr"])

    dsnae_train_history = defaultdict(list)
    dsnae_val_history = defaultdict(list)
    critic_train_history = defaultdict(list)
    gen_train_history = defaultdict(list)

    if kwargs["retrain_flag"]:
        for epoch in range(int(kwargs["pretrain_num_epochs"])):
            if epoch % 50 == 0:
                print(f"AE training epoch {epoch}")
            for step, s_batch in enumerate(s_train_dataloader):
                t_batch = next(iter(t_train_dataloader))
                dsnae_train_history = basis_dsn_ae_train_step(
                    s_dsnae=s_dsnae,
                    t_dsnae=t_dsnae,
                    s_batch=s_batch,
                    t_batch=t_batch,
                    device=device,
                    optimizer=ae_optimizer,
                    history=dsnae_train_history,
                )

            dsnae_val_history = eval_basis_dsnae_epoch(
                model=s_dsnae, data_loader=s_test_dataloader, device=device, history=dsnae_val_history
            )
            dsnae_val_history = eval_basis_dsnae_epoch(
                model=t_dsnae, data_loader=t_test_dataloader, device=device, history=dsnae_val_history
            )

            for k in list(dsnae_val_history.keys()):
                if k != "best_index":
                    dsnae_val_history[k][-2] += dsnae_val_history[k][-1]
                    dsnae_val_history[k].pop()

            if kwargs["es_flag"]:
                save_flag, stop_flag = model_save_check(dsnae_val_history, metric_name="loss", tolerance_count=20)
                if save_flag:
                    os.makedirs(kwargs["model_save_folder"], exist_ok=True)
                    torch.save(s_dsnae.state_dict(), os.path.join(kwargs["model_save_folder"], "a_s_dsnae.pt"))
                    torch.save(t_dsnae.state_dict(), os.path.join(kwargs["model_save_folder"], "a_t_dsnae.pt"))
                if stop_flag:
                    break
        if kwargs["es_flag"]:
            s_dsnae.load_state_dict(torch.load(os.path.join(kwargs["model_save_folder"], "a_s_dsnae.pt")))
            t_dsnae.load_state_dict(torch.load(os.path.join(kwargs["model_save_folder"], "a_t_dsnae.pt")))

        for epoch in range(int(kwargs["train_num_epochs"])):
            if epoch % 50 == 0:
                print(f"confounder WGAN epoch {epoch}")
            for step, s_batch in enumerate(s_train_dataloader):
                t_batch = next(iter(t_train_dataloader))
                critic_train_history = critic_dsn_train_step(
                    critic=confounding_classifier,
                    s_dsnae=s_dsnae,
                    t_dsnae=t_dsnae,
                    s_batch=s_batch,
                    t_batch=t_batch,
                    device=device,
                    optimizer=classifier_optimizer,
                    history=critic_train_history,
                    gp=10.0,
                )
                if (step + 1) % 2 == 0:
                    gen_train_history = gan_dsn_gen_train_step(
                        critic=confounding_classifier,
                        s_dsnae=s_dsnae,
                        t_dsnae=t_dsnae,
                        s_batch=s_batch,
                        t_batch=t_batch,
                        device=device,
                        optimizer=t_ae_optimizer,
                        alpha=1.0,
                        history=gen_train_history,
                    )

        os.makedirs(kwargs["model_save_folder"], exist_ok=True)
        torch.save(s_dsnae.state_dict(), os.path.join(kwargs["model_save_folder"], "a_s_dsnae.pt"))
        torch.save(t_dsnae.state_dict(), os.path.join(kwargs["model_save_folder"], "a_t_dsnae.pt"))

    else:
        try:
            t_dsnae.load_state_dict(torch.load(os.path.join(kwargs["model_save_folder"], "a_t_dsnae.pt")))
        except FileNotFoundError:
            raise FileNotFoundError("No pre-trained encoder found in model_save_folder.")

    return t_dsnae.shared_encoder, (dsnae_train_history, dsnae_val_history, critic_train_history, gen_train_history)


__all__ = [
    "train_code_adv",
    "DSNBasisAE",
    "MLP",
    "RevGrad",
    "BaseAE",
    "get_device",
    "model_save_check",
]
