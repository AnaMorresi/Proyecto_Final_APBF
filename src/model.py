import torch
import torch.nn as nn
import torch.nn.functional as F

# VAE convolucional completo para entrada (B, 1, 224, 224, 10).
# Sin flatten, sin capas lineales.
#
# Encoder: (224,224,10) -> (56,56,3) -> (14,14,1)  [stride-4, kernel-5]
# Latente z shape: (B, latent_ch, 14, 14, 1)


def _enc_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv3d(in_ch, out_ch, kernel_size=5, stride=4, padding=2),
        nn.BatchNorm3d(out_ch),
        nn.Dropout3d(p=0.1),
        nn.LeakyReLU(0.2, inplace=True),
    )


def _dec_block(in_ch: int, out_ch: int, output_padding) -> nn.Sequential:
    return nn.Sequential(
        nn.ConvTranspose3d(
            in_ch, out_ch, kernel_size=5, stride=4,
            padding=2, output_padding=output_padding,
        ),
        nn.BatchNorm3d(out_ch),
        nn.LeakyReLU(0.2, inplace=True),
    )


class Encoder(nn.Module):
    def __init__(self, latent_ch: int = 16):
        super().__init__()
        self.convs = nn.Sequential(
            _enc_block(1,  32),  # -> (B,  32, 56, 56, 3)
            _enc_block(32, 64),  # -> (B,  64, 14, 14, 1)
        )
        self.conv_mu = nn.Conv3d(64, latent_ch, kernel_size=1)
        self.conv_lv = nn.Conv3d(64, latent_ch, kernel_size=1)

    def forward(self, x: torch.Tensor):
        h = self.convs(x)
        return self.conv_mu(h), self.conv_lv(h)


class Decoder(nn.Module):
    def __init__(self, latent_ch: int = 16):
        super().__init__()
        self.conv_in = nn.Conv3d(latent_ch, 64, kernel_size=1)
        self.convs = nn.Sequential(
            _dec_block(64, 32, (3, 3, 2)),  # (14,14,1) -> (56,56,3)
            nn.ConvTranspose3d(             # (56,56,3) -> (224,224,10)
                32, 1, kernel_size=5, stride=4,
                padding=2, output_padding=(3, 3, 1),
            ),
        )

    def forward(self, z: torch.Tensor):
        return self.convs(self.conv_in(z))


class VAE3D(nn.Module):
    def __init__(self, latent_ch: int = 16):
        super().__init__()
        self.encoder = Encoder(latent_ch)
        self.decoder = Decoder(latent_ch)

    def reparametrize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        if self.training:
            return mu + torch.exp(0.5 * log_var) * torch.randn_like(mu)
        return mu

    def forward(self, x: torch.Tensor):
        mu, log_var = self.encoder(x)
        z = self.reparametrize(mu, log_var)
        return self.decoder(z), mu, log_var

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode deterministico -- retorna mu (B, latent_ch, 14, 14, 1)."""
        with torch.no_grad():
            mu, _ = self.encoder(x)
        return mu


# -- VAE3DDeep -- 4 bloques stride-2, kernel-3 
#
# Mas gradual que VAE3D:
#   Encoder: (224,224,10) -> (112,112,5) -> (56,56,3) -> (28,28,2) -> (14,14,1)
#   Latente: (latent_ch, 14, 14, 1)  -- misma resolucion espacial que VAE3D
#   ~607K params con latent_ch=64  (vs 532K de VAE3D)

def _enc_block_s2(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm3d(out_ch),
        nn.Dropout3d(p=0.1),
        nn.LeakyReLU(0.2, inplace=True),
    )


def _dec_block_s2(in_ch: int, out_ch: int, output_padding) -> nn.Sequential:
    return nn.Sequential(
        nn.ConvTranspose3d(
            in_ch, out_ch, kernel_size=3, stride=2,
            padding=1, output_padding=output_padding,
        ),
        nn.BatchNorm3d(out_ch),
        nn.LeakyReLU(0.2, inplace=True),
    )


class EncoderDeep(nn.Module):
    def __init__(self, latent_ch: int = 64):
        super().__init__()
        self.convs = nn.Sequential(
            _enc_block_s2(1,   16),   # (224,224,10) -> (112,112,5)
            _enc_block_s2(16,  32),   # (112,112,5)  -> (56, 56, 3)
            _enc_block_s2(32,  64),   # (56, 56, 3)  -> (28, 28, 2)
            _enc_block_s2(64,  128),  # (28, 28, 2)  -> (14, 14, 1)
        )
        self.conv_mu = nn.Conv3d(128, latent_ch, kernel_size=1)
        self.conv_lv = nn.Conv3d(128, latent_ch, kernel_size=1)

    def forward(self, x: torch.Tensor):
        h = self.convs(x)
        return self.conv_mu(h), self.conv_lv(h)


class DecoderDeep(nn.Module):
    def __init__(self, latent_ch: int = 64):
        super().__init__()
        self.conv_in = nn.Conv3d(latent_ch, 128, kernel_size=1)
        self.convs = nn.Sequential(
            _dec_block_s2(128, 64,  (1, 1, 1)),  # (14,14,1)  -> (28, 28, 2)
            _dec_block_s2(64,  32,  (1, 1, 0)),  # (28,28,2)  -> (56, 56, 3)
            _dec_block_s2(32,  16,  (1, 1, 0)),  # (56,56,3)  -> (112,112, 5)
            nn.ConvTranspose3d(                   # (112,112,5)-> (224,224,10)
                16, 1, kernel_size=3, stride=2,
                padding=1, output_padding=(1, 1, 1),
            ),
        )

    def forward(self, z: torch.Tensor):
        return self.convs(self.conv_in(z))


class VAE3DDeep(nn.Module):
    """VAE con 4 bloques stride-2 (kernel-3). Mas profundo que VAE3D."""

    def __init__(self, latent_ch: int = 64):
        super().__init__()
        self.encoder = EncoderDeep(latent_ch)
        self.decoder = DecoderDeep(latent_ch)

    def reparametrize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        if self.training:
            return mu + torch.exp(0.5 * log_var) * torch.randn_like(mu)
        return mu

    def forward(self, x: torch.Tensor):
        mu, log_var = self.encoder(x)
        z = self.reparametrize(mu, log_var)
        return self.decoder(z), mu, log_var

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode deterministico -- retorna mu (B, latent_ch, 14, 14, 1)."""
        with torch.no_grad():
            mu, _ = self.encoder(x)
        return mu


def vae_loss(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    mu: torch.Tensor,
    log_var: torch.Tensor,
    beta: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """ELBO loss. Retorna (total, mse, kl)."""
    mse = F.mse_loss(x_hat, x, reduction="mean")
    kl  = -0.5 * torch.mean(1.0 + log_var - mu.pow(2) - log_var.exp())
    return mse + beta * kl, mse, kl
