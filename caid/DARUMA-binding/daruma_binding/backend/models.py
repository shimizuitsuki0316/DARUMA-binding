import os, sys
from struct import unpack
import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.init as init

backend_dir = os.path.dirname(__file__)
sys.path.append(backend_dir)

# Layers
Dropout1d = getattr(nn, "Dropout1d", None) or nn.Dropout 

class PermuteLayer(nn.Module):
    def __init__(self, *dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(*self.dims)


class MovingAverageSmoothing:
    def __init__(self, kernel_size):
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

    def __call__(self, x):
        x = F.pad(x, (self.padding, self.padding), mode='replicate')
        x_unfolded = x.unfold(dimension=1, size=self.kernel_size, step=1)
        return x_unfolded.mean(dim=-1)
    

class MultiKernelConv1d_withDropout(nn.Module):
    """
    A module that applies multiple Conv1d layers with different kernel sizes and concatenates their outputs.
    - Input: (B, C_in, L)
    - Output: (B, C_out * len(kernel_sizes), L)
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_sizes: list[int], dropout_rates: list[float], relu: bool = True):
        super().__init__()

        if len(kernel_sizes) != len(dropout_rates):
            raise ValueError("kernel_sizes and dropout_rates must have the same length.")
        
        self.convs = nn.ModuleList()
        for k, dropout in zip(kernel_sizes, dropout_rates):
            conv_block = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=k, padding=k//2),
                nn.ReLU(inplace=True) if relu else nn.Identity(),
                Dropout1d(dropout)
            )
            self.convs.append(conv_block)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C_in, L)
        outs = [conv(x) for conv in self.convs]
        return torch.cat(outs, dim=1)


# Class
class myModel(nn.Module):
    def __init__(self, in_ch=1024, out_ch=2):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch

    def _load_weights(self, path):
        state_dict = torch.load(path)
        self.load_state_dict(state_dict)
    
    def _load_weights_to_cpu(self, path):
        state_dict = torch.load(path, map_location=torch.device('cpu'))
        self.load_state_dict(state_dict)
        
    @staticmethod
    def logit_to_prob(logits):
        return torch.softmax(logits, dim=-1)[..., 1]


class daruma_like(myModel):
    def __init__(self, dropout_rate=0.5, in_ch=1024, out_ch=2):
        super().__init__(in_ch=in_ch, out_ch=out_ch)

        # FEU
        self.FEU = nn.Sequential(
            nn.Conv1d(in_channels=self.in_ch, out_channels=128, kernel_size=9, padding=4),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=9, padding=4),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=9, padding=4),
            nn.ReLU(inplace=True),
        )

        # PU
        self.PU = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=121, padding=0),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            PermuteLayer(0, 2, 1),  # (1, 128, L) -> (1, L, 128)
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, self.out_ch),
        )
    
    def forward(self, x):
        x = F.pad(x, (60, 60), mode="constant", value=0)

        x = self.FEU(x)
        x = self.PU(x)

        return x # shape (1, L, out_ch)


class plddt_daruma_like_ProtT5_nr60(daruma_like):
    def __init__(self, dropout_rate=0.5, in_ch=1024):
        if in_ch != 1024:
            raise ValueError("in_ch must be 1024 for daruma_like_ProtT5_nr60_trained")
        super().__init__(dropout_rate=dropout_rate, in_ch=in_ch, out_ch=2)

class rsa_daruma_like_ProtT5_nr60(daruma_like):
    def __init__(self, dropout_rate=0.5, in_ch=1024):
        if in_ch != 1024:
            raise ValueError("in_ch must be 1024 for daruma_like_ProtT5_nr60_trained")
        super().__init__(dropout_rate=dropout_rate, in_ch=in_ch, out_ch=2)

class ss8_daruma_like8_ProtT5_nr60(daruma_like):
    def __init__(self, dropout_rate=0.5, in_ch=1024):
        if in_ch != 1024:
            raise ValueError("in_ch must be 1024 for daruma_like_ProtT5_nr60_trained")
        super().__init__(dropout_rate=dropout_rate, in_ch=in_ch, out_ch=8)


class af_daruma_like_ProtT5_features(myModel):
    def __init__(self, in_ch=1024, **kwargs):
        super().__init__(in_ch=in_ch, out_ch=2)

        self.af_plddt_model = plddt_daruma_like_ProtT5_nr60(dropout_rate=0)
        self.af_rsa_model = rsa_daruma_like_ProtT5_nr60(dropout_rate=0)
        self.af_ss8_model = ss8_daruma_like8_ProtT5_nr60(dropout_rate=0)
    
    def forward(self, x):
        plddt_logit = self.af_plddt_model(x)
        plddt_prob = torch.softmax(plddt_logit, dim=2)
        rsa_logit = self.af_rsa_model(x)
        rsa_prob = torch.softmax(rsa_logit, dim=2)
        ss8_logit = self.af_ss8_model(x)
        ss8_prob = torch.softmax(ss8_logit, dim=2)
        features = torch.cat([plddt_prob, rsa_prob, ss8_prob], dim=2)
        return features # (1, L, 12)


# AF-feature Model
class af_daruma_like_ProtT5_features_mk1dconv_relu(myModel):
    def __init__(self, dropout_rate=0.5, in_ch=1024, features=12, return_af_features=False):
        super().__init__(in_ch=in_ch, out_ch=2)

        self.mk1 = MultiKernelConv1d_withDropout(features, out_channels=8, kernel_sizes=[3, 7, 15, 31, 63, 127], dropout_rates=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], relu=True)

        self.output_layer = nn.Sequential(
            nn.Conv1d(48, 2, kernel_size=1, stride=1, padding=0),
        )

        self.af_feature_extractor = af_daruma_like_ProtT5_features()

        self.return_af_features = return_af_features

    def forward(self, x):
        with torch.no_grad():
            features = self.af_feature_extractor(x) # (1, L, 12)

        if self.return_af_features:
            features = features.detach().requires_grad_(True)
            self.af_features = features
        
        x = features.transpose(1, 2) # (1, 12, L)
        
        x = self.mk1(x) # (1, 48, L)
        x = self.output_layer(x) # (1, 2, L)

        return x.transpose(1, 2) # (1, L, 2)



class daruma_binding():
    def __init__(self, device):
        self.model = af_daruma_like_ProtT5_features_mk1dconv_relu(dropout_rate=0, in_ch=1024, return_af_features=True)
        weight_path = os.path.join(backend_dir, "model_weights/daruma_binding_weight.pth")
        self.model._load_weights_to_cpu(weight_path)
        self.model.to(device)
        self.model.eval()
        self.threshold = 0.26 # The threshold of 0.26 is used, which maximizes the F1 score on the validation data.
        self.smoother = MovingAverageSmoothing(kernel_size=7)
    
    def predict_from_x(self, x):
        with torch.no_grad():
            logit = self.model(x)
            prob = self.model.logit_to_prob(logit)

            prob = self.smoother(prob)
            pred = (prob > self.threshold).float()
        
        return prob.squeeze(0).cpu().numpy(), pred.squeeze(0).cpu().numpy()

    def get_af_features(self):
        return self.model.af_features



if __name__ == "__main__":
    # dummy input for testing
    device = torch.device("cpu")
    print("Using device:", device)

    model = daruma_binding(device)
    print(model.model)
    
    torch.manual_seed(42)
    dummy_input = torch.randn(1, 1024, 1000).to(device)
    print("Input shape:", dummy_input.shape)
    prob, pred = model.predict_from_x(dummy_input)
    print("Output shape:", prob.shape)
    print("Output sample:", prob[:5])



