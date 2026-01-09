import torch
import torch.nn as nn
import math
import torch.nn.functional as F


def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)

# Taken from [3]
def causal_mask(size):
  mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int)
  return mask == 0


class LayerNormalization(nn.Module):
    def __init__(self, eps: float = 10**-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))    # Multiplies
        self.bias = nn.Parameter(torch.zeros(1))    # Added

    def forward(self, x):
        mean = x.mean(dim = -1, keepdim = True)
        std = x.std(dim = -1, keepdim = True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias

class MLP(nn.Module):
  def __init__(self, img_size: int, d_ff: int): # d_ff: feed forward dimension
    super().__init__()
    self.linear_1 = nn.Linear(img_size, d_ff)
    self.gelu = nn.GELU()
    self.linear_2 = nn.Linear(d_ff, img_size)

  def forward(self, x):
    out_linear_1 = self.linear_1(x)
    out_gelu = self.gelu(out_linear_1)
    out_linear_2 = self.linear_2(out_gelu)
    return out_linear_2


class TMSA(nn.Module):
  def __init__(self, d_model: int, num_heads: int, dropout: float, img_size: int):    # d_model = space_embedding_size = time_embedding_size
    super().__init__()
    self.space_embedding_size = d_model
    self.time_embedding_size = d_model
    self.d_model = d_model
    self.num_heads = num_heads
    self.seq_len = img_size * img_size
    self.img_size = img_size
    self.d = d_model // num_heads
    self.mask = causal_mask(self.seq_len)
    assert d_model % num_heads == 0, 'space_embedding_size is not divisible by num_heads!'

    # Linear projections for xs
    self.Wqs = nn.Linear(d_model, d_model, bias = False) # y = x*A^T + b  => A^T = (space_embedding_size, space_embedding_size), Wqs = A^T
    self.Wks = nn.Linear(d_model, d_model, bias = False)
    self.Wvs = nn.Linear(d_model, d_model, bias = False)

    # Linear projections for xt
    self.Wqt = nn.Linear(d_model, d_model, bias = False)
    self.Wkt = nn.Linear(d_model, d_model, bias = False)
    self.Wvt = nn.Linear(d_model, d_model, bias = False)

    self.WK = nn.Linear(self.d, self.seq_len, bias = False) # y = x*A^T + b, A^T = w^K

    self.wo = nn.Linear(d_model, d_model, bias=False) # Wo

    self.dropout = nn.Dropout(dropout)

  @staticmethod
  def compute_attention_scores(query, key, value, wK, mask, dropout: nn.Dropout):
    d = query.shape[-1]

    attention_scores = ((query @ key.transpose(-2, -1) + wK(query)) / math.sqrt(d))

    # Apply mask if required
    if mask is not None:
      attention_scores.masked_fill_(mask == 0, -1e9)

    # Apply softmax
    attention_scores = F.softmax(attention_scores, dim = -1)

    # Apply dropout if required
    if dropout is not None:
      attention_scores = dropout(attention_scores)

    # return here
    return attention_scores @ value # ,attention_scores for visualization

  def forward(self, xs, xt):
    xs = xs.view(xs.shape[0], self.seq_len, xs.shape[1])

    # Space query, key and value
    query_s = self.Wqs(xs)      # query.shape: (batch, seq_len, d_model)
    key_s = self.Wks(xs)
    value_s = self.Wvs(xs)

    qs_1 = query_s.view(query_s.shape[0], query_s.shape[1], self.num_heads, self.d).transpose(1, 2)
    ks_1 = key_s.view(key_s.shape[0], key_s.shape[1], self.num_heads, self.d).transpose(1, 2)
    vs_1 = value_s.view(value_s.shape[0], value_s.shape[1], self.num_heads, self.d).transpose(1, 2)

    # Temporal query, key and value
    query_t = self.Wqt(xt)
    key_t = self.Wkt(xt)
    value_t = self.Wvt(xt)

    qt_1 = query_t.view(query_t.shape[0], -1, self.num_heads, self.d).transpose(1, 2)
    kt_1 = key_t.view(key_t.shape[0], -1, self.num_heads, self.d).transpose(1, 2)
    vt_1 = value_t.view(value_t.shape[0], -1, self.num_heads, self.d).transpose(1, 2)

    # Concatenation
    qs = qs_1 + qt_1
    ks = ks_1 + kt_1
    vs = vs_1 + vt_1

    # Compute attention scores
    h = self.compute_attention_scores(qs, ks, vs, self.WK, self.mask, self.dropout)

    # Combine all the heads together
    h = h.transpose(1, 2).contiguous().view(h.shape[0], -1, self.num_heads * self.d)

    output = self.wo(h)
    output = output.view(h.shape[0], h.shape[2], int(math.sqrt(h.shape[1])), -1)

    # Multiply by Wo
    return output


class DiffiTBlock(nn.Module):
  def __init__(self, d_model: int, num_heads: int, dropout: float, d_ff: int, img_size: int, label_size: int = None):
    super().__init__()
    self.ln = LayerNormalization()
    self.tmsa = TMSA(d_model, num_heads, dropout, img_size)
    self.mlp = MLP(img_size, d_ff)
    self.time_embedding = TimeEmbedding(d_model, img_size*img_size)

    # Only for latent model
    if label_size is not None:
      self.label_size = label_size
      self.label_embedding = LabelEmbedding(label_size, d_model)

  def forward(self, xs, t, l=None):
    xt = self.time_embedding(t)
    tmsa_comb = xt

    if l is not None:
      tmsa_comb += self.label_embedding(l)

    xs1 = self.tmsa(self.ln(xs), tmsa_comb) + xs
    xs2 = self.mlp(self.ln(xs1)) + xs1

    return xs2

class Tokenizer(nn.Module):
  def __init__(self, out_channels: int, in_channels: int):
    super().__init__()
    self.conv3x3 = nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = 3, padding = 1)

  def forward(self, x):
    return self.conv3x3(x)


class Head(nn.Module):
  def __init__(self, in_channels: int, out_channels: int):
    super().__init__()
    self.group_norm = nn.GroupNorm(num_groups=in_channels//4, num_channels=in_channels)
    self.conv3x3 = nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = 3, padding = 1)

  def forward(self, x):
    return self.conv3x3(self.group_norm(x))


# From paragraph 3.2 of the DiffiT paper; implementation from the HuggingFace blog (https://huggingface.co/blog/annotated-diffusion)
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
      super().__init__()
      self.dim = dim

    def forward(self, time):
      device = time.device
      half_dim = self.dim // 2

      embeddings = math.log(10000) / (half_dim - 1)
      embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
      embeddings = time[:, None] * embeddings[None, :]
      embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
      return embeddings


# From paragraph 3.2 of the DiffiT paper
class TimeEmbedding(nn.Module):
  def __init__(self, d_model: int, seq_len: int):
    super().__init__()
    self.seq_len = seq_len
    self.d_model = d_model

    self.time_embedding_mlp = nn.Sequential(
        SinusoidalPositionEmbeddings(seq_len),
        nn.Linear(seq_len, d_model),
        nn.SiLU(),
        nn.Linear(d_model, d_model)
    )

  def forward(self, time_steps):
    return self.time_embedding_mlp(time_steps) # (batch, seq_len, d_model)


class DiffiTResBlock(nn.Module):
  def __init__(self, in_channels: int, out_channels: int, num_heads: int, dropout: float, d_ff: int, img_size: int, label_size: int = None):
    super().__init__()
    self.seq_len = img_size * img_size

    self.conv3x3 = nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = 3, padding = 1)
    self.swish = nn.SiLU()
    self.group_norm = nn.GroupNorm(num_groups = in_channels//4, num_channels = in_channels)
    self.diffit_block = DiffiTBlock(out_channels, num_heads, dropout, d_ff, img_size, label_size)

  def forward(self, xs, t, l=None):
    xs_1 = self.conv3x3(self.swish(self.group_norm(xs)))
    xs = xs + self.diffit_block(xs_1, t, l)

    return xs


# From page 14 of the DiffiT paper
class Downsample(nn.Module):
  def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 2, padding: int = 1):
    super().__init__()
    self.conv = nn.Conv2d(
        in_channels = in_channels,
        out_channels = out_channels,
        kernel_size = kernel_size,
        stride = stride,
        padding = padding
    )

  def forward(self, x):
    return self.conv(x)


class Upsample(nn.Module):
  def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 2, padding: int = 1, output_padding: int = 1):
    super().__init__()
    self.conv = nn.ConvTranspose2d(
        in_channels = in_channels,
        out_channels = out_channels,
        kernel_size = kernel_size,
        stride = stride,
        padding = padding,
        output_padding = output_padding
    )

  def forward(self, x):
    return self.conv(x)


class ResBlockGroup(nn.Module):
  def __init__(self, num_heads: int, dropout: float, d_ff: int, L: int, in_channels: int, out_channels: int, img_size: int, label_size: int = None):
    super().__init__()
    self.L = L
    self.diffit_res_block = DiffiTResBlock(in_channels, out_channels, num_heads, dropout, d_ff, img_size, label_size)

  def forward(self, x, t, l=None):
    for _ in range(self.L):
      x = self.diffit_res_block(x, t, l)
    return x

def p_losses(noise, predicted_noise):
  # return F.mse_loss(noise, predicted_noise)
  return F.smooth_l1_loss(noise, predicted_noise)

class DiffiTEncoder(nn.Module):
  def __init__(self, in_channels: int, d_model: int, num_heads: int, dropout: float, d_ff: int, img_size: int, label_size: int, L1: int = 4, L2: int = 4, L3: int = 4, L4: int = 4):
    super().__init__()
    d_model_2 = d_model*2

    self.tokenizer = Tokenizer(out_channels=d_model, in_channels=in_channels)
    self.diffit_res_block_group_1 = ResBlockGroup(num_heads, dropout, d_ff, L1, in_channels=d_model, out_channels=d_model, img_size=img_size, label_size=label_size)
    self.downsample_1 = Downsample(in_channels=d_model, out_channels=d_model_2)
    self.diffit_res_block_group_2 = ResBlockGroup(num_heads, dropout, d_ff, L2, in_channels=d_model_2, out_channels=d_model_2, img_size=img_size//2, label_size=label_size)
    self.downsample_2 = Downsample(in_channels=d_model_2, out_channels=d_model_2)
    self.diffit_res_block_group_3 = ResBlockGroup(num_heads, dropout, d_ff, L3, in_channels=d_model_2, out_channels=d_model_2, img_size=img_size//4, label_size=label_size)
    self.downsample_3 = Downsample(in_channels=d_model_2, out_channels=d_model_2)
    self.diffit_res_block_group_4 = ResBlockGroup(num_heads, dropout, d_ff, L4, in_channels=d_model_2, out_channels=d_model_2, img_size=img_size//8, label_size=label_size)

  def forward(self, x, t, l):
    out_1 = self.downsample_1(self.diffit_res_block_group_1(self.tokenizer(x), t, l))
    out_2 = self.downsample_2(self.diffit_res_block_group_2(out_1, t, l))
    out_3 = self.diffit_res_block_group_4(self.downsample_3(self.diffit_res_block_group_3(out_2, t, l)), t, l)
    return out_3


class DiffiTDecoder(nn.Module):
  def __init__(self, out_channels: int, d_model: int, num_heads: int, dropout: float, d_ff: int, img_size: int, label_size: int, L1: int = 4, L2: int = 4, L3: int = 4):
    super().__init__()
    d_model_2 = d_model//2

    self.upsample_1 = Upsample(in_channels=d_model, out_channels=d_model)
    self.diffit_res_block_group_3 = ResBlockGroup(num_heads, dropout, d_ff, L3, in_channels=d_model, out_channels=d_model, img_size=img_size//4, label_size=label_size)
    self.upsample_2 = Upsample(in_channels=d_model, out_channels=d_model)
    self.diffit_res_block_group_2 = ResBlockGroup(num_heads, dropout, d_ff, L2, in_channels=d_model, out_channels=d_model, img_size=img_size//2, label_size=label_size)
    self.upsample_3 = Upsample(in_channels=d_model, out_channels=d_model_2)
    self.diffit_res_block_group_1 = ResBlockGroup(num_heads, dropout, d_ff, L1, in_channels=d_model_2, out_channels=d_model_2, img_size=img_size, label_size=label_size)
    self.head = Head(in_channels=d_model_2, out_channels=out_channels)

  def forward(self, x, t, l):
    out_1 = self.upsample_1(x)
    out_2 = self.upsample_2(self.diffit_res_block_group_3(out_1, t, l))
    out_3 = self.diffit_res_block_group_1(self.upsample_3(self.diffit_res_block_group_2(out_2, t, l)), t, l)
    return self.head(out_3)


class LatentDiffiTTransformerBlock(nn.Module):
  def __init__(self, d_model: int, num_heads: int, dropout: float, d_ff: int, N: int, img_size: int, label_size: int):
    super().__init__()
    self.N = N
    self.diffit_block = DiffiTBlock(d_model, num_heads, dropout, d_ff, img_size=img_size, label_size=label_size)

  def forward(self, xs, t, l):
    for _ in range(self.N):
      xs = self.diffit_block(xs, t, l)
    return xs


class LabelEmbedding(nn.Module):
  def __init__(self, label_size: int, d_model: int):
    super().__init__()
    self.label_size = label_size

    self.embedding_layer = nn.Embedding(label_size, d_model)
    self.linear_layer = nn.Linear(d_model, d_model)

  def forward(self, l):
    return self.linear_layer(self.embedding_layer(l))


# Adapted from [6]
class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, embed_dim):
      super().__init__()
      assert (img_size % patch_size == 0), 'img_size is not divisible by patch_size!'

      self.proj = nn.Conv2d(
          in_channels = embed_dim,
          out_channels = embed_dim,
          kernel_size = patch_size,         # The receptive field will contain exactly one patch at a time
          stride = patch_size,              # No overlapping between patches; each patch is processed in isolation
      )

    def forward(self, x):
      # x.shape = (n_samples, in_channels, img_size, img_size)
      x = self.proj(x)   # (n_samples, embed_dim, n_patches ** 0.5, n_patches ** 0.5)
      return x


# Adapted from [6]
class Unpatch(nn.Module):
  def __init__(self, img_size, patch_size, embed_dim):
    super().__init__()
    assert (img_size % patch_size == 0), 'img_size is not divisible by patch_size!'

    self.proj = nn.ConvTranspose2d(
        in_channels = embed_dim,
        out_channels = embed_dim,
        kernel_size = patch_size,
        stride = patch_size
    )

  def forward(self, x):
    # x.shape = (n_samples, n_patches, embed_dim)
    x = self.proj(x)  # (n_samples, out_channels, img_size, img_size)
    return x


class DiffiT(nn.Module):
  def __init__(
      self,
      input_size=32,
      patch_size=2,
      in_channels=4,
      hidden_size=1152,
      depth=30,
      num_heads=16,
      mlp_ratio=4.0,
      dropout=0.1,
      num_classes=1000,
      learn_sigma=True,

      stride = 2,
  ):
    super().__init__()
    self.learn_sigma = learn_sigma
    self.in_channels = in_channels
    self.out_channels = in_channels * 2 if learn_sigma else in_channels
    self.patch_size = patch_size
    self.num_heads = num_heads
    self.label_size = num_classes
    self.d_ff = hidden_size * mlp_ratio

    self.image_size_input_latent_block = (input_size // stride**3) // patch_size    # stride = 2 defined by the paper
    self.seq_len_input_latent_block = self.image_size_input_latent_block * self.image_size_input_latent_block

    self.encoder = DiffiTEncoder(self.in_channels, hidden_size, num_heads, dropout, self.d_ff, img_size=input_size, label_size=num_classes)
    self.patch_embedding = PatchEmbedding(input_size, patch_size, hidden_size*2)
    self.latent_block = LatentDiffiTTransformerBlock(hidden_size*2, num_heads, dropout, self.d_ff, depth, img_size=self.image_size_input_latent_block, label_size=num_classes)
    self.unpatchify = Unpatch(input_size, patch_size, hidden_size*2)
    self.decoder = DiffiTDecoder(self.out_channels, hidden_size*2, num_heads, dropout, self.d_ff, img_size=input_size, label_size=num_classes)
  def forward(self, x, t, y):
    # image --> encoder
    encoder_output = self.encoder(x, t, y)
    # encoder --> patch embedding
    patch_embedding_output = self.patch_embedding(encoder_output)
    # patch embedding --> latent block
    latent_block_output = self.latent_block(patch_embedding_output, t, y)
    # latent block --> unpatchify
    unpatchify_output = self.unpatchify(latent_block_output)
    # unpatchify --> decoder
    decoder_output = self.decoder(unpatchify_output, t, y)
    # decoder --> image
    return decoder_output

class UShapedNetwork(nn.Module):
  def __init__(
      self,
      input_size=32,
      in_channels=4,
      hidden_size=1152,
      num_heads=16,
      mlp_ratio=4.0,
      dropout=0.1,
      learn_sigma=True,

      L1: int = 2,
      L2: int = 2,
      L3: int = 2,
      L4: int = 2
    ):
    super().__init__()
    d_model_2 = hidden_size*2

    self.d_ff = hidden_size * mlp_ratio
    self.in_channels = in_channels
    self.out_channels = in_channels * 2 if learn_sigma else in_channels

    self.diffit_res_block_group_1 = ResBlockGroup(num_heads, dropout, self.d_ff, L1, in_channels=hidden_size, out_channels=hidden_size, img_size=input_size)
    self.diffit_res_block_group_2 = ResBlockGroup(num_heads, dropout, self.d_ff, L2, in_channels=d_model_2, out_channels=d_model_2, img_size=input_size//2)
    self.diffit_res_block_group_3 = ResBlockGroup(num_heads, dropout, self.d_ff, L3, in_channels=d_model_2, out_channels=d_model_2, img_size=input_size//4)

    self.downsample_1 = Downsample(in_channels=hidden_size, out_channels=d_model_2)
    self.downsample_2 = Downsample(in_channels=d_model_2, out_channels=d_model_2)

    self.upsample_1 = Upsample(in_channels=d_model_2, out_channels=d_model_2)
    self.upsample_2 = Upsample(in_channels=d_model_2, out_channels=hidden_size)

    self.tokenizer = Tokenizer(out_channels=hidden_size, in_channels=self.in_channels)
    self.head = Head(in_channels=hidden_size, out_channels=self.out_channels)

  def uShape(self, xs, t):
    output_downsample_1 = self.downsample_1(self.diffit_res_block_group_1(xs, t))
    output_downsample_2 = self.downsample_2(self.diffit_res_block_group_2(output_downsample_1, t))
    uLeft = output_downsample_2

    uCenter = self.diffit_res_block_group_3(uLeft, t)

    input_upsample_1 = uCenter + uLeft
    input_upsample_2 = self.diffit_res_block_group_2(self.upsample_1(input_upsample_1), t) + output_downsample_1
    uRight = self.diffit_res_block_group_1(self.upsample_2(input_upsample_2), t)

    return uRight

  def forward(self, x, t):
    decoder_output = self.head(self.uShape(self.tokenizer(x), t))
    return decoder_output


#################################################################################
#                                   DiffiT Configs                                  #
#################################################################################

def DiffiT_XL_2(**kwargs):
    return DiffiT(depth=30, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

def DiffiT_XL_4(**kwargs):
    return DiffiT(depth=30, hidden_size=1152, patch_size=4, num_heads=16, **kwargs)

def DiffiT_XL_8(**kwargs):
    return DiffiT(depth=30, hidden_size=1152, patch_size=8, num_heads=16, **kwargs)

def DiffiT_L_2(**kwargs):
    return DiffiT(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)

def DiffiT_L_4(**kwargs):
    return DiffiT(depth=24, hidden_size=1024, patch_size=4, num_heads=16, **kwargs)

def DiffiT_L_8(**kwargs):
    return DiffiT(depth=24, hidden_size=1024, patch_size=8, num_heads=16, **kwargs)

def DiffiT_B_2(**kwargs):
    return DiffiT(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)

def DiffiT_B_4(**kwargs):
    return DiffiT(depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs)

def DiffiT_B_8(**kwargs):
    return DiffiT(depth=12, hidden_size=768, patch_size=8, num_heads=12, **kwargs)

def DiffiT_S_2(**kwargs):
    return DiffiT(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)

def DiffiT_S_4(**kwargs):
    return DiffiT(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)

def DiffiT_S_8(**kwargs):
    return DiffiT(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)


DiffiT_models = {
    'DiffiT-XL/2': DiffiT_XL_2,  'DiffiT-XL/4': DiffiT_XL_4,  'DiffiT-XL/8': DiffiT_XL_8,
    'DiffiT-L/2':  DiffiT_L_2,   'DiffiT-L/4':  DiffiT_L_4,   'DiffiT-L/8':  DiffiT_L_8,
    'DiffiT-B/2':  DiffiT_B_2,   'DiffiT-B/4':  DiffiT_B_4,   'DiffiT-B/8':  DiffiT_B_8,
    'DiffiT-S/2':  DiffiT_S_2,   'DiffiT-S/4':  DiffiT_S_4,   'DiffiT-S/8':  DiffiT_S_8,
}
