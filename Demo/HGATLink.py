import dgl
import dgl.function as fn
import dgl.nn.pytorch as dglnn
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch
from utils import to_etype_name
import math
from torch_geometric.utils import softmax
import matplotlib.pyplot as plt
class HGATLinkConv(nn.Module):
    def __init__(
            self, in_feats,k_feats, out_feats,heads,d_k,weight=True,weight_k=True,relation_att=True, device=None, dropout_rate=0.0
    ):
        super(HGATLinkConv, self).__init__()
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._k_feats = k_feats
        self.device = device
        self.dropout = nn.Dropout(dropout_rate)
        self.n_heads = heads
        self.d_k = d_k
        if weight:
            self.weight = nn.Parameter(th.Tensor(in_feats, out_feats))
        else:
            self.register_parameter("weight", None)
        if weight_k:
            self.weight_k = nn.Parameter(th.Tensor(k_feats, out_feats))
        else:
            self.register_parameter("weight_k", None)

        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize parameters."""
        if self.weight is not None:
            nn.init.xavier_uniform_(self.weight)
            nn.init.xavier_uniform_(self.weight_k)

    def forward(self, graph, feat,weight=None,weight_k=None):
       
            
        with graph.local_scope():
            if isinstance(feat, tuple):
                feat, _ = feat  

            cj = graph.srcdata["cj"]
            ci = graph.dstdata["ci"]

            if self.device is not None:
                cj = cj.to(self.device)
                ci = ci.to(self.device)
            if weight is not None:
                if self.weight is not None:
                    raise dgl.DGLError(
                        "External weight is provided while at the same time the"
                        " module has defined its own weight parameter. Please"
                        " create the module with flag weight=False."
                    )
            else:
                weight = self.weight
                weight_k = self.weight_k
                
            res = feat
            if weight is not None:
                feat = dot_or_identity(feat, weight, self.device)
                weight_k = dot_or_identity(res, weight_k, self.device)
            #Attention mechanisms
            q_mat = (weight_k * ci).view(-1, self.n_heads, self.d_k)
            
            k_mat = q_mat
            q_mat_norm = F.normalize(q_mat, p=2., dim=-1)
            k_mat_norm = F.normalize(k_mat, p=2., dim=-1)
            alpha = torch.abs(k_mat_norm * q_mat_norm).view(k_mat_norm.size(0), -1)
            Tau = 0.25
            
            feat = F.relu(feat * self.dropout(cj))
            
            graph.srcdata["h"] = feat
            
            graph.update_all(
                fn.copy_u(u="h", out="m"), fn.max(msg="m", out="h")
            )
            
            rst = graph.dstdata["h"]
            attn = F.softmax(alpha/Tau, dim=1)
            #The target genetic profile adjusts itself to update
            out = rst*attn

        return out



class DGLLayer(nn.Module):
    def __init__(
            self,
            rating_vals,
            gene_in_units,  
            cell_in_units,  
            msg_units,
            out_units,
            dropout_rate=0.0,

            device=None,
    ):
        super(DGLLayer, self).__init__()
        self.rating_vals = rating_vals

        self.ufc = nn.Linear(msg_units, out_units)
        self.ifc = nn.Linear(msg_units, out_units)

       
        assert msg_units % len(rating_vals) == 0
        
        msg_units = msg_units // len(rating_vals)

        self.dropout = nn.Dropout(dropout_rate)
        self.W_r = nn.ParameterDict()
        #Steps to Increase Attention Mechanisms Dimensional Consistency on Targets
        self.W_k = nn.ParameterDict()
        subConv = {}
        for rating in rating_vals:
        
            origin_rating = rating
            rating = to_etype_name(rating)
            rev_rating = "rev-%s" % rating
            #Multiple Head Attention Mechanisms
            heads = 4
            d_k =64
            self.W_r = None
            self.W_k = None
            subConv[rating] = HGATLinkConv(
                gene_in_units,
                cell_in_units,
                msg_units,
                heads,
                d_k,
                weight=True,
                weight_k=True,
                relation_att=True,
                device=device,
                dropout_rate=dropout_rate,
            )
            subConv[rev_rating] = HGATLinkConv(
                cell_in_units,
                gene_in_units,
                msg_units,
                heads,
                d_k,
                weight=True,
                weight_k=True,
                relation_att=True,
                device=device,
                dropout_rate=dropout_rate,
            )

        self.conv = dglnn.HeteroGraphConv(subConv, aggregate='stack')
        self.agg_act = nn.ReLU()
        self.out_act = lambda x: x
        self.device = device
        self.reset_parameters()

    def partial_to(self, device):
        
        assert device == self.device
        if device is not None:
            self.ufc.cuda(device)
            if self.share_gene_item_param is False:
                self.ifc.cuda(device)
            self.dropout.cuda(device)

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, graph, ufeat=None, ifeat=None):
       
        in_feats = {"gene": ufeat, "cell": ifeat}

        mod_args = {}
        for i, rating in enumerate(self.rating_vals):
            rating = to_etype_name(rating)
            rev_rating = "rev-%s" % rating
            mod_args[rating] = (
                self.W_r[rating] if self.W_r is not None else None,
            )
            mod_args[rev_rating] = (
                self.W_r[rev_rating] if self.W_r is not None else None,
            )
        
        #1071*15*256  704*15*256
        out_feats = self.conv(graph, in_feats, mod_args=mod_args)

        ufeat = out_feats["gene"]
        ifeat = out_feats["cell"]
       
        ufeat = ufeat.view(ufeat.shape[0], -1)
      
        ifeat = ifeat.view(ifeat.shape[0], -1)

        # fc and non-linear
        ufeat = self.agg_act(ufeat)
        ifeat = self.agg_act(ifeat)
        ufeat = self.dropout(ufeat)
        ifeat = self.dropout(ifeat)
        ufeat = self.ufc(ufeat)
        ifeat = self.ifc(ifeat)
        return self.out_act(ufeat), self.out_act(ifeat)


class Decoder(nn.Module):

    def __init__(self, dropout_rate=0.0):
        super(Decoder, self).__init__()

        self.dropout = nn.Dropout(dropout_rate)
        self.reset_parameters()

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, graph, ufeat, ifeat):

        with graph.local_scope():
            ufeat = self.dropout(ufeat)
            ifeat = self.dropout(ifeat)
            graph.nodes["cell"].data["h"] = ifeat
            graph.nodes["gene"].data["h"] = ufeat
            graph.apply_edges(fn.u_dot_v("h", "h", "sr"))

            return graph.edata['sr']


def dot_or_identity(A, B, device=None):
    # if A is None, treat as identity matrix
    if A is None:
        return B
    elif len(A.shape) == 1:
        if device is None:
            return B[A]
        else:
            return B[A].to(device)
    else:
        return A @ B


class LinearNet(nn.Module):
    def __init__(self, emb_dim=256):
        super(LinearNet, self).__init__()
        self.ac = nn.GELU()
       
        
        
        
        d_model=emb_dim
        num_head=4
        num_layers=2
        
        self.dropout1=torch.nn.Dropout(0.5)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_head, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers) 
        self.decoder_layer =  nn.TransformerDecoderLayer(d_model=d_model, nhead=4, dim_feedforward=emb_dim,dropout=0.5)
        self.transformer_decoder =   nn.TransformerDecoder(self.decoder_layer, num_layers=1)
        self.position_embedding = nn.Embedding(2, emb_dim)
        self.flatten = nn.Flatten()
        self.linear256 = nn.Linear(512, 256)
        self.layernorm256 = nn.LayerNorm(256)
        self.batchnorm256 = nn.BatchNorm1d(256)
        
        self.linear2 = nn.Linear(256, 1)
        
        
        self.linear512 = nn.Linear(1024, 512)
        self.layernorm512  = nn.LayerNorm(512)
        self.batchnorm512  = nn.BatchNorm1d(512)
        self.linear3 = nn.Linear(512, 1)
        
        
        self.actf = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)
        self.pool = nn.AvgPool1d(kernel_size=2, stride=2) 
        #self.reset_parameters()

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        
        x = self.dropout1(x)
          
        out = self.transformer_encoder(x)
        out = self.transformer_decoder(x,out)
        
        out = self.flatten(out) 
               
        out = self.linear256(out)
 
        out = self.dropout(out)
        out = self.batchnorm256(out) 
        out = self.ac(out)
        x = self.linear2(out)
      
        return x
