import torch 
import torch.nn as nn
from torch_geometric.data import Data

from torch_scatter import scatter
from torch_sparse import coalesce

import numpy as np 

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
from skimage.data import coffee, astronaut
from skimage.util import img_as_float
from skimage.transform import resize_local_mean
from skimage.segmentation import mark_boundaries

def img_as_graph(img):
    img_tensor = torch.tensor(img_as_float(img))
    height, width = img_tensor.shape[:2]

    node_index   = torch.arange(width * height).reshape(width, height)
    down_edges   = torch.column_stack((node_index[1:, :].ravel(), node_index[:height-1, :].ravel()))
    right_edges  = torch.column_stack((node_index[:, 1:].ravel(), node_index[:, :width-1].ravel()))
    dright_edges = torch.column_stack([node_index[1:, 1:].ravel(), node_index[:height-1, :width-1].ravel()])
    uright_edges = torch.column_stack([node_index[:height-1, 1:].ravel(), node_index[1:, :width-1].ravel()])
    edge_index = torch.vstack((right_edges, down_edges, dright_edges, uright_edges)).T

    p_width, p_height = 1./width, 1./height
    pos_x, pos_y = np.meshgrid(np.linspace(p_width/2, 1 - p_width, width), 
                               np.linspace(p_height/2, 1 - p_height, height))
    pos_x = torch.tensor(pos_x).flatten()
    pos_y = torch.tensor(pos_y).flatten()
    pos = torch.stack((pos_x, pos_y)).T

    x = img_tensor.flatten(end_dim=-2)
    x = torch.cat((x, pos), dim=1)

    return Data(x=x, edge_index=edge_index, pos=pos)

def compute_edge_weights(g):
    vi, vj = g.x[g.edge_index[0]], g.x[g.edge_index[1]]
    return torch.norm(vi - vj, dim=1)

def get_segments(vi:int, S:torch.Tensor):
   return  S.select(0, vi).coalesce()

def get_nodes_from_segments(S:torch.Tensor, cs:torch.Tensor):
    S_idx, S_val = S.indices(), S.values()
    indices = torch.isin(S_idx[1,:], cs)
    C_idx, C_val = S_idx[:, indices], S_val[indices]
    return torch.sparse_coo_tensor(C_idx, C_val, (n,n)).coalesce()

def clean_up(indices, values, treshold=0.000001):
    idxs = values > treshold
    return torch.sparse_coo_tensor(indices[:,idxs],
                                   values[idxs],
                                   (n,n), is_coalesced=True)

def merge_segments(P, S, segment_sizes, internal_diffs, w):
    Cj = get_nodes_from_segments(S, P.indices()[1])

    CiC = torch.sparse.mm(Cj, P.T)

    Pj = torch.sparse_coo_tensor(torch.stack([P.indices()[1], P.indices()[1]]), 
                                              -P.values(), 
                                              size=(n,n))
    Pj = Pj.coalesce()
    CjC = torch.sparse.mm(Cj, Pj)

    S_new_idx = torch.concat([S.indices(), CiC.indices(), CjC.indices()], dim=-1)
    S_new_val = torch.concat([S.values(), CiC.values(), CjC.values()], dim=-1)
    S_new_idx, S_new_val = coalesce(S_new_idx, S_new_val, n, n)
    S_new = clean_up(S_new_idx, S_new_val)
    
    segment_sizes += torch.sparse.mm(P,segment_sizes.unsqueeze(1)).flatten() + torch.sparse.mm(Pj,segment_sizes.unsqueeze(1)).flatten()

    Pi = scatter(P.values(), P.indices()[0,:], dim_size=n)
    internal_diffs = Pi*w + (1-Pi)*internal_diffs

    return  S_new, segment_sizes, internal_diffs

def tau(v_idx, k=50):
    return k/segment_size[v_idx]

def MInt(Vi:torch.Tensor,
         Vj:torch.Tensor,
         mu:float = 0.025, k:float = 50) -> tuple[torch.Tensor, torch.Tensor]: 
    Vi_idx, Vj_idx = Vi.indices().flatten(), Vj.indices().flatten()
    Vi_val = (internal_diffs[Vi_idx] + tau(Vi_idx, k)) * Vi.values()
    Vj_val = (internal_diffs[Vj_idx] + tau(Vj_idx, k)) * Vj.values()

    Vi_val = Vi_val.unsqueeze(dim=1)
    Vj_val = Vj_val.unsqueeze(dim=0)

    Vi_val, Vj_val = -(Vi_val @ torch.ones_like(Vj_val)), -(torch.ones_like(Vi_val) @ Vj_val)

    #       Smooth Maximum Unit
    smu = -(Vi_val + Vj_val + torch.sqrt((Vi_val - Vj_val)**2 + mu**2))/2 
    return smu 

def merge_probability(Vi:torch.Tensor,
                      Vj:torch.Tensor, 
                      w, 
                      temperature=1, 
                      k=50, 
                      mu=0.025):

    p_val = torch.sigmoid((MInt(Vi, Vj, mu, k) - w) * temperature).flatten()

    p_vi_idx, p_vj_idx = torch.meshgrid(Vi.indices().flatten(), Vj.indices().flatten(), indexing='ij')
    idxs = torch.logical_not(p_vi_idx == p_vj_idx).flatten()
    p_idx = torch.stack([p_vi_idx.flatten(), p_vj_idx.flatten()])[:,idxs]

    return torch.sparse_coo_tensor(p_idx, p_val[idxs], (n,n)).coalesce()

def to_segments(S, width, height):
    Sd = S.to_dense()
    return torch.argmax(Sd, dim=1).numpy().reshape(width, height)

def save_gif(segments,  int_diffs, sizes, img, path):
    fig = plt.figure(layout='constrained')
    gs = GridSpec(3,2,figure=fig)
    axs0 = fig.add_subplot(gs[0:2,:])
    axs0.axis('off')
    axs0.set_title('step 1000')
    im = axs0.imshow(mark_boundaries(img, segments[0]),
                    animated=True)
    hist_range = (0.0, 1)
    hist_bins = 20
    axs1 = fig.add_subplot(gs[2,0])
    axs1.set_ylabel('internal difs')
    axs1.hist(int_diffs[0], range=hist_range, bins=hist_bins, log=True)
    sizes_range = None
    sizes_bins = 20
    axs2 = fig.add_subplot(gs[2,1])
    axs2.set_ylabel('segment sizes')
    axs2.hist(sizes[0], range=sizes_range, bins=sizes_bins, log=True)
    def update(i):
        axs0.set_title('step ' + str(1000*i))
        im.set_data(mark_boundaries(img, segments[i]))
        hist = axs1.hist(int_diffs[i], range=hist_range, bins=hist_bins, log=True)
        sizes_hist = axs2.hist(sizes[i], range=sizes_range, bins=sizes_bins)
        return im, hist, sizes_hist 

    gif = animation.FuncAnimation(fig, update, frames=len(segments),
                                repeat_delay=10)
    gif.save(path)

if __name__ == '__main__':
    import argparse 
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--filename', type=str, default='out.gif')
    parser.add_argument('--to_drive', action='store_true')
    parser.add_argument('-k', '--k', type=float, default=50)
    parser.add_argument('-mu', '--mu', type=float, default=0.025)
    args = parser.parse_args()

    filename = args.filename 
    k, mu = args.k, args.mu

    drive_path = '/run/user/1000/gvfs/google-drive:host=gmail.com,user=j.pelayo.rodrigues/0AONhl6RyXed5Uk9PVA/1Wuff04MHTco0SMzJY4ST4wkcyYNj0cW9'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    img = astronaut()
    img = resize_local_mean(img, (128, 128))
    width, height, _ = img.shape
    g = img_as_graph(img).to(device)

    m = g.edge_index.shape[-1]
    n = g.x.shape[0]

    S_idx = torch.tensor(np.array([np.arange(n), np.arange(n)], dtype=np.int64), device=device)
    S_val  = torch.tensor(np.ones(n), device=device)
    S = torch.sparse_coo_tensor(S_idx, S_val, (n,n)).coalesce()

    edge_weights = compute_edge_weights(g) * 255
    edge_queue = torch.argsort(edge_weights)

    internal_diffs = torch.zeros(n, dtype=torch.float64,  device=device)
    segment_size = torch.ones(n, dtype=torch.float64, device=device)

    segments = []
    int_diffs = []
    sizes = []
    for idx in range(len(edge_queue)):
        edge_idx = edge_queue[idx]
        w = edge_weights[edge_idx]

        vi, vj = g.edge_index[:, edge_idx]
        Vi = get_segments(vi, S)
        Vj = get_segments(vj, S)

        P = merge_probability(Vi, Vj, w, temperature=5, k=k, mu=mu)
        if P.values().isnan().any().item():
            print('nan')
            break

        S, segment_size, internal_diffs = merge_segments(P, S, segment_size, internal_diffs, w)
        if idx % 1000 == 0 and idx > 0:
            segments.append(to_segments(S.to('cpu'), width, height))
            int_diffs.append(internal_diffs.to('cpu'))
            sizes.append(segment_size.to('cpu'))

    if args.to_drive:
        out_path = os.path.join(drive_path, filename)
    else: 
        out_path = filename
    save_gif(segments, int_diffs, sizes, out_path)