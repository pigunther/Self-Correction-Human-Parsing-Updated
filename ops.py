import mxnet as mx
from mxnet import gluon
from torch.nn import functional as F
import torch
from torchvision.ops import roi_align

class AdaIN(torch.nn.Module):
    def __init__(self, channels_in, channels_out, norm=True):
        super(AdaIN, self).__init__()
        self.channels_in = channels_in
        self.channels_out = channels_out
        self.norm = norm

        self.affine_scale = torch.nn.Linear(channels_in, channels_out, bias=True)
        self.affine_bias = torch.nn.Linear(channels_in, channels_out, bias=True)

    def forward(self, x, w):
        ys = self.affine_scale(w)
        yb = self.affine_bias(w)
        ys = torch.unsqueeze(ys, -1)
        yb = torch.unsqueeze(yb, -1)

        xm = torch.reshape(x, shape=(x.shape[0], x.shape[1], -1))  # (N,C,H,W) --> (N,C,K)
        if self.norm:
            xm_mean = torch.mean(xm, dim=2, keepdims=True)
            xm_centered = xm - xm_mean
            xm_std_rev = torch.rsqrt(torch.mean(torch.mul(xm_centered, xm_centered), dim=2, keepdims=True))  # 1 / std (rsqrt, not sqrt)
            xm_norm = xm_centered - xm_std_rev
        else:
            xm_norm = xm
        xm_scaled = xm_norm*ys + yb
        return torch.reshape(xm_scaled, x.shape)


class AppendCoordFeatures(torch.nn.Module):
    def __init__(self, norm_radius, append_dist=True, spatial_scale=1.0):
        super(AppendCoordFeatures, self).__init__()
        self.xs = None
        self.spatial_scale = spatial_scale
        self.norm_radius = norm_radius
        self.append_dist = append_dist

    def _ctx_kwarg(self, x):
        if isinstance(x, mx.nd.NDArray):
            return {"ctx": x.context}
        return {}

    def get_coord_features(self, points, rows, cols, batch_size,
                           **ctx_kwarg):
        row_array = torch.arange(start=0, end=rows, step=1, **ctx_kwarg)
        col_array = torch.arange(start=0, end=cols, step=1, **ctx_kwarg)
        coord_rows = torch.reshape(row_array, (1, 1, rows, 1)).repeat(1,1,1,cols)
        coord_cols = torch.reshape(col_array, (1, 1, 1, cols)).repeat(1,1, rows, 1)
        coord_rows = coord_rows.repeat(batch_size, 1,1,1)
        coord_cols = coord_cols.repeat(batch_size, 1,1,1)


        coords = torch.cat((coord_rows, coord_cols), dim=1).float().cuda()

        add_xy = (points * self.spatial_scale).unsqueeze(-1)
        add_xy = add_xy.repeat(1,1, rows*cols)
        add_xy = torch.reshape(add_xy, shape=(add_xy.shape[0], add_xy.shape[1], rows, cols)).float()
        coords = (coords - add_xy) / (self.norm_radius * self.spatial_scale)
        if self.append_dist:
            dist = torch.sqrt(torch.sum(torch.mul(coords, coords), dim=1, keepdims=True))
            coord_features = torch.cat((coords, dist), dim=1)
        else:
            coord_features = coords

        coord_features = torch.clamp(coord_features, -1, 1)
        return coord_features

    def forward(self, x, coords):
        self.xs = x.shape

        batch_size, rows, cols = self.xs[0], self.xs[2], self.xs[3]
        coord_features = self.get_coord_features(coords, rows, cols, batch_size,
                                                 **self._ctx_kwarg(x))
        return torch.cat((coord_features.float(), x), dim=1)


class ExtractQueryFeatures(torch.nn.Module):
    def __init__(self, extraction_method='ROIPooling', spatial_scale=1.0, eps=1e-4):
        super(ExtractQueryFeatures, self).__init__()
        self.cshape = None
        self.extraction_method = extraction_method
        self.spatial_scale = spatial_scale
        self.eps = eps

    def _ctx_kwarg(self, x):
        if isinstance(x, mx.nd.NDArray):
            return {"ctx": x.context}
        return {}

    def forward(self, x, coords):
        self.cshape = coords.shape

        batch_size, num_points = self.cshape[0], self.cshape[1]
        ctx_kwarg = self._ctx_kwarg(coords)
        coords = torch.reshape(coords, shape=(-1, 2))
        idx = [i for i in range(coords.shape[1]-1, -1, -1)]
        idx = torch.LongTensor(idx).cuda()
        coords = torch.index_select(coords, 1, idx)

        if self.extraction_method == 'ROIAlign':
            coords = coords - 0.5 / self.spatial_scale
            coords2 = coords
        else:
            coords2 = coords
        rois = torch.cat((coords, coords2), dim=1)
        bi = torch.arange(start=0, end=batch_size, step=1, **ctx_kwarg)
        bi = bi.repeat(num_points)
        bi = torch.reshape(bi, shape=(-1, 1)).type(torch.float32)
        rois = torch.cat((bi.cuda(), rois.float()), dim=1)
        w = roi_align(x, rois, (1, 1), spatial_scale=self.spatial_scale)
        w = torch.reshape(w, shape=(w.shape[0], -1))

        return w


