"""Full ANN Dreamer components adapted to the shared trainer's leading axis.

Network layers are from NM512/dreamerv3-torch (MIT); there are no spiking
neurons in this path. The singleton axis is an interface axis, not SNN time.
"""
from .vendor import ann_networks as ann


class Encoder(ann.MultiEncoder):
    def forward(self, obs):
        return super().forward(obs).unsqueeze(0)


class Decoder(ann.MultiDecoder):
    def forward(self, features):
        if features.shape[0] != 1:
            raise ValueError("ANN features require one interface axis")
        return super().forward(features[0])


class Head(ann.MLP):
    def forward(self, features, dtype=None):
        if features.shape[0] != 1:
            raise ValueError("ANN features require one interface axis")
        return super().forward(features[0], dtype)


def encoder(c, shapes):
    return Encoder(shapes, mlp_keys="$^", cnn_keys="image", act="SiLU",
                   norm=True, cnn_depth=c.cnn_depth, kernel_size=4, minres=4,
                   mlp_layers=2, mlp_units=c.hidden, symlog_inputs=True)


def decoder(c, features, shapes):
    return Decoder(features, shapes, mlp_keys="$^", cnn_keys="image", act="SiLU",
                   norm=True, cnn_depth=c.cnn_depth, kernel_size=4, minres=4,
                   mlp_layers=2, mlp_units=c.hidden, cnn_sigmoid=False,
                   image_dist="mse", vector_dist="symlog_mse", outscale=1.0)


def head(c, features, shape, layers=2, dist="symlog_disc", scale=1.0):
    return Head(features, shape, layers, c.hidden, act="SiLU", norm=True,
                dist=dist, outscale=scale, device=c.device)


def actor(c, features, actions):
    return Head(features, (actions,), 2, c.hidden, act="SiLU", norm=True,
                dist="normal", std="learned", min_std=0.1, max_std=1.0,
                outscale=1.0, device=c.device)
