#drive_data2_dis10_10000 no cocoa
# 817 cocoa_temple 0.9982387576333077
# 413 lighthouse 0.9980398262738198
# 1000 zengarden 1.9451516507180466
# 501 hacienda 0.9994352372406468
# 668 snowtuxpeak 0.9992819577826596
# 621 cornfield_crossing 0.9980388028245876
# 630 scotland 0.9981931027183482
# 4650 664.2857142857143
import torch
import torch.nn.functional as F
import pdb


def spatial_argmax(logit):
    """
    Compute the soft-argmax of a heatmap
    :param logit: A tensor of size BS x H x W
    :return: A tensor of size BS x 2 the soft-argmax in normalized coordinates (-1 .. 1)
    """

    weights = F.softmax(logit.view(logit.size(0), -1), dim=-1).view_as(logit)

    height = weights.size(1)
    width = weights.size(2)
    w1 = weights[:, 0:height-1, 0:width-1]
    w2 = weights[:, 1:height, 1:width]

    x1 = (w1.sum(1) * torch.linspace(-1, 1,
          w1.size(2)).to(logit.device)[None]).sum(1)
    y1 = (w1.sum(2) * torch.linspace(-1, 1,
          w1.size(1)).to(logit.device)[None]).sum(1)
    x2 = (w2.sum(1) * torch.linspace(-1, 1,
          w2.size(2)).to(logit.device)[None]).sum(1)
    y2 = (w2.sum(2) * torch.linspace(-1, 1,
          w2.size(1)).to(logit.device)[None]).sum(1)
    # c1 = weights.sum(1)
    # c2 = torch.linspace(-1, 1, logit.size(2)).to(logit.device)[None]
    # x = c1*c2
    # pdb.set_trace()

    a = torch.stack((x1, y1, x2, y2), 1)

    return a


class Planner(torch.nn.Module):
    def __init__(self, channels=[16, 32, 64, 64]):

        super().__init__()

        def conv_block(c, h): return [torch.nn.Conv2d(
            h, c, 5, 2, 2), torch.nn.ReLU(True)]

        h, _conv = 3, []
        for c in channels:
            _conv += conv_block(c, h)
            h = c

        self._conv = torch.nn.Sequential(*_conv, torch.nn.Conv2d(h, 1, 1))
        # self.classifier = torch.nn.Linear(h, 2)
        # self.classifier = torch.nn.Conv2d(h, 1, 1)

    def forward(self, img):
        """
        Your code here
        Predict the aim point in image coordinate, given the supertuxkart image
        @img: (B,3,96,128)
        return (B,2)
        """
        x = self._conv(img)
        # print(img.shape)
        # print(x.shape)
        # print(spatial_argmax(x[:, 0]).shape)
        return spatial_argmax(x[:, 0])
        # return self.classifier(x.mean(dim=[-2, -1]))


def save_model(model):
    from torch import save
    from os import path
    if isinstance(model, Planner):
        return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'planner.th'))
    raise ValueError("model type '%s' not supported!" % str(type(model)))


def load_model():
    from torch import load
    from os import path
    r = Planner()
    r.load_state_dict(load(path.join(path.dirname(
        path.abspath(__file__)), 'planner.th'), map_location='cpu'))
    return r


def test_planner(pytux, track, verbose=False):
    from .controller import control

    track = [track] if isinstance(track, str) else track
    planner = load_model().eval()
    total_frames = 0
    for t in track:
        steps, how_far = pytux.rollout(
            t, control, planner, max_frames=1000, verbose=verbose)
        total_frames += steps
        print(steps, t, how_far)
    avg_frames = total_frames / len(track)
    print(total_frames, avg_frames)


if __name__ == '__main__':
    from .utils import PyTux
    from argparse import ArgumentParser

    parser = ArgumentParser("Test the planner")
    parser.add_argument('track', nargs='+')
    parser.add_argument('-v', '--verbose', action='store_true')

    pytux = PyTux()
    test_planner(pytux, **vars(parser.parse_args()))
    pytux.close()

    parser = ArgumentParser("Test the planner")
    parser.add_argument('track', nargs='+')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()
    test_planner(args)
