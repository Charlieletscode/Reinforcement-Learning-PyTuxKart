#drive_data2_dis10_20000

# 558 cocoa_temple 0.9999951950023979
# 378 zengarden 0.9995076099648883
# 502 hacienda 0.999206485824685
# 503 snowtuxpeak 0.9991522096385653
# 667 cornfield_crossing 0.9984459166435663
# 538 scotland 0.9992243029567811
# 3554 507.7142857142857

# 557 cocoa_temple 0.999785798264789
# 403 lighthouse 0.9988483276992588
# 380 zengarden 1.0
# 500 hacienda 0.998002139966357
# 498 snowtuxpeak 1.0001298096361058
# 623 cornfield_crossing 0.9987226381361818
# 535 scotland 0.9980380184344498
# 3496 499.42857142857144

# 558 cocoa_temple 0.9999735303640864
# 404 lighthouse 0.9991486975981992
# 356 zengarden 0.9981862072882145
# 502 hacienda 0.999899541921627
# 497 snowtuxpeak 0.9991892278294396
# 627 cornfield_crossing 0.9990299776787654
# 538 scotland 0.9988552427859164
# 3482 497.42857142857144

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

    firstcoord = (weights.sum(1) * torch.linspace(-1, 1,
                  logit.size(2)).to(logit.device)[None]).sum(1)
    secondcoord = (weights.sum(2) * torch.linspace(-1, 1,
                   logit.size(1)).to(logit.device)[None]).sum(1)

    return firstcoord, secondcoord

class Planner(torch.nn.Module):
    def __init__(self, channels=[16, 32, 64, 64, 64]):

        super().__init__()
        
        ll = []
        ll.append(torch.nn.BatchNorm2d(num_features=3))
        ll.append(torch.nn.Conv2d(in_channels=3, out_channels=32,
                kernel_size=3, stride=1, padding=1))
        ll.append(torch.nn.ReLU(inplace=True))
        ll.append(torch.nn.Conv2d(in_channels=32, out_channels=32,
                kernel_size=3, stride=1, padding=1))
        ll.append(torch.nn.ReLU(inplace=True))
        ll.append(torch.nn.MaxPool2d(kernel_size=2, stride=2))
        ll.append(torch.nn.BatchNorm2d(num_features=32))
        ll.append(torch.nn.Conv2d(in_channels=32, out_channels=64,
                kernel_size=5, stride=1, padding=2))
        ll.append(torch.nn.ReLU(inplace=True))
        ll.append(torch.nn.Conv2d(in_channels=64, out_channels=64,
                kernel_size=5, stride=1, padding=2))
        ll.append(torch.nn.ReLU(inplace=True))
        ll.append(torch.nn.Conv2d(in_channels=64, out_channels=64,
                kernel_size=5, stride=1, padding=2))
        ll.append(torch.nn.ReLU(inplace=True))
        ll.append(torch.nn.MaxPool2d(kernel_size=2, stride=2))
        ll.append(torch.nn.BatchNorm2d(num_features=64))
        ll.append(torch.nn.Conv2d(in_channels=64, out_channels=64,
                kernel_size=5, stride=1, padding=2))
        ll.append(torch.nn.ReLU(inplace=True))
        ll.append(torch.nn.Conv2d(in_channels=64, out_channels=2, kernel_size=1))

        self._conv = torch.nn.Sequential(*ll)

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
        x1, y1 = spatial_argmax(x[:, 0])
        x2, y2 = spatial_argmax(x[:, 1])
        
        # pdb.set_trace()
        return torch.stack((x1, y1, x2, y2), 1)
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
