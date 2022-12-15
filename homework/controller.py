import pystk
import math

from .utils import PyTux


def control(aim_point, current_vel):
    """
    Set the Action for the low-level controller
    :param aim_point: Aim point, in screen coordinate frame [-1..1]
    :param current_vel: Current velocity of the kart
    :return: a pystk.Action (set acceleration, brake, steer, drift)
    """
    lr = aim_point[0]
    ud = aim_point[1]
    c = 1000000
    a = 1.5
    doAcc = True
    action = pystk.Action()

    if (abs(lr) > 0.34):
        action.brake = True
        if (current_vel > 20):
            action.drift = True
            doAcc = False
        if (abs(lr) < 0.65):
            action.drift = True
    if (action.drift != True and abs(lr) < 0.05):
        action.nitro = True

    if (doAcc):
        action.acceleration = 1-pow(abs(lr), a)
    else:
        action.acceleration = 0.2*(1-pow(abs(lr), a))

    if (lr > 0):
        action.steer = -pow(c, -lr)+1
    else:
        action.steer = pow(c, lr)-1

    """
    Your code here
    Hint: Use action.acceleration (0..1) to change the velocity. Try targeting a target_velocity (e.g. 20).
    Hint: Use action.brake to True/False to brake (optionally)
    Hint: Use action.steer to turn the kart towards the aim_point, clip the steer angle to -1..1
    Hint: You may want to use action.drift=True for wide turns (it will turn faster)
    """

    return action


def test_controller(pytux, track, verbose=False):
    import numpy as np

    track = [track] if isinstance(track, str) else track
    total_frames = 0
    for t in track: 
        steps, how_far = pytux.rollout(
            t, control, max_frames=1000, verbose=verbose)
        total_frames += steps
        print(steps, t, how_far)
    avg_frames = total_frames / len(track)
    print(total_frames, avg_frames)


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('track', nargs='+')
    parser.add_argument('-v', '--verbose', action='store_true')

    pytux = PyTux()
    test_controller(pytux, **vars(parser.parse_args()))
    pytux.close()
