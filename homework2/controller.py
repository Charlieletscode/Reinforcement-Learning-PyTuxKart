import pystk
import math

from .utils import PyTux
import numpy as np


# def control(aim_point1, aim_point2, current_vel):
#     ratio = 1.2
#     # ratio = 0.01 # ap1 / ap2 when dis = 5
#     # ratio = 0.3 # ap1 / ap2 when dis = 7
#     # ratio = 1.5 # ap1 / ap2 when dis = 9
#     w1 = 1 - 1/(ratio+1)
#     w2 = 1/(ratio+1)
#     lr = aim_point1[0]*w1 + aim_point2[0]*w2

#     c = 1000000
#     a = 1.5
#     doAcc = True
#     action = pystk.Action()

#     if (abs(lr) > 0.34):
#         action.brake = True
#         if (current_vel > 20):
#             action.drift = True
#             doAcc = False
#         if (abs(lr) < 0.65):
#             action.drift = True
#     if (action.drift != True and abs(lr) < 0.05):
#         action.nitro = True

#     if (doAcc):
#         action.acceleration = 1-pow(abs(lr), a)
#     else:
#         action.acceleration = 0.2*(1-pow(abs(lr), a))

#     if (lr > 0):
#         action.steer = -pow(c, -lr)+1
#     else:
#         action.steer = pow(c, lr)-1

#     return action

# def control(aim_point1, aim_point2, current_vel):
#     lr = (aim_point1[0]*0.5+aim_point2[0]*0.5)
#     ud = (aim_point1[1]*0.5+aim_point2[1]*0.5)

#     # lr = (aim_point1[0])
#     # lr1 = (aim_point1[1]*7+aim_point2[0]*3) / 10
#     c = 1000000
#     a = 1.5
#     doAcc = True
#     action = pystk.Action()

#     if (abs(lr) > 0.34):
#         action.brake = True
#         if (current_vel > 20):
#             action.drift = True
#             doAcc = False
#         # lr = (aim_point1[0])
#         if (abs(lr) < 0.65):
#             action.drift = True
#         else:
#              action.acceleration = -1
#     # lr = (aim_point1[0])
#     if (action.drift != True and abs(lr) < 0.05):
#         action.nitro = True

#     if (doAcc):
#         action.acceleration = 1-pow(abs(lr), a)
#     else:
#         action.acceleration = 0.2*(1-pow(abs(lr), a))

#     if (lr > 0):
#         action.steer = -pow(c, -lr)+1
#     else:
#         action.steer = pow(c, lr)-1

#     if (abs(aim_point1[0])<0.20):
#         action.acceleration = 1

#     return action

# def control(aim_point1, aim_point2, current_vel):
#     """
#     Set the Action for the low-level controller
#     :param aim_point: Aim point, in screen coordinate frame [-1..1]
#     :param current_vel: Current velocity of the kart
#     :return: a pystk.Action (set acceleration, brake, steer, drift)
#     """
#     ratio = 0.3
#     w1 = 1 - 1/(ratio+1)
#     w2 = 1/(ratio+1)
#     lr = aim_point1[0]*w1 + aim_point2[0]*w2
#     # lr = (aim_point1[0]*0.33+ aim_point2[0]*0.67)
#     # ud = aim_point[1]
#     c = 1000000
#     a = 1.5
#     doAcc = True
#     action = pystk.Action()

#     if (abs(lr) > 0.34):
#         action.brake = True
#         if (np.abs(np.tan((np.abs(aim_point1[0]-aim_point2[0]))/np.abs((aim_point1[1]-aim_point2[1])+0.001))))>10:
#             action.drift=True
#             action.acceleration=1
#         if (current_vel > 20):
#             action.drift = True
#             doAcc = False
#         if (np.abs(lr) < 0.65):
#             action.drift = True
#     if (action.drift != True and abs(lr) < 0.05):
#         action.nitro = True

#     if (doAcc):
#         action.acceleration = 1-pow(np.abs(lr), a)
#     else:
#         action.acceleration = 0.2*(1-pow(np.abs(lr), a))

#     if (lr > 0):
#         action.steer = -pow(c, -lr)+1
#     else:
#         action.steer = pow(c, lr)-1

#     if (abs(aim_point1[0])<0.20):
#         action.acceleration = 1

#     return action

def control(aim_point1, aim_point2, current_vel):
    """
    Set the Action for the low-level controller
    :param aim_point: Aim point, in screen coordinate frame [-1..1]
    :param current_vel: Current velocity of the kart
    :return: a pystk.Action (set acceleration, brake, steer, drift)
    """
    lr = (aim_point1[0]*0.33+ aim_point2[0]*0.67)
    # ud = aim_point[1]
    c = 1000000
    a = 1.5
    doAcc = True
    action = pystk.Action()


    if (abs(lr) > 0.34):
        action.brake = True
        if (np.abs(np.tan((np.abs(aim_point1[0]-aim_point2[0]))/np.abs((aim_point1[1]-aim_point2[1])+0.001))))>10:
            action.drift=True

        if (current_vel > 20):
            action.drift = True
            doAcc = False
        if (np.abs(lr) < 0.65):
            action.drift = True
    if (action.drift != True and abs(lr) < 0.05):
        action.nitro = True

    if (doAcc):
        action.acceleration = 1-pow(np.abs(lr), a)
    else:
        action.acceleration = 0.2*(1-pow(np.abs(lr), a))

    if (lr > 0):
        action.steer = -pow(c, -lr)+1
    else:
        action.steer = pow(c, lr)-1

    if (abs(aim_point1[0])<0.20):
        action.acceleration = 1

    if current_vel<5.8:
        action.acceleration=0.83

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
