import numpy as np

def normalize(vec):
    return vec / (np.linalg.norm(vec, axis=-1, keepdims=True) + 1e-9)


def poses_avg(poses):
    center = poses[:, :3, 3].mean(0)
    forward = poses[:, :3, 2].sum(0)
    up = poses[:, :3, 1].sum(0)
    c2w = view_matrix(forward, up, center)
    return c2w


""" All following opencv convenrtion
    < opencv / colmap convention, standard pinhole camera >
    the camera is facing [+z] direction, x right, y downwards
                z
               ↗
              /
             /
            o------> x
            |
            |
            |
            ↓ 
            y
"""


def look_at(
    cam_location: np.ndarray, 
    point: np.ndarray, 
    up=np.array([0., -1., 0.])          # openCV convention
    # up=np.array([0., 1., 0.])         # openGL convention
    ):
    # Cam points in positive z direction
    forward = normalize(point - cam_location)     # openCV convention
    # forward = normalize(cam_location - point)   # openGL convention
    return view_matrix(forward, up, cam_location)


def view_matrix(
    forward: np.ndarray, 
    up: np.ndarray,
    cam_location: np.ndarray):
    rot_z = normalize(forward)
    rot_x = normalize(np.cross(up, rot_z))
    rot_y = normalize(np.cross(rot_z, rot_x))
    mat = np.stack((rot_x, rot_y, rot_z, cam_location), axis=-1)
    hom_vec = np.array([[0., 0., 0., 1.]])
    if len(mat.shape) > 2:
        hom_vec = np.tile(hom_vec, [mat.shape[0], 1, 1])
    mat = np.concatenate((mat, hom_vec), axis=-2)
    return mat


def c2w_track_spiral(c2w, up_vec, rads, focus: float, zrate: float, rots: int, N: int, zdelta: float = 0.):
    # TODO: support zdelta
    """generate camera to world matrices of spiral track, looking at the same point [0,0,focus]

    Args:
        c2w ([4,4] or [3,4]):   camera to world matrix (of the spiral center, with average rotation and average translation)
        up_vec ([3,]):          vector pointing up
        rads ([3,]):            radius of x,y,z direction, of the spiral track
        # zdelta ([float]):       total delta z that is allowed to change 
        focus (float):          a focus value (to be looked at) (in camera coordinates)
        zrate ([float]):        a factor multiplied to z's angle
        rots ([int]):           number of rounds to rotate
        N ([int]):              number of total views
    """

    c2w_tracks = []
    rads = np.array(list(rads) + [1.])
    
    # focus_in_cam = np.array([0, 0, -focus, 1.])   # openGL convention
    focus_in_cam = np.array([0, 0, focus, 1.])      # openCV convention
    focus_in_world = np.dot(c2w[:3, :4], focus_in_cam)

    for theta in np.linspace(0., 2. * np.pi * rots, N+1)[:-1]:
        cam_location = np.dot(
            c2w[:3, :4], 
            # np.array([np.cos(theta), -np.sin(theta), -np.sin(theta*zrate), 1.]) * rads    # openGL convention
            np.array([np.cos(theta), np.sin(theta), np.sin(theta*zrate), 1.]) * rads        # openCV convention
        )
        c2w_i = look_at(cam_location, focus_in_world, up=up_vec)
        c2w_tracks.append(c2w_i)
    return c2w_tracks