import numpy


def perspective(fov_y, aspect, z_near, z_far):
    """Return a perspective projection matrix."""
    z_near=float(z_near); z_far=float(z_far)
    f = 1.0 / numpy.tan(fov_y / 2.0)
    return numpy.matrix([
        [f/aspect, 0, 0, 0],
        [0, f, 0, 0],
        [0, 0, (z_far+z_near)/(z_near-z_far), (2*z_far*z_near)/(z_near-z_far)],
        [0, 0, -1, 0]], dtype=numpy.float32)

def orthographic(left, right, bottom, top, near=-1.0, far=1.0):
    """Return an orthographic projection matrix."""
    left=float(left); right=float(right); bottom=float(bottom); top=float(top)
    near=float(near); far=float(far)
    return numpy.matrix([
        [2.0/(right-left), 0, 0, -(right+left)/(right-left)],
        [0, 2.0/(top-bottom), 0, -(top+bottom)/(top-bottom)],
        [0, 0, -2.0/(far-near), -(far+near)/(far-near)],
        [0, 0, 0, 1]], dtype=numpy.float32)
