# Copyright 2013,2021 Adam Bark

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# utility3d/views.py generate a projection matrix for OpenGL

import numpy


def perspective(fov_y, aspect, z_near, z_far):
    """Return a perspective projection matrix."""
    z_near = float(z_near)
    z_far = float(z_far)
    f = 1.0 / numpy.tan(fov_y / 2.0)
    return numpy.matrix([
        [f/aspect, 0, 0, 0],
        [0, f, 0, 0],
        [0, 0, (z_far+z_near)/(z_near-z_far), (2*z_far*z_near)/(z_near-z_far)],
        [0, 0, -1, 0]], dtype=numpy.float32)


def orthographic(left, right, bottom, top, near=-1.0, far=1.0):
    """Return an orthographic projection matrix."""
    return numpy.matrix([
        [2.0/(right-left), 0, 0, -(right+left)/(right-left)],
        [0, 2.0/(top-bottom), 0, -(top+bottom)/(top-bottom)],
        [0, 0, -2.0/(far-near), -(far+near)/(far-near)],
        [0, 0, 0, 1]], dtype=numpy.float32)
