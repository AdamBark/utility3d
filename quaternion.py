# Copyright 2013,2015,2021 Adam Bark

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# utility3d/quaternion.py an implementation of quaternions for 3d
# rotation also generates rotation matrices to be used in an OpenGL
# vertex shader.

import numpy


class Quaternion(object):
    def __init__(self, w, x, y, z):
        self.w = float(w)
        self.v = numpy.array((x, y, z), dtype=float)

    def __neg__(self):
        """Return the inverse of the quaternion."""
        return Quaternion(self.w, *-self.v)

    def __imul__(self, q):
        self.w = q.w * self.w - numpy.dot(q.v, self.v)
        self.v = q.v*self.w + self.v*q.w + numpy.cross(q.v, self.v)
        return self

    def __mul__(self, q):
        return Quaternion(
            q.w * self.w - numpy.dot(q.v, self.v),
            *(q.v*self.w + self.v*q.w + numpy.cross(q.v, self.v))
            )

    def __repr__(self):
        return "Quaternion({}, ({}, {}, {}))".format(self.w, self.v[0],
                                                     self.v[1], self.v[2])

    def normalise(self):
        mod_q = numpy.sqrt(self.w**2 + sum(self.v**2))
        self.w /= mod_q
        self.v /= mod_q

    def rotate_vector(self, vec):
        """Return a vector equivalent to vec rotated by this quaternion."""
        return ((self * vec * -self).v)

    def pointing_vector(self, origin=(0.0, 0.0, -1.0)):
        """Return the vector about which roll occurs.

        If the angle of rotation is 0 then the real part, w,
        is 1 so return origin rather than NaNs.

        """
        if self.w != 1.0:
            return (self.v/numpy.sqrt(1 - self.w**2))
        else:
            return origin

    def matrix(self):
        a = self.w
        b, c, d = self.v
        return numpy.matrix((
            (a**2+b**2-c**2-d**2, 2*b*c - 2*a*d,       2*b*d + 2*a*c,       0),
            (2*b*c + 2*a*d,       a**2-b**2+c**2-d**2, 2*c*d - 2*a*b,       0),
            (2*b*d - 2*a*c,       2*c*d + 2*a*b,       a**2-b**2-c**2+d**2, 0),
            (0,                   0,                   0,                   1)
        ))
