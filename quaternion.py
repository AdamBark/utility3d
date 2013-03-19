import math
import ctypes
import numpy

class Quaternion(object):
    def __init__(self, w, x, y, z):
        self.w = w
        self.v = numpy.array((complex(x),complex(y),complex(z)))
        
    def __neg__(self):
        # Actually returning the inverse of the quaternion
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
        return "Quaternion(%f, (%f, %f, %f))" % (self.w, self.v[0],
                                                 self.v[1], self.v[2])
    
    def normalise(self):
        mod_q = numpy.sqrt(self.w**2 + sum(self.v**2))
        self.w /= mod_q
        self.v /= mod_q
        
    def rotation_vector(self, vec):
        # Rotates vec around the axis defined by the quaternion
        return numpy.real((self * vec * -self).v)
    
    def pointing_vector(self):
        return self.v/math.sqrt(1-self.w**2)
    
    def matrix(self):
        a = self.w
        b, c, d = self.v
        #return numpy.matrix((
        #    (1 - 2*c**2 - 2*d**2, 2*b*c - 2*a*d,       2*b*d + 2*a*c,       0),
        #    (2*b*c + 2*a*d,       1 - 2*b**2 - 2*d**2, 2*c*d - 2*a*b,       0),
        #    (2*b*d - 2*a*c,       2*c*d + 2*a*b,       1 - 2*b**2 - 2*c**2, 0),
        #    (0,                   0,                   0,                   1)))
        #return numpy.matrix((
        #    (a**2+b**2-c**2-d**2, 2*(b*c-a*d), 2*(b*d+a*c), 0),
        #    (2*(c*b+a*d), 
        return numpy.matrix((
            (a**2+b**2-c**2-d**2, 2*b*c - 2*a*d,       2*b*d + 2*a*c,       0),
            (2*b*c + 2*a*d,       a**2-b**2+c**2-d**2, 2*c*d - 2*a*b,       0),
            (2*b*d - 2*a*c,       2*c*d + 2*a*b,       a**2-b**2-c**2+d**2, 0),
            (0,                   0,                   0,                   1)))