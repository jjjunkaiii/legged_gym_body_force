from isaacgym import gymtorch, gymapi, gymutil
import math
import numpy as np

class WireframeArrowGeometry(gymutil.LineGeometry):

    def __init__(self, radius=0.02, height=0.1, num_segments=8, length=1.0, pose=None, color=None):
        if color is None:
            color = (1, 0, 0)

        num_lines = 2 * num_segments + 1

        verts = np.empty((num_lines, 2), gymapi.Vec3.dtype)
        colors = np.empty(num_lines, gymapi.Vec3.dtype)
        idx = 0

        # Apex of the cone
        apex = (0.0, 0.0, height+length)  # Convert to tuple

        step = 2 * math.pi / num_segments

        for i in range(num_segments):
            theta1 = i * step
            theta2 = (i + 1) % num_segments * step

            # First point on the base
            x1 = radius * math.cos(theta1)
            y1 = radius * math.sin(theta1)
            z1 = length

            # Second point on the base
            x2 = radius * math.cos(theta2)
            y2 = radius * math.sin(theta2)
            z2 =length

            # Line from the apex to the first point on the base
            verts[idx][0] = apex  # Apex as a tuple
            verts[idx][1] = (x1, y1, z1)
            colors[idx] = color

            idx += 1

            # Line between two consecutive points on the base
            verts[idx][0] = (x1, y1, z1)
            verts[idx][1] = (x2, y2, z2)
            colors[idx] = color

            idx += 1
        # Main line
        verts[idx][0] = apex
        verts[idx][1] = (0, 0, 0)
        colors[idx] = color

        if pose is None:
            self.verts = verts
        else:
            self.verts = pose.transform_points(verts)

        self._colors = colors

    def vertices(self):
        return self.verts

    def colors(self):
        return self._colors