import numpy as np
import matplotlib.pyplot as plt

class PoissonSampler:
    def __init__(self, k, r, width, height, radius):
        # Choose up to k points around each reference point as candidates for a new sample point
        self.k = k

        # Minimum distance between samples
        self.r = r

        # width and height
        self.width = width
        self.height = height
        self.radius = radius

        # Cell side length
        self.a = r / np.sqrt(2)

        # Number of cells in the x- and y-directions of the grid
        self.nx, self.ny = int(width / self.a) + 1, int(height / self.a) + 1

        # A list of coordinates in the grid of cells
        self.coords_list = [(ix, iy) for ix in range(self.nx) for iy in range(self.ny)]

        # Initilalize the dictionary of cells: each key is a cell's coordinates, the corresponding value is the index of that cell's point's coordinates in the samples list (or None if the cell is empty).
        self.cells = {coords: None for coords in self.coords_list}

    def get_cell_coords(self, pt):
        """Get the coordinates of the cell that pt = (x,y) falls in."""

        return int(pt[0] // self.a), int(pt[1] // self.a)

    def get_neighbours(self, coords):
        """Return the indexes of points in cells neighbouring cell at coords.

        For the cell at coords = (x,y), return the indexes of points in the cells
        with neighbouring coordinates illustrated below: ie those cells that could
        contain points closer than r.

                                        ooo
                                        ooooo
                                        ooXoo
                                        ooooo
                                        ooo

        """

        dxdy = [(-1,-2),(0,-2),(1,-2),(-2,-1),(-1,-1),(0,-1),(1,-1),(2,-1),
                (-2,0),(-1,0),(1,0),(2,0),(-2,1),(-1,1),(0,1),(1,1),(2,1),
                (-1,2),(0,2),(1,2),(0,0)]
        neighbours = []
        for dx, dy in dxdy:
            neighbour_coords = coords[0] + dx, coords[1] + dy
            if not (0 <= neighbour_coords[0] < self.nx and
                    0 <= neighbour_coords[1] < self.ny):
                # We're off the grid: no neighbours here.
                continue
            neighbour_cell = self.cells[neighbour_coords]
            if neighbour_cell is not None:
                # This cell is occupied: store this index of the contained point.
                neighbours.append(neighbour_cell)
        return neighbours

    def point_valid(self, pt):
        """Is pt a valid point to emit as a sample?

        It must be no closer than r from any other point: check the cells in its
        immediate neighbourhood.

        """

        cell_coords = self.get_cell_coords(pt)
        for idx in self.get_neighbours(cell_coords):
            nearby_pt = self.samples[idx]
            # Squared distance between or candidate point, pt, and this nearby_pt.
            distance2 = (nearby_pt[0]-pt[0])**2 + (nearby_pt[1]-pt[1])**2
            if distance2 < self.r**2:
                # The points are too close, so pt is not a candidate.
                return False
        # All points tested: if we're here, pt is valid
        return True

    def get_point(self, k, refpt):
        """Try to find a candidate point relative to refpt to emit in the sample.

        We draw up to k points from the annulus of inner radius r, outer radius 2r
        around the reference point, refpt. If none of them are suitable (because
        they're too close to existing points in the sample), return False.
        Otherwise, return the pt.

        """
        i = 0
        while i < k:
            rho, theta = np.random.uniform(self.r, 2 * self.r), np.random.uniform(0, 2*np.pi)
            pt = refpt[0] + rho*np.cos(theta), refpt[1] + rho*np.sin(theta)

            cx = pt[0] - self.width / 2
            cy = pt[1] - self.height / 2
            inCircle = (cx * cx + cy * cy) < self.radius * self.radius

            if not (0 <= pt[0] < self.width and 0 <= pt[1] < self.height and inCircle):
                # This point falls outside the domain, so try again.
                continue
            if self.point_valid(pt):
                return pt
            i += 1
        # We failed to find a suitable point in the vicinity of refpt.
        return False

    def poissonSample(self, targetNumbers):

        for tryIdx in range(10):
            # Pick a random point to start with.
            pt = (np.random.uniform(0, self.width), np.random.uniform(0, self.height))
            self.samples = [pt]
            # Our first sample is indexed at 0 in the samples list...
            self.cells[self.get_cell_coords(pt)] = 0
            # ... and it is active, in the sense that we're going to look for more points
            # in its neighbourhood.
            active = [0]

            nsamples = 1
            # As long as there are points in the active list, keep trying to find samples.
            while active:
                # choose a random "reference" point from the active list.
                idx = np.random.choice(active)
                refpt = self.samples[idx]
                # Try to pick a new point relative to the reference point.
                pt = self.get_point(self.k, refpt)
                if pt:
                    # Point pt is valid: add it to the samples list and mark it as active
                    self.samples.append(pt)
                    nsamples += 1
                    active.append(len(self.samples)-1)
                    self.cells[self.get_cell_coords(pt)] = len(self.samples) - 1
                else:
                    # We had to give up looking for valid points near refpt, so remove it
                    # from the list of "active" points.
                    active.remove(idx)

            if nsamples == targetNumbers:
                break
            else:
                self.cells = {coords: None for coords in self.coords_list}

        return self.samples