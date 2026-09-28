"""The 3D warehouse, built from env.get_state() with NVIDIA's Simple_Warehouse assets.

Import only after SimulationApp exists (this module imports omni/pxr). Every size and
offset below was measured by loading the asset and reading its bounds with
isaacsim.core.utils.bounds.compute_aabb (metres, Z up); nothing here is guessed.
Swapping a model means changing its path and, if its size differs, the numbers next to it.
"""
import math

import numpy as np
from isaacsim.core.api.materials import PreviewSurface
from isaacsim.core.prims import XFormPrim
from isaacsim.core.utils import prims, stage
from isaacsim.storage.native import get_assets_root_path
from pxr import Gf, UsdGeom, UsdLux, UsdShade

# One rack bay (4 m beam) or one forklift (3.5 m long) per cell, at real scale.
CELL = 4.0

PROPS = "/Isaac/Environments/Simple_Warehouse/Props/"
FORKLIFT = "/Isaac/Props/Forklift/forklift.usd"        # 1.2 x 3.5 x 2.2, forks on its -y end
RACK_FRAME = PROPS + "SM_RackFrame_03.usd"             # upright 0.13 x 1.0 x 3.0, origin at base
RACK_SHELF = PROPS + "SM_RackShelf_01.usd"             # level 4.0 x 1.08, top surface at +0.025
PALLET = PROPS + "SM_PaletteA_01.usd"                  # 1.2 x 1.0 x 0.21, origin at base
BOX = PROPS + "SM_CardBoxA_01.usd"                     # 0.7 x 0.5 x 0.5, origin at base
DOCK_DECAL = PROPS + "SM_FloorDecal_QuadRed2x1.usd"    # 1.08 x 2.15, origin on its +x edge
FLOOR = PROPS + "SM_floor02.usd"                       # 6 x 6 tile, origin at centre
WALL = PROPS + "SM_WallA_6M.usd"                       # 0.2 (x) x 6 (y) x 3.1, origin on its -x face

SHELF_LEVELS = (0.2, 1.4, 2.6)          # rack level heights; crates rest on the lowest one
CRATE_ON_SHELF_Z = SHELF_LEVELS[0] + 0.025
FORK_LOAD = np.array([0.0, -1.1, 0.1])  # pallet on the forks, in forklift coordinates
DOCK_SIZE = 3.2                         # dock pad drawn inside its 4 m cell
MARGIN = 2.0                            # floor/walls this far beyond the outer cell edges

ROBOT_COLORS = [("orange", (1.0, 0.45, 0.0)), ("blue", (0.1, 0.4, 1.0)), ("green", (0.1, 0.8, 0.2)),
                ("purple", (0.6, 0.2, 0.9)), ("red", (0.9, 0.1, 0.1)), ("yellow", (0.95, 0.85, 0.1)),
                ("cyan", (0.1, 0.85, 0.9)), ("pink", (1.0, 0.4, 0.7)), ("white", (0.95, 0.95, 0.95)),
                ("black", (0.1, 0.1, 0.1))]


def grid_to_world(x, y):
    """Cell centre in metres. Grid y grows downward, world Y grows up the screen, so the 3D
    map reads the same way as the ASCII render when viewed from above."""
    return np.array([x * CELL, -y * CELL, 0.0])


def yaw_quat(yaw):
    """Rotation about Z as the (w, x, y, z) quaternion Isaac expects."""
    return np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])


class WarehouseScene:
    def __init__(self, state):
        self.grid = state["grid"]
        self.root = get_assets_root_path()
        self.n_robots = len(state["robots"])
        width, height = state["width"], state["height"]
        # Every crate that can exist sits on a shelf, on a dock, or on the forks of a robot
        # whose shelf/dock stays reserved; so shelves + docks bounds the pool.
        self.pool_size = sum(row.count("S") + row.count("D") for row in self.grid)

        self._lights()
        lo = np.array([-CELL / 2 - MARGIN, -(height - 0.5) * CELL - MARGIN])
        hi = np.array([(width - 0.5) * CELL + MARGIN, CELL / 2 + MARGIN])
        self._floor(lo, hi)
        self._walls(lo, hi)
        for y, row in enumerate(self.grid):
            for x, c in enumerate(row):
                if c == "S":
                    self._rack(f"/World/Racks/rack_{x}_{y}", grid_to_world(x, y), self.rack_yaw(x, y))
                elif c == "D":
                    self._dock(f"/World/Docks/dock_{x}_{y}", grid_to_world(x, y))
        for i in range(self.n_robots):
            self._forklift(i)
        for k in range(self.pool_size):
            self._crate(k)

        self.robots = XFormPrim([f"/World/Robots/robot_{i}" for i in range(self.n_robots)])
        self.crates = XFormPrim([f"/World/Crates/crate_{k}" for k in range(self.pool_size)])
        self.crates.set_visibilities([False] * self.pool_size)
        self.slot_of = {}   # crate id -> pool slot; slots are freed when a crate leaves

        centre = (lo + hi) / 2
        self.overview = (np.array([centre[0], lo[1] - 0.6 * (hi[1] - lo[1]), 0.9 * max(hi - lo)]),
                         np.array([centre[0], centre[1], 0.0]))

    # ---------------------------------------------------------------- per frame

    def show(self, robot_xy, robot_yaw, crates):
        """robot_xy: world (x, y) per robot. crates: list of (id, pose or None to hide), where
        pose is (position, yaw)."""
        positions = [(x, y, 0.0) for x, y in robot_xy]
        self.robots.set_world_poses(positions=np.array(positions),
                                    orientations=np.array([yaw_quat(a) for a in robot_yaw]))
        for cid, pose in crates:
            if pose is None:
                if cid in self.slot_of:
                    slot = self.slot_of.pop(cid)
                    self.crates.set_visibilities([False], indices=[slot])
                continue
            if cid not in self.slot_of:
                free = sorted(set(range(self.pool_size)) - set(self.slot_of.values()))
                if not free:
                    raise RuntimeError(f"crate pool of {self.pool_size} exhausted")
                self.slot_of[cid] = free[0]
                self.crates.set_visibilities([True], indices=[free[0]])
            position, yaw = pose
            self.crates.set_world_poses(positions=np.array([position]),
                                        orientations=np.array([yaw_quat(yaw)]),
                                        indices=[self.slot_of[cid]])

    def carried_position(self, robot_xy, robot_yaw):
        """Where a pallet sits on a robot's forks, in world coordinates."""
        c, s = math.cos(robot_yaw), math.sin(robot_yaw)
        fx, fy, fz = FORK_LOAD
        return np.array([robot_xy[0] + c * fx - s * fy, robot_xy[1] + s * fx + c * fy, fz])

    def resting_position(self, x, y):
        """A crate sitting in cell (x, y): on the lowest rack level, or on the dock floor."""
        z = CRATE_ON_SHELF_Z if self.grid[y][x] == "S" else 0.0
        return grid_to_world(x, y) + np.array([0.0, 0.0, z])

    def rack_yaw(self, x, y):
        """Racks run along x unless the shelf is part of a north-south row (maze walls)."""
        def shelf(cx, cy):
            return 0 <= cy < len(self.grid) and 0 <= cx < len(self.grid[0]) and self.grid[cy][cx] == "S"
        along_x = shelf(x - 1, y) + shelf(x + 1, y)
        along_y = shelf(x, y - 1) + shelf(x, y + 1)
        return math.pi / 2 if along_y > along_x else 0.0

    def resting_yaw(self, x, y):
        """A crate lines up with its rack; on a dock it keeps yaw 0."""
        return self.rack_yaw(x, y) if self.grid[y][x] == "S" else 0.0

    # ---------------------------------------------------------------- building

    def _add(self, path, asset, position, yaw=0.0, scale=(1.0, 1.0, 1.0)):
        prims.create_prim(path, usd_path=self.root + asset, position=np.asarray(position, dtype=float),
                          orientation=yaw_quat(yaw), scale=np.array(scale, dtype=float))

    def _lights(self):
        st = stage.get_current_stage()
        sun = UsdLux.DistantLight.Define(st, "/World/Lights/sun")
        sun.CreateIntensityAttr(2500)
        UsdGeom.Xformable(sun).AddRotateXYZOp().Set(Gf.Vec3f(-50, 0, 30))
        UsdLux.DomeLight.Define(st, "/World/Lights/sky").CreateIntensityAttr(600)

    def _floor(self, lo, hi):
        # Tiles are stretched slightly so they cover the area exactly, whatever the map size.
        size = hi - lo
        n = np.ceil(size / 6.0).astype(int)
        step = size / n
        for i in range(n[0]):
            for j in range(n[1]):
                centre = lo + step * (np.array([i, j]) + 0.5)
                self._add(f"/World/Floor/tile_{i}_{j}", FLOOR, (*centre, 0.0),
                          scale=(step[0] / 6.0, step[1] / 6.0, 1.0))

    def _walls(self, lo, hi):
        for side, (start, end, fixed, along_y) in enumerate([
            (lo[1], hi[1], lo[0] - 0.2, True),    # west: the panel's thickness grows toward +x
            (lo[1], hi[1], hi[0], True),          # east
            (lo[0], hi[0], lo[1] - 0.2, False),   # south
            (lo[0], hi[0], hi[1], False),         # north
        ]):
            n = math.ceil((end - start) / 6.0)
            length = (end - start) / n
            for k in range(n):
                mid = start + length * (k + 0.5)
                # A panel runs along its local y; turning it 90 degrees makes it run along x.
                position = (fixed, mid, 0.0) if along_y else (mid, fixed, 0.0)
                self._add(f"/World/Walls/wall_{side}_{k}", WALL, position,
                          yaw=0.0 if along_y else math.pi / 2, scale=(1.0, length / 6.0, 1.0))

    def _rack(self, path, centre, yaw):
        # One bay: an upright at each end of the 4 m cell, with shelf levels between them.
        c, s = math.cos(yaw), math.sin(yaw)
        for side in (-1, 1):
            offset = side * (CELL / 2 - 0.07)
            self._add(f"{path}/upright_{'lr'[side > 0]}", RACK_FRAME, centre + [c * offset, s * offset, 0], yaw)
        for k, z in enumerate(SHELF_LEVELS):
            self._add(f"{path}/level_{k}", RACK_SHELF, centre + [0, 0, z - 0.025], yaw)

    def _dock(self, path, centre):
        sx, sy = DOCK_SIZE / 1.077, DOCK_SIZE / 2.154
        # The decal's origin is on its +x edge, so shift it half its width to centre it.
        self._add(path, DOCK_DECAL, centre + [DOCK_SIZE / 2, 0, 0.01], scale=(sx, sy, 1.0))

    def _forklift(self, i):
        path = f"/World/Robots/robot_{i}"
        prims.create_prim(path)
        self._add(f"{path}/body", FORKLIFT, (0.0, 0.0, 0.0))
        # A coloured plate on the roof tells the robots apart, matching the stats panel.
        # The RTX renderer ignores plain displayColor, so the plate gets a real material.
        plate = UsdGeom.Cube.Define(stage.get_current_stage(), f"{path}/plate")
        plate.CreateSizeAttr(1.0)
        UsdGeom.XformCommonAPI(plate).SetTranslate(Gf.Vec3d(0.0, 0.6, 2.2))
        UsdGeom.XformCommonAPI(plate).SetScale(Gf.Vec3f(0.9, 1.0, 0.08))
        paint = PreviewSurface(f"/World/Looks/robot_{i}", color=np.array(ROBOT_COLORS[i][1]), roughness=0.4)
        UsdShade.MaterialBindingAPI.Apply(plate.GetPrim()).Bind(paint.material)

    def _crate(self, k):
        # A pallet carrying four boxes: two side by side, two high.
        path = f"/World/Crates/crate_{k}"
        prims.create_prim(path)
        self._add(f"{path}/pallet", PALLET, (0.0, 0.0, 0.0))
        for j, (y, z) in enumerate([(-0.25, 0.211), (0.25, 0.211), (-0.25, 0.711), (0.25, 0.711)]):
            self._add(f"{path}/box_{j}", BOX, (0.0, y, z))
