import base64
import zlib
import struct
import math

import numpy as np
from scipy import interpolate

from plane import Plane


class Depthmap:
    def __init__(self, dm_b64_string: str):
        self.data = self.parse(dm_b64_string)

        # https://medium.com/@nocomputer/creating-point-clouds-with-google-street-view-185faad9d4ee
        self.num_planes = self.get_int(self.data, 1)
        self.map_width = self.get_int(self.data, 3)
        self.map_height = self.get_int(self.data, 5)
        self.offset = self.data[7]

        self.dimension = self.map_height * self.map_width

        self.indices = self.data[self.offset:self.offset + self.map_height * self.map_width]

        self.planes = []
        for i in range(self.num_planes):
            offset = self.offset + self.dimension + i * 16
            n = [self.get_float(self.data, offset + j * 4) for j in range(4)]
            self.planes.append(Plane(n))

        self.depth_map = np.empty(self.dimension)
        for y in range(self.map_height):
            for x in range(self.map_width):
                x_normalize = (self.map_width - x - 1.0) / (self.map_width - 1.0)
                y_normalize = (self.map_height - y - 1.0) / (self.map_height - 1.0)
                theta = x_normalize * 2 * math.pi + (math.pi / 2)
                phi = y_normalize * math.pi

                v = [math.sin(phi) * math.cos(theta), math.sin(phi) * math.sin(theta), math.cos(phi)]

                plane_idx = self.indices[y * self.map_width + x]

                if plane_idx > 0:
                    plane = self.planes[plane_idx]
                    t = np.abs(plane.d / (v[0] * plane.x + v[1] * plane.y + v[2] * plane.z))
                    self.depth_map[y * self.map_width + (self.map_width - x - 1)] = t
                else:
                    self.depth_map[y * self.map_width + (self.map_width - x - 1)] = 0

    def point_cloud(self, pano_yaw: float, tilt_yaw: float, tilt_pitch: float, lat: float, lon: float, elevation: float):
        fov_h = math.radians(120)
        theta = math.radians(0)
        phi = math.radians(47.06)
        h_thumb = 5632
        w_thumb = 12800

    @staticmethod
    def rotate_x(pitch):
        # picth is radian
        r_x = np.array([[1.0, 0.0, 0.0],
                        [0.0, math.cos(pitch), -1 * math.sin(pitch)],
                        [0.0, math.sin(pitch), math.cos(pitch)]])
        return r_x

    @staticmethod
    def rotate_y(yaw):
        #
        r_y = np.array([[math.cos(yaw), 0.0, math.sin(yaw)],
                        [0.0, 1.0, 0.0],
                        [-1 * math.sin(yaw), 0.0, math.cos(yaw)]])
        return r_y

    @staticmethod
    def rotate_z(roll):
        #
        r_z = np.array([[math.cos(roll), -1 * math.sin(roll), 0.0],
                        [math.sin(roll), math.cos(roll), 0.0],
                        [0.0, 0.0, 1.0]])
        return r_z

    def clip_pano3(self, theta0, phi0, fov_h, fov_v, width, height, img):  # fov < 120
        """
          theta0 is pitch
          phi0 is yaw
          render view at (pitch, yaw) with fov_h by fov_v
          width is the number of horizontal pixels in the view
          """
        # m = np.dot(self.rotate_y(phi0), self.rotate_x(theta0))
        # m = self.rotate_y(phi0).dot(self.rotate_x(theta0-tilt_pitch))
        m = self.rotate_y(phi0).dot(self.rotate_x(theta0))

        img = np.array(img)
        try:
            (base_height, base_width, bands) = img.shape
        except:
            base_height, base_width = img.shape
            bands = 1
        # np.array(dm[dm['depthMap']]).reshape((dm["height"], dm["width"]))

        # height =int(math.floor(width * np.tan(fov_v / 2) / np.tan(fov_h / 2)))
        # width =int(width)

        if bands > 1:
            new_img = np.zeros((height, width, bands), np.float)
            DI = np.ones((height * width, bands), np.int)
        else:
            # img = np.expand_dims(img, 3)
            # new_img = np.zeros((height, width,3), np.uint8)
            new_img = np.zeros((int(height), int(width)), np.float)
            DI = np.ones((int(height * width), 3), np.int)

        # new_img = np.zeros((height, width, 3), np.uint8)
        # DI = np.ones((height * width, 3), np.int)
        trans = np.array([[2. * np.tan(fov_h / 2) / float(width), 0., -np.tan(fov_h / 2)],
                          [0., -2. * np.tan(fov_v / 2) / float(height), np.tan(fov_v / 2)]])

        xx, yy = np.meshgrid(np.arange(width), np.arange(height))

        DI[:, 0] = xx.reshape(height * width)
        DI[:, 1] = yy.reshape(height * width)

        v = np.ones((height * width, 3), np.float)

        v[:, :2] = np.dot(DI, trans.T)
        v = np.dot(v, m.T)

        diag = np.sqrt(v[:, 2] ** 2 + v[:, 0] ** 2)
        theta = np.pi / 2 - np.arctan2(v[:, 1], diag)
        phi = np.arctan2(v[:, 0], v[:, 2]) + np.pi

        ey = np.rint(theta * base_height / np.pi).astype(np.int)
        ex = np.rint(phi * base_width / (2 * np.pi)).astype(np.int)

        ex[ex >= base_width] = base_width - 1
        ey[ey >= base_height] = base_height - 1

        new_img[DI[:, 1], DI[:, 0]] = img[ey, ex]
        return new_img, theta.reshape(height * width), phi.reshape(height * width)

    def constructRawPointsCloud3(self, thumbnail_predicts, fov_h, theta0, phi0, w_thumb, h_thumb, pano_yaw):

        fov_v = math.atan(h_thumb * math.tan(fov_h / 2) / w_thumb) * 2
        # pano_thetas, pano_phis = equiRectangle2pinhole(theta0, phi0, fov_h, fov_v, w_thumb,h_thumb, pano_yaw)
        img, pano_thetas, pano_phis = self.clip_pano3(theta0, phi0 - pano_yaw, fov_h, fov_v, w_thumb, h_thumb,
                                                      thumbnail_predicts)
        grid_col, grid_row = np.meshgrid(np.linspace(-math.pi, math.pi, self.map_width),
                                         np.linspace(math.pi, 0, self.map_height))
        #
        grid_col = grid_col.ravel()
        grid_row = grid_row.ravel()

        grid_points = np.stack((grid_col, grid_row), axis=1)
        depthmap_data = np.array(self.depth_map)

        pano_thetas1 = pano_thetas
        pano_thetas = pano_thetas1 - np.pi / 2.0

        pano_phis1 = pano_phis - np.pi
        pano_phis = pano_phis1 - (phi0 - pano_yaw)
        pano_phis = np.where(pano_phis > math.pi, pano_phis - math.pi * 2.0, pano_phis)
        pano_phis = np.where(pano_phis < -math.pi, pano_phis + math.pi * 2.0, pano_phis)

        new_grid = np.array([pano_phis1, pano_thetas1]).T

        depths = interpolate.griddata(grid_points, depthmap_data, new_grid, method='linear').ravel()

        filter_idxs = np.argwhere((depths < 20.0) & (depths > 0.0))
        if filter_idxs.size == 0:
            return 0
        pano_thetas = np.take(pano_thetas, filter_idxs)
        pano_phis = np.take(pano_phis, filter_idxs)
        depths = np.take(depths, filter_idxs)
        ppoint_classes = np.take(img.ravel(), filter_idxs)
        ppoint_classes = np.take(np.flipud(img).ravel(), filter_idxs)

        # calculate points cloud
        pointX = depths * np.cos(pano_thetas) * np.sin(pano_phis)
        pointY = depths * np.cos(pano_thetas) * np.cos(pano_phis)
        pointZ = depths * np.sin(pano_thetas)
        # pointsCloud = np.stack((pointX,pointY,pointZ,ppoint_classes))
        # pointsCloud= np.array([pointX,pointY,pointZ,ppoint_classes]).T
        pointsCloud = np.column_stack((pointX, pointY, pointZ, ppoint_classes))
        # transform camera coordinate
        # get camera center world coordinate
        # world_x, world_y = czhUtil.lonlat_to_proj(cam_pos_lng, cam_pos_lat)
        # pointsCloud_xyz = pointsCloud[:, :3]
        # pointsCloud_xyz[:] = rotate_x(-theta).dot(rotate_z(-phi)).dot(pointsCloud_xyz.T).T
        #
        # pointsCloud_xyz[:] = pointsCloud_xyz + np.array([world_x, world_y, cam_pos_elev])

        return pointsCloud
        # return raw_pointsCloud  #,pointsColor

    @staticmethod
    def parse(b64_string):
        """
        Base 64 decodes and then decompresses the deptham data provided by the Google streetview unofficial API
        :param b64_string: Base 64 string to decode
        :return: np array of decoded and decompressed depthmap data.
        """
        # fix the 'incorrect padding' error. The length of the string needs to be divisible by 4.
        b64_string += "=" * ((4 - len(b64_string) % 4) % 4)
        # convert the URL safe format to regular format.
        data = b64_string.replace("-", "+").replace("_", "/")

        data = base64.b64decode(data)  # decode the string
        data = zlib.decompress(data)  # decompress the data

        return np.array([d for d in data])

    @staticmethod
    def get_bin(a):
        ba = bin(a)[2:]
        return "0" * (8 - len(ba)) + ba

    @classmethod
    def get_int(cls, arr, ind):
        a = arr[ind]
        b = arr[ind + 1] << 8
        return a | b

    @classmethod
    def get_float(cls, arr, ind):
        return cls.bin_to_float("".join(cls.get_bin(i) for i in arr[ind: ind + 4][::-1]))

    @staticmethod
    def bin_to_float(binary):
        return struct.unpack("!f", struct.pack("!I", int(binary, 2)))[0]
