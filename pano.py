import requests

from PIL import Image

from depthmap import Depthmap


class Pano:
    def __init__(self, pano_id: str = None, lat: float = None, lon: float = None, zoom=2):
        base_url = f"http://cbk0.google.com/cbk?output=json&dm=1"
        if pano_id is not None:
            query = f"&panoid={pano_id}"
        elif lat is not None and lon is not None:
            query = f"&ll={lat},{lon}"

        response = requests.get(base_url + query)
        if response.status_code == 404:
            raise Exception("Panorama not found")

        self.data = response.json()

        # Data
        self.image_width = int(self.data["Data"]["image_width"])
        self.image_height = int(self.data["Data"]["image_height"])
        self.tile_width = int(self.data["Data"]["tile_width"])
        self.tile_height = int(self.data["Data"]["tile_height"])

        # Projection
        self.pano_yaw_deg = float(self.data["Projection"]["pano_yaw_deg"])
        self.tilt_yaw_deg = float(self.data["Projection"]["tilt_yaw_deg"])
        self.tilt_pitch_deg = float(self.data["Projection"]["tilt_pitch_deg"])

        # Location
        self.pano_id = self.data["Location"]["panoId"]
        self.zoom_levels = int(self.data["Location"]["zoomLevels"])
        self.lat = float(self.data["Location"]["lat"])
        self.lon = float(self.data["Location"]["lng"])
        self.original_lat = float(self.data["Location"]["original_lat"])
        self.original_lon = float(self.data["Location"]["original_lng"])
        self.elevation = float(self.data["Location"]["elevation_wgs84_m"])
        if "best_view_direction_deg" in self.data["Location"]:
            self.best_view_direction_deg = float(self.data["Location"]["best_view_direction_deg"])
        else:
            self.best_view_direction_deg = -1

        # model
        self.depth_map = Depthmap(self.data["model"]["depth_map"])
        self.panorama = self.get_full_pano(self.pano_id, zoom)

    @staticmethod
    def empty_image(tile_x: int, tile_y: int, tile_size: int = 512) -> Image:
        """
        Create an empty image that is x tiles wide and y tiles long
        :param tile_x: width in number of tiles
        :param tile_y: height in number of tiles
        :param tile_size: default 512. Size of the square tile.
        :return: An empty PIL image
        """
        return Image.new('RGB', (tile_x * tile_size, tile_y * tile_size))

    @staticmethod
    def fill_tile(full_image: Image, tile: Image, tile_x: int, tile_y: int, tile_size: int = 512) -> Image:
        """
        Given a full image and a square tile, paste it into the correct spot to form a full image
        :param full_image: Original image to paste the tile into
        :param tile: A tile to paste into the full image
        :param tile_x: x position the tile belongs in
        :param tile_y: y position the tile belongs in
        :param tile_size: default 512. Size of the square tile.
        :return: Original image is returned, but it is redundant as the original image is modified.
        """
        full_image.paste(im=tile, box=(tile_size * tile_x, tile_size * tile_y))
        return full_image

    @staticmethod
    def get_pano_tile(pano_id: str, tile_x: int = 0, tile_y: int = 0, zoom: int = 5) -> Image:
        """
        Get individual tile of panorama given panorama id and tile x and y position with zoom.
        :param pano_id: Panorama ID
        :param tile_x:
        :param tile_y:
        :param zoom: Zoom level
        :return: One PIL image tile at x and y of size 512 x 512
        """
        pano_url = f"http://maps.google.com/cbk?output=tile&panoid={pano_id}&zoom={zoom}&x={tile_x}&y={tile_y}"

        response = requests.get(pano_url, stream=True)

        if response.status_code == 400:
            return None

        tile = Image.open(response.raw)
        return tile

    @classmethod
    def get_full_pano(cls, pano_id: str, zoom: int = 2, max_x: int = 32, max_y: int = 16) -> Image:
        """
        Given pano_id, download and stitch together a full panorama image
        :param pano_id: Panorama ID
        :param zoom: Zoom level
        :param max_x: Maximum number of x tiles
        :param max_y: Maximum number of y tiles
        :return: PIL image of full panorama
        """
        full = cls.empty_image(max_x, max_y)

        initial_max_x = max_x

        x = 0
        y = 0

        while y < max_y:
            while x < max_x:
                tile = cls.get_pano_tile(pano_id, x, y, zoom=zoom)
                if tile is None:
                    if max_x == initial_max_x:
                        max_x = x
                    else:
                        max_y = y
                else:
                    cls.fill_tile(full, tile, x, y)
                x += 1

            x = 0
            y += 1

        full = full.crop((0, 0, 512 * max_x, 512 * max_y))

        return full
