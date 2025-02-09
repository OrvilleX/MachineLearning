
class TrackItem:
    def __init__(self, track_id, **kwargs):
        self.track_id = track_id
        self.y_min = kwargs.get('y_min', 0.)
        self.y_max = kwargs.get('y_max', 0.)
        self.x_min = kwargs.get('x_min', 0.)
        self.x_max = kwargs.get('x_max', 0.)
        self.conf = kwargs.get('conf', 0.)
        self.center_y = kwargs.get('center_y', 0.)
        self.center_x = kwargs.get('center_x', 0.)
        self.width = kwargs.get('width', 0.)
        self.height = kwargs.get('height', 0.)

    def set_xyxy(self, x_min, y_min, x_max, y_max):
        self.x_min = int(x_min)
        self.y_min = int(y_min)
        self.x_max = int(x_max)
        self.y_max = int(y_max)

    def set_xywh(self, center_x, center_y, width, height):
        self.center_x = int(center_x)
        self.center_y = int(center_y)
        self.width = width
        self.height = height
