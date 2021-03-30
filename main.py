import os

import torch
import tornado.ioloop
import tornado.web

from PIL import Image
from io import BytesIO

from model import Net, predict_image

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = Net()
model.load_state_dict(torch.load(os.path.join('/Users/kilDz/Downloads/', f"model_epoch_12.pth"), map_location=device.type))
model.eval()


class ImageHandler(tornado.web.RequestHandler):
    def get(self):
        self.write('<html><body><form action="/" enctype="multipart/form-data" method="post">'
                   '<input type="file" name="headimg">'
                   '<input type="submit" value="Submit">'
                   '</form></body></html>')

    def post(self):
        im = Image.open(BytesIO(self.request.files["headimg"][0]["body"]))
        image = im.resize((300, 300))
        with torch.no_grad():
            prediction = predict_image(image)
        self.write(prediction)


def make_app():
    return tornado.web.Application([
        (r"/", ImageHandler),
    ])


if __name__ == "__main__":
    app = make_app()
    app.listen(3000)
    tornado.ioloop.IOLoop.current().start()
