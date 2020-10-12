from flask import Flask, request, render_template, send_file, flash, redirect
import torch
from torchvision import transforms
import cv2
import numpy as np
from model import MultiScaleResidualNetwork
import io


app = Flask(__name__)
app.secret_key = 'some secret key'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


def get_model(scale):

    if scale == "x2":
        net = MultiScaleResidualNetwork(2, 8, 64, 64, infer=True)
        checkpoint = torch.load("./models/x2.pt", map_location=torch.device('cpu'))
    else:
        net = MultiScaleResidualNetwork(4, 8, 64, 64, infer=True)
        checkpoint = torch.load("./models/x4.pt", map_location=torch.device('cpu'))

    net.load_state_dict(checkpoint["model"])
    net.eval()
    return net


def get_tensor(image_bytes):

    np_img = np.fromstring(image_bytes, np.uint8)
    image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    means = torch.tensor([np.mean(image[0, :, :]), np.mean(image[1, :, :]), np.mean(image[2, :, :])])
    image = torch.from_numpy(image)
    image = image - means

    return image.permute(2, 0, 1).unsqueeze(0).type(torch.float32)


def get_prediction(image_bytes, scale):

    model = get_model(scale)
    tensor = get_tensor(image_bytes)
    output = model.forward(tensor)
    return output


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def hello_world():

    if request.method == 'GET':
        return render_template('index.html')

    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file')
            return redirect(request.url)

        file = request.files['file']
        if file.filename == '':
            flash('No file')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            image = file.read()

            output = get_prediction(image_bytes=image, scale=request.form['submit_button'])

            img = transforms.ToPILImage()(output[0, :, :, :])
            img_io = io.BytesIO()
            img.save(img_io, 'JPEG')
            img_io.seek(0)

            return send_file(img_io, mimetype='image/jpeg', as_attachment=True, attachment_filename='res.jpeg')

        else:
            flash('Wrong file type')
            return redirect(request.url)


if __name__ == '__main__':
    app.run(debug=True)
