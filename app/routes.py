from flask import Blueprint, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
from model.evaluate import predict_image

main = Blueprint('main', __name__)

@main.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Save uploaded file
        if "file" not in request.files:
            return redirect(request.url)
        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join("app/static/uploads", filename)
            file.save(filepath)

            # Predict the uploaded image
            prediction, probabilities = predict_image(filepath)

            return render_template("results.html", prediction=prediction, probabilities=probabilities, image_url=filepath)

    return render_template("index.html")