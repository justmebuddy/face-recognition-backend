from flask import Flask, request, jsonify
import face_recognition, cv2, numpy as np, base64, os
from datetime import datetime
from io import BytesIO
from PIL import Image

app = Flask(__name__)

known_encodings = []
known_names = []

for file in os.listdir("known_faces"):
    img = face_recognition.load_image_file(f"known_faces/{file}")
    enc = face_recognition.face_encodings(img)
    if enc:
        known_encodings.append(enc[0])
        known_names.append(file.split(".")[0])

@app.route("/recognize", methods=["POST"])
def recognize():
    data = request.json
    img_data = base64.b64decode(data["image"])
    img = Image.open(BytesIO(img_data)).convert("RGB")
    np_img = np.array(img)

    small_frame = cv2.resize(np_img, (0, 0), fx=0.25, fy=0.25)
    rgb_small = small_frame[:, :, ::-1]
    
    locations = face_recognition.face_locations(rgb_small)
    encodings = face_recognition.face_encodings(rgb_small, locations)

    response = []
    for loc, encoding in zip(locations, encodings):
        matches = face_recognition.compare_faces(known_encodings, encoding)
        name = "Unknown"
        if True in matches:
            idx = matches.index(True)
            name = known_names[idx]
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            top, right, bottom, left = [v*4 for v in loc]
            face_img = np_img[top:bottom, left:right]
            filename = f"suspects/suspect_{timestamp}.jpg"
            cv2.imwrite(filename, cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR))
        response.append(name)
    return jsonify(response)
