from flask import Flask, Response, request, abort
from flask_cors import CORS
import cv2
import ipaddress
import argparse

parser = argparse.ArgumentParser(description="Flask server for video streaming.")
parser.add_argument("--camera_port", type=int, required=True, help="Camera port number (/dev/video?).")
parser.add_argument("--network", type=str, default="192.168.1.0/24", help="Allowed network range (default: 192.168.1.0/24).")
parser.add_argument("--port", type=int, default=5000, help="Flask server port (default: 5000).")
args = parser.parse_args()

app = Flask(__name__)
CORS(app)

camera = cv2.VideoCapture(args.camera_port)

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.before_request
def limit_remote_addr():
    allowed_network = ipaddress.IPv4Network(args.network)
    client_ip = ipaddress.IPv4Address(request.remote_addr)
    if client_ip not in allowed_network:
        abort(403)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=args.port)
