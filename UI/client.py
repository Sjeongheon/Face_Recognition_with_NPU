import cv2
import argparse

parser = argparse.ArgumentParser(description="Connect to video stream from server.")
parser.add_argument("--ip", required=True, help="IP address of the video stream server.")
parser.add_argument("--port", default=5000, help="Port of the video stream server (default: 5000).")
args = parser.parse_args()

video_url = f"http://{args.ip}:{args.port}/video_feed"

cap = cv2.VideoCapture(video_url)

if not cap.isOpened():
    print(f"Unable to connect to stream URL: {video_url}")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Unable to read frames from stream.")
        break

    cv2.imshow("Video Stream", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
