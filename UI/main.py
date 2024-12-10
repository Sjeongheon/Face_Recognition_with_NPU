from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import base64
import numpy as np
from PIL import Image
import io
import os
from datetime import datetime
import asyncio
import subprocess
import signal
import sys
import cv2

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")


class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []
        self.building_process = None
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.image_count = 0

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        print("WebSocket 클라이언트가 연결되었습니다.")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        print("WebSocket 클라이언트의 연결이 해제되었습니다.")
        self.stop_building_process()

    def start_building_process(self):
        if not self.building_process:
            # main.py와 같은 경로 있는 최상위 폴더를 시작으로함 (github에 존재하는 원본 폴더기준)
            self.building_process = subprocess.Popen(
                [sys.executable, "./face-recognition/building_dataset.py"])
            return True
        return False

    def stop_building_process(self):
        if self.building_process:
            self.building_process.terminate()
            self.building_process = None
        self.image_count = 0

    def reset_count(self):
        self.image_count = 0

    def start_face_learning(self):
        """얼굴 학습 프로세스 시작"""
        try:
            # main.py와 같은 경로 있는 최상위 폴더를 시작으로함 (github에 존재하는 원본 폴더기준)
            subprocess.run([
                sys.executable,
                "./face-recognition/add_persons.py",
                "--backup-dir", "./face-recognition/datasets/backup",
                "--add-persons-dir", "./face-recognition/datasets/new_persons",
                "--faces-save-dir", "./face-recognition/datasets/data/",
                "--features-path", "./face-recognition/datasets/face_features/feature"
            ])
            return True
        except Exception as e:
            print(f"얼굴 학습 중 오류 발생: {str(e)}")
            return False

    def process_recognition(self, frame):
        """프레임을 저장하고 recognize.py를 실행하여 얼굴 인식을 수행"""
        try:
            # 임시 이미지 파일로 저장
            temp_image_path = "./face-recognition/temp_recognition.jpg"
            
            cv2.imwrite(temp_image_path, frame)
            
            # recognize.py 실행
            # main.py와 같은 경로 있는 최상위 폴더를 시작으로함 (github에 존재하는 원본 폴더기준)
            process = subprocess.Popen(
                [sys.executable, "./face-recognition/recognize.py", temp_image_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                universal_newlines=True
            )

            # 결과 읽기
            stdout, stderr = process.communicate()
            
            # 임시 파일 삭제
            if os.path.exists(temp_image_path):
                os.remove(temp_image_path)

            # 결과 확인
            if "님 환영합니다!" in stdout:
                name = stdout.split("님")[0]
                return name

            return None

        except Exception as e:
            print(f"Recognition error: {str(e)}")
            return None


manager = ConnectionManager()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)

    # async def send_good_after_delay():
    #     await asyncio.sleep(5)  # 15초 대기
    #     await websocket.send_json({
    #         "status": "good"
    #     })
    #     print("10초 후 good 메시지 전송됨")
    # # 비동기 태스크 시작
    # asyncio.create_task(send_good_after_delay())

    try:
        while True:
            data = await websocket.receive_text()
            try:
                if data.startswith("start_building"):
                    if manager.start_building_process():
                        await websocket.send_json({
                            "status": "success",
                            "message": "얼굴 등록이 시작되었습니다.",
                            "type": "start"
                        })
                elif data.startswith("stop_building"):
                    manager.stop_building_process()
                    await websocket.send_json({
                        "status": "success",
                        "message": "얼굴 등록이 중지되었습니다.",
                        "type": "stop"
                    })
                elif data.startswith("start_recognition"):
                    await websocket.send_json({
                        "status": "success",
                        "message": "얼굴 인식이 시작되었습니다.",
                        "type": "recognition_start"
                    })
                else:  # 이미지 데이터 처리
                    try:
                        # Base64 데이터 처리
                        image_data = data.split(",", 1)[1]
                        image_bytes = base64.b64decode(image_data)

                        # 이미지로 변환
                        image = Image.open(io.BytesIO(image_bytes))
                        frame = cv2.cvtColor(
                            np.array(image), cv2.COLOR_RGB2BGR)

                        # 인식 모드에서는 recognize.py로 처리
                        if data.startswith("recognize"):
                            print("인식모드실행")
                            # # 먼저 얼굴 감지 및 사각형 그리기
                            # faces = manager.face_cascade.detectMultiScale(
                            #     frame,
                            #     scaleFactor=1.2,
                            #     minNeighbors=6,
                            #     minSize=(20, 20)
                            # )
                            # # 얼굴이 감지되면 사각형 그리기
                            # if len(faces) > 0:
                            #     for (x, y, w, h) in faces:
                            #         cv2.rectangle(
                            #             frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

                            # 얼굴 인식 수행
                            recognized_name = manager.process_recognition(
                                frame)
                            
                            _, buffer = cv2.imencode('.jpg', frame)
                            processed_image = base64.b64encode(
                                buffer).decode('utf-8')
                            if recognized_name:
                                await websocket.send_json({
                                    "status": "success",
                                    "message": f"{recognized_name}님이 인식되었습니다.",
                                    "name": recognized_name,
                                    "type": "recognition_result",
                                    # 처리된 이미지도 함께 전송
                                    "image": f"data:image/jpeg;base64,{processed_image}"
                                })
                                await websocket.close()
                                return
                            else:
                                # 인식 실패시에도 이미지는 전송
                                await websocket.send_json({
                                    "status": "processing",
                                    "type": "recognition_processing",
                                    "image": f"data:image/jpeg;base64,{processed_image}"
                                })
                        else:
                            # 얼굴 감지
                            faces = manager.face_cascade.detectMultiScale(
                                frame,
                                scaleFactor=1.2,
                                minNeighbors=6,
                                minSize=(20, 20)
                            )

                            # 얼굴이 감지된 경우에만 처리
                            if len(faces) > 0:
                                for (x, y, w, h) in faces:
                                    cv2.rectangle(
                                        frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

                                manager.image_count += 1

                                # datasets/new_persons/1 디렉토리에 저장
                                # main.py와 같은 경로 있는 최상위 폴더를 시작으로함 (github에 존재하는 원본 폴더기준)
                                save_dir = "./face_recognition/datasets/new_persons/1"
                                os.makedirs(save_dir, exist_ok=True)
                                filename = os.path.join(
                                    save_dir, f"image{manager.image_count}.jpg")
                                cv2.imwrite(filename, frame)

                                # 처리된 이미지를 base64로 인코딩
                                _, buffer = cv2.imencode('.jpg', frame)
                                processed_image = base64.b64encode(
                                    buffer).decode('utf-8')

                                await websocket.send_json({
                                    "status": "success",
                                    "message": f"이미지 저장 완료: {filename}",
                                    "image": f"data:image/jpeg;base64,{processed_image}",
                                    "count": manager.image_count,
                                    "type": "image"
                                })

                                if manager.image_count >= 100:
                                    # 얼굴 저장 프로세스 종료
                                    print("저장종료")
                                    manager.stop_building_process()
                                    # 얼굴 학습 프로세스 시작
                                    print("얼굴학습시작부분")
                                    if manager.start_face_learning():
                                        await websocket.send_json({
                                            "status": "complete",
                                            "message": "얼굴 등록 및 학습이 완료되었습니다.",
                                            "type": "complete"
                                        })
                                    else:
                                        await websocket.send_json({
                                            "status": "error",
                                            "message": "얼굴 학습 중 오류가 발생했습니다.",
                                            "type": "error"
                                        })
                                    manager.reset_count()
                                    # WebSocket 연결 종료
                                    await websocket.close()
                                    return  # WebSocket 핸들러 종료

                                    # await websocket.send_json({
                                    #     "status": "complete",
                                    #     "message": "얼굴 등록이 완료되었습니다.",
                                    #     "type": "complete"
                                    # })
                                    # manager.reset_count()

                    except Exception as e:
                        print(f"이미지 처리 중 오류 발생: {str(e)}")
                        await websocket.send_json({
                            "status": "error",
                            "message": f"이미지 처리 중 오류 발생: {str(e)}",
                            "type": "error"
                        })

            except Exception as e:
                print(f"처리 중 오류 발생: {str(e)}")
                await websocket.send_json({
                    "status": "error",
                    "message": f"처리 중 오류 발생: {str(e)}"
                })

    except Exception as e:
        print(f"WebSocket 오류: {str(e)}")
        manager.disconnect(websocket)


@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("proj.html", "r", encoding='utf-8') as f:
        return f.read()


# @app.get("/", response_class=HTMLResponse)
# async def read_root():
#     with open("proj.html", "r", encoding='utf-8') as f:
#         content = f.read()
#         return content
