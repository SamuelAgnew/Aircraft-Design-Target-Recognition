# Web streaming example
# Source code from the official PiCamera package
# http://picamera.readthedocs.io/en/latest/recipes2.html#web-streaming

import io
import picamera
import logging
import socketserver
from threading import Condition
from http import server
from main import detection, solution
from config import Settings
from collections import Counter
import time
import colour_recognition
import character_recognition
from saving import Saving
from gps import *

PAGE="""\
<html>
<head>
<title>Team Nyx Raspberry Pi Live Stream</title>
</head>
<body>
<center><h1>Team Nyx Raspberry Pi Live Stream</h1></center>
<center><img src="stream.mjpg" width="640" height="480"></center>
</body>
</html>
"""

class StreamingOutput(object):
    def __init__(self):
        self.frame = None
        self.buffer = io.BytesIO()
        self.condition = Condition()

    def write(self, buf):
        if buf.startswith(b'\xff\xd8'):
            # New frame, copy the existing buffer's content and notify all
            # clients it's available
            self.buffer.truncate()
            with self.condition:
                self.frame = self.buffer.getvalue()
                self.condition.notify_all()
            self.buffer.seek(0)
        return self.buffer.write(buf), self.frame

class StreamingHandler(server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(301)
            self.send_header('Location', '/index.html')
            self.end_headers()
        elif self.path == '/index.html':
            content = PAGE.encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.send_header('Content-Length', len(content))
            self.end_headers()
            self.wfile.write(content)
        elif self.path == '/stream.mjpg':
            self.send_response(200)
            self.send_header('Age', 0)
            self.send_header('Cache-Control', 'no-cache, private')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=FRAME')
            self.end_headers()
            try:
                while True:
                    with output.condition:
                        output.condition.wait()
                        frame = output.frame
                    self.wfile.write(b'--FRAME\r\n')
                    self.send_header('Content-Type', 'image/jpeg')
                    self.send_header('Content-Length', len(frame))
                    self.end_headers()
                    self.wfile.write(frame)
                    self.wfile.write(b'\r\n')
            except Exception as e:
                logging.warning(
                    'Removed streaming client %s: %s',
                    self.client_address, str(e))
        else:
            self.send_error(404)
            self.end_headers()

class StreamingServer(socketserver.ThreadingMixIn, server.HTTPServer):
    allow_reuse_address = True
    daemon_threads = True


def image_recognition(image):
    counter = 1
    marker = 1
    distance = 0
    predicted_character_list = []
    predicted_color_list = []
    end = time.time()
    start = time.time()
    config = Settings()
    save = Saving(config.name_of_folder, config.exist_ok)
    if counter == 1 or end - start < 10:
      frame = image.array
      end = time.time()
      
      color, roi, frame, success = detection(frame, config)
      
      if success:
        counter = counter + 1
    
        # time that the target has been last seen
        start = time.time()
    
        predicted_character, contour_image, chosen_image = character_recognition.character(color)
        predicted_color, processed_image = colour_recognition.colour(color)
    
        predicted_character_list.append(predicted_character)
        predicted_color_list.append(predicted_color)
    
        if config.save_results:
            name_of_results = ["color", "roi", "frame","contour_image","processed_image", "chosen_image"]
            image_results = [color, roi, frame, contour_image, processed_image, chosen_image]
            for value, data in enumerate(name_of_results):
                image_name = f"{marker}_{data}_{counter}.jpg"
                image = image_results[value]
                if image is not None:
                    save.save_the_image(image_name, image)
      
      if counter == 8:
        print("Starting Recognition Thread")
        common_character = Counter(predicted_character_list).most_common(1)[0][0]
        common_color = Counter(predicted_color_list).most_common(1)[0][0]
        marker, counter = solution(counter, marker, common_character, common_color, save.save_dir)
        predicted_character_list = []
        predicted_color_list = []
      
      else:
        print("Starting Recognition Thread")
        common_character = Counter(predicted_character_list).most_common(1)[0][0]
        common_color = Counter(predicted_color_list).most_common(1)[0][0]
        marker, counter = solution(counter, marker, common_character, common_color, save.save_dir)
        predicted_character_list = []
        predicted_color_list = []
        cap.truncate(0)


with picamera.PiCamera(resolution='640x480', framerate=24) as camera:
    output = StreamingOutput()
    #from picamera.array import PiRGBArray
    #cap = PiRGBArray(camera, size=(640, 480))
    #for image in camera.capture_continuous(cap, format="bgr", use_video_port=True):
      #image_recognition(image)
      #break
    #print("hi")
    
    #Uncomment the next line to change your Pi's Camera rotation (in degrees)
    #camera.rotation = 90
    camera.start_recording(output, format='mjpeg')
    try:
        address = ('', 8000)
        server = StreamingServer(address, StreamingHandler)
        server.serve_forever()
    finally:
        camera.stop_recording()

