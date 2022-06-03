#!/usr/bin/env python3
# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Trigger PiCamera when face is detected."""

import vlc
import time
from datetime import datetime

from aiy.vision.inference import CameraInference
from aiy.vision.models import face_detection

from picamera import PiCamera

def generate_filename():
    timestamp = datetime.now()
    return "image_" + timestamp.strftime("%Y-%m-%d_%H:%M:%S.%f") + ".jpg"

def test_filenames():
    for i in range(50):
        print(generate_filename())
        time.sleep(0.3)

def main():
    with PiCamera() as camera:
        # Configure camera
        camera.resolution = (1640, 922)  # Full Frame, 16:9 (Camera v2)

        # Do inference on VisionBonnet
        with CameraInference(face_detection.model()) as inference:
            for result in inference.run():
                if len(face_detection.get_faces(result)) >= 1:
                    camera.capture(generate_filename())
                    player = vlc.MediaPlayer("/home/pi/Music/Rick2.mp3")
                    player.play()
                    time.sleep(65)


if __name__ == '__main__':
    main()
    # test_filenames()

