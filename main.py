

import cv2

from edgesight.config import (FRAME_WIDTH, FRAME_HEIGHT, FPS)

def main():
    print("EdgeSight initialized successfully!")

    print(f"OpenCV version: {cv2.__version__}")

    print(f"Configured resolution: {FRAME_WIDTH}x{FRAME_HEIGHT}")
    print(f"Target FPS: {FPS}")

if __name__ == "__main__":
    main()