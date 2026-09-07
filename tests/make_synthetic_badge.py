import sys
sys.path.insert(0, ".")

import cv2
import numpy as np

img_path = r"C:\VisionLeadAgent-wx\my_agent\_preview\last.png"
img = cv2.imread(img_path)
print("img shape", img.shape)

# 在聊天图标右上角画红色数字徽章
# 从 last.png 观察，绿色聊天图标中心约 (30, 110)
cx, cy = 45, 95
r = 12
cv2.circle(img, (cx, cy), r, (50, 50, 235), -1)
cv2.putText(img, "2", (cx-4, cy+4), cv2.FONT_HERSHEY_SIMPLEX,
            0.35, (255, 255, 255), 1, cv2.LINE_AA)

out_path = r"C:\VisionLeadAgent-wx\my_agent\tests\synthetic_badge.png"
cv2.imwrite(out_path, img)
print("saved", out_path)
