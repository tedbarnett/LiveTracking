"""Why isn't the guitar detected? Show each gate separately on the ambient frame:
projection quad, raw white mask, depth-near mask, and where the biggest white blob is."""
import os
import time

import cv2
import numpy as np
import pyrealsense2 as rs

OUT = os.path.join(os.path.dirname(__file__), "out")
EXP = 1200


def order_quad(pts):
    pts = np.array(pts, dtype=np.float32).reshape(-1, 2)
    s = pts.sum(1); d = np.diff(pts, axis=1).reshape(-1)
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)],
                     pts[np.argmax(s)], pts[np.argmax(d)]], np.float32)


devs = list(rs.context().query_devices())
if devs:
    devs[0].hardware_reset()
for _ in range(30):
    time.sleep(0.5)
    if list(rs.context().query_devices()):
        break
time.sleep(2.0)

p = rs.pipeline(); c = rs.config()
c.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)
c.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
prof = p.start(c)
ds = prof.get_device().first_depth_sensor().get_depth_scale()
align = rs.align(rs.stream.color)
sensor = prof.get_device().query_sensors()[1]
sensor.set_option(rs.option.enable_auto_exposure, 0)
sensor.set_option(rs.option.exposure, EXP)

import pygame
pygame.init()
sizes = pygame.display.get_desktop_sizes()
pi = max(range(len(sizes)), key=lambda i: sizes[i][0]*sizes[i][1])
screen = pygame.display.set_mode(sizes[pi], pygame.NOFRAME, display=pi)


def show(rgb):
    screen.fill(rgb); pygame.display.flip()
    for _ in range(8):
        pygame.event.pump(); time.sleep(0.03)


def grab():
    for _ in range(25):
        p.wait_for_frames()
    f = align.process(p.wait_for_frames())
    col = np.asanyarray(f.get_color_frame().get_data())
    dep = np.asanyarray(f.get_depth_frame().get_data()).astype(np.float32)*ds
    return col, dep


show((0,0,0)); time.sleep(1.0); cam, depth = grab()
show((255,255,255)); time.sleep(1.0); camw, _ = grab()
show((0,0,0))
p.stop(); pygame.quit()

# projection quad
diff = np.clip(cv2.cvtColor(camw,cv2.COLOR_BGR2GRAY).astype(np.int16)
               - cv2.cvtColor(cam,cv2.COLOR_BGR2GRAY).astype(np.int16),0,255).astype(np.uint8)
_, m = cv2.threshold(diff,30,255,cv2.THRESH_BINARY)
m = cv2.morphologyEx(m,cv2.MORPH_CLOSE,np.ones((25,25),np.uint8))
m = cv2.morphologyEx(m,cv2.MORPH_OPEN,np.ones((9,9),np.uint8))
cnts,_ = cv2.findContours(m,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
big = max(cnts,key=cv2.contourArea)
ap = cv2.approxPolyDP(big,0.02*cv2.arcLength(big,True),True)
quad = order_quad(ap.reshape(-1,2) if len(ap)==4 else cv2.boxPoints(cv2.minAreaRect(big)))
pmask = np.zeros(diff.shape,np.uint8); cv2.fillPoly(pmask,[quad.astype(np.int32)],255)

# gates
hsv = cv2.cvtColor(cam,cv2.COLOR_BGR2HSV)
white = ((hsv[:,:,1]<70)&(hsv[:,:,2]>150)).astype(np.uint8)*255
wall_vals = depth[(pmask>0)&(depth>0)]
wall_d = float(np.median(wall_vals)) if wall_vals.size else 0
near = ((depth>0.3)&(depth<wall_d-0.15)).astype(np.uint8)*255

print(f"wall_d={wall_d:.2f}  proj bbox={cv2.boundingRect(big)}")
print(f"raw white px={int((white>0).sum())}  near px={int((near>0).sum())}")
print(f"white&proj px={int((cv2.bitwise_and(white,pmask)>0).sum())}")

# biggest raw white blob + is its center inside projection quad?
wc,_ = cv2.findContours(white,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
wc = [x for x in wc if cv2.contourArea(x)>300]
ann = cam.copy()
cv2.polylines(ann,[quad.astype(np.int32)],True,(0,255,0),2)
for x in wc:
    cv2.drawContours(ann,[x],0,(0,255,255),2)
if wc:
    wb = max(wc,key=cv2.contourArea)
    M = cv2.moments(wb); cxc=int(M['m00'] and M['m10']/M['m00']); cyc=int(M['m00'] and M['m01']/M['m00'])
    inside = cv2.pointPolygonTest(quad.astype(np.int32),(cxc,cyc),False)
    cv2.circle(ann,(cxc,cyc),6,(0,0,255),-1)
    print(f"biggest white blob area={cv2.contourArea(wb):.0f} center=({cxc},{cyc}) "
          f"insideProjection={inside>=0}")

cv2.imwrite(os.path.join(OUT,"dbg_annot.png"),ann)
cv2.imwrite(os.path.join(OUT,"dbg_white.png"),white)
cv2.imwrite(os.path.join(OUT,"dbg_near.png"),near)
cv2.imwrite(os.path.join(OUT,"dbg_proj.png"),pmask)
print("saved dbg_annot/white/near/proj.png")
