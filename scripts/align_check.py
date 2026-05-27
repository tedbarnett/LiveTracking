"""Show the projection area (where the projector throws light) vs the guitar
location, so we can see whether the projector needs re-aiming."""
import os
import time

import cv2
import numpy as np
import pyrealsense2 as rs

OUT = os.path.join(os.path.dirname(__file__), "out")
EXP = 1000

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
s = prof.get_device().query_sensors()[1]
s.set_option(rs.option.enable_auto_exposure, 0)
s.set_option(rs.option.exposure, EXP)

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
    return (np.asanyarray(f.get_color_frame().get_data()),
            np.asanyarray(f.get_depth_frame().get_data()).astype(np.float32)*ds)


show((0,0,0)); time.sleep(1.0); cam, depth = grab()
show((255,255,255)); time.sleep(1.0); camw, _ = grab()
show((0,0,0))
p.stop(); pygame.quit()

diff = np.clip(cv2.cvtColor(camw,cv2.COLOR_BGR2GRAY).astype(np.int16)
               - cv2.cvtColor(cam,cv2.COLOR_BGR2GRAY).astype(np.int16),0,255).astype(np.uint8)
_, m = cv2.threshold(diff,30,255,cv2.THRESH_BINARY)
m = cv2.morphologyEx(m,cv2.MORPH_CLOSE,np.ones((25,25),np.uint8))
m = cv2.morphologyEx(m,cv2.MORPH_OPEN,np.ones((9,9),np.uint8))
cnts,_ = cv2.findContours(m,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
big = max(cnts,key=cv2.contourArea)
px,py,pw,ph = cv2.boundingRect(big)

# guitar = white & near, largest blob in lower 55% of frame (the sofa)
hsv = cv2.cvtColor(cam,cv2.COLOR_BGR2HSV)
white = ((hsv[:,:,1]<80)&(hsv[:,:,2]>140)).astype(np.uint8)*255
pmask = np.zeros(diff.shape,np.uint8); cv2.fillPoly(pmask,[big],255)
wall_d = float(np.median(depth[(pmask>0)&(depth>0)]))
near = ((depth>0.3)&(depth<wall_d-0.15)).astype(np.uint8)*255
g = cv2.bitwise_and(white, cv2.bitwise_or(near,(depth==0).astype(np.uint8)*255))
g[:int(0.40*g.shape[0]),:] = 0   # ignore wall region (upper 40%)
g = cv2.morphologyEx(g,cv2.MORPH_CLOSE,np.ones((15,15),np.uint8))
gc,_ = cv2.findContours(g,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
gc = [x for x in gc if cv2.contourArea(x)>300]

ann = cam.copy()
cv2.rectangle(ann,(px,py),(px+pw,py+ph),(0,255,0),3)
cv2.putText(ann,"PROJECTOR light",(px+5,py+25),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
gtop = None
if gc:
    gb = max(gc,key=cv2.contourArea)
    gx,gy,gw,gh = cv2.boundingRect(gb)
    gtop = gy
    cv2.rectangle(ann,(gx,gy),(gx+gw,gy+gh),(0,140,255),3)
    cv2.putText(ann,"GUITAR",(gx+5,gy+gh-8),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,140,255),2)
    print(f"projection bottom y={py+ph}, guitar top y={gy}, guitar bbox=({gx},{gy},{gw},{gh})")
    print(f"overlap rows: {max(0, (py+ph) - gy)} px")
print(f"projection bbox=({px},{py},{pw},{ph})  wall_d={wall_d:.2f}")
cv2.imwrite(os.path.join(OUT,"align.png"),ann)
print("saved align.png")
