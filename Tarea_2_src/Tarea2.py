#!/usr/bin/env python
# manual

"""
This script allows you to manually control the simulator or Duckiebot
using the keyboard arrows.
"""

import sys
import argparse
import pyglet
import cv2
from pyglet.window import key
import numpy as np
import gym
import gym_duckietown
from gym_duckietown.envs import DuckietownEnv
from gym_duckietown.wrappers import UndistortWrapper



# from experiments.utils import save_img

parser = argparse.ArgumentParser()
parser.add_argument('--env-name', default='Duckietown_udem1')
parser.add_argument('--map-name', default='udem1')
parser.add_argument('--distortion', default=False, action='store_true')
parser.add_argument('--draw-curve', action='store_true', help='draw the lane following curve')
parser.add_argument('--draw-bbox', action='store_true', help='draw collision detection bounding boxes')
parser.add_argument('--domain-rand', action='store_true', help='enable domain randomization')
parser.add_argument('--frame-skip', default=1, type=int, help='number of frames to skip')
parser.add_argument('--seed', default=1, type=int, help='seed')
args = parser.parse_args()

if args.env_name and args.env_name.find('Duckietown') != -1:
    env = DuckietownEnv(
        seed = args.seed,
        map_name = args.map_name,
        draw_curve = args.draw_curve,
        draw_bbox = args.draw_bbox,
        domain_rand = args.domain_rand,
        frame_skip = args.frame_skip,
        distortion = args.distortion,
    )
else:
    env = gym.make(args.env_name)

env.reset()
env.render()

# Initialize to check if HSV min/max value changes
hMin = sMin = vMin = hMax = sMax = vMax = 0
phMin = psMin = pvMin = phMax = psMax = pvMax = 0

def bordes(obs, umbral_low, pr):
    obs2 = obs.copy()
    yellow = filter_color(obs, "yellow")
    #red = filter_color(obs, "red")
    white = filter_color(obs, "white")
    images = [white, yellow]

    lines_yellow = (0, 255, 255)
    #lines_red = (0, 0, 255)
    lines_white = (255, 255, 255)
    lines = [lines_white, lines_yellow]

    for img_line in zip(images, lines):
        umbral_high = umbral_low*pr
        image_gauss = cv2.GaussianBlur(img_line[0],(5,5),2)
        edges = cv2.Canny(image_gauss, umbral_low, umbral_high, apertureSize = 3)

        lines = cv2.HoughLines(edges, 1, np.pi/180, 60, None)
        
        if lines is not None:
            for rho, theta in lines[:, 0]:
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))
                cv2.line(obs2, (x1, y1), (x2, y2), img_line[1], 3)
    return obs2

def bordes_pro(obs, umbral_low, pr):
    obs2 = obs.copy()
    yellow = filter_color(obs, "yellow")
    #red = filter_color(obs, "red")
    white = filter_color(obs, "white")
    images = [white, yellow]

    lines_yellow = (0, 255, 255)
    #lines_red = (0, 0, 255)
    lines_white = (255, 255, 255)
    lines = [lines_white, lines_yellow]
    
    for img_line in zip(images, lines):
        umbral_high = umbral_low*pr
        image_gauss = cv2.GaussianBlur(img_line[0],(5,5),1)
        edges = cv2.Canny(image_gauss, umbral_low, umbral_high, apertureSize = 3)

        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, None, 50, 30)
        
        if lines is not None:
            for i in range(0, len(lines)):
                l = lines[i][0]
                cv2.line(obs2, (l[0], l[1]), (l[2], l[3]), img_line[1], 3, cv2.LINE_AA)
    return obs2

def filter_color(image, color):
    # Amarillo
    lower_yellow = np.array([89, 64, 138])
    upper_yellow= np.array([96, 240, 255])

    # Rojo
    lower_red = np.array([119, 61, 115])
    upper_red = np.array([128, 255, 255])

    # Blanco
    lower_white = np.array([0, 0, 145])
    upper_white = np.array([179, 52, 255])

    # Create HSV Image and threshold into a range.
    yellow = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    red = yellow.copy()
    white = yellow.copy()

    mask_yellow = cv2.inRange(yellow, lower_yellow, upper_yellow)
    mask_red = cv2.inRange(red, lower_red, upper_red)
    mask_white = cv2.inRange(white, lower_white, upper_white)

    kernel = np.ones((5,5),np.uint8)
    
    if color == "yellow": 
        op_morf2 = cv2.erode(mask_yellow,kernel,iterations = 1)
    elif color == "red": 
        op_morf2 = cv2.erode(mask_red,kernel,iterations = 1)
    elif color == "white": 
        op_morf2 = cv2.erode(mask_white,kernel,iterations = 1)
    else:
        op_morf2 = cv2.erode(image,kernel,iterations = 1)

    return cv2.dilate(op_morf2,kernel,iterations = 2)
    

@env.unwrapped.window.event
def on_key_press(symbol, modifiers):\

    """
    This handler processes keyboard commands that
    control the simulation
    """

    if symbol == key.BACKSPACE or symbol == key.SLASH:
        print('RESET')
        env.reset()
        env.render()
    elif symbol == key.PAGEUP:
        env.unwrapped.cam_angle[0] = 0
    elif symbol == key.ESCAPE:
        env.close()
        sys.exit(0)

    # Take a screenshot
    # UNCOMMENT IF NEEDED - Skimage dependency
    # elif symbol == key.RETURN:
    #     print('saving screenshot')
    #     img = env.render('rgb_array')
    #     save_img('screenshot.png', img)

# Register a keyboard handler
key_handler = key.KeyStateHandler()
env.unwrapped.window.push_handlers(key_handler)

def update(dt):
    global hMin ,sMin ,vMin ,hMax ,sMax ,vMax 
    global phMin ,psMin ,pvMin ,phMax ,psMax ,pvMax 
    """
    This function is called at every frame to handle
    movement/stepping and redrawing
    """

    action = np.array([0.0, 0.0])

    if key_handler[key.UP]:
        action = np.array([0.44, 0.0])
    if key_handler[key.DOWN]:
        action = np.array([-0.44, 0])
    if key_handler[key.LEFT]:
        action = np.array([0.35, +1])
    if key_handler[key.RIGHT]:
        action = np.array([0.35, -1])
    if key_handler[key.SPACE]:
        action = np.array([0, 0])

    # Speed boost
    if key_handler[key.LSHIFT]:
        action *= 1.5

    obs, reward, done, info = env.step(action)
    obs = cv2.cvtColor(obs, cv2.COLOR_BGR2RGB)
    print('step_count = %s, reward=%.3f' % (env.unwrapped.step_count, reward))

    if key_handler[key.RETURN]:
        from PIL import Image
        im = Image.fromarray(obs)

        im.save('screen.png')

    if done:
        print('done!')
        #env.reset()
        env.render()

    # Bordes
    edges = bordes(obs, 70, 3)
    cv2.imshow('edges',edges)

    # Bordes probabilistico
    edges2 = bordes_pro(obs, 70, 3)
    cv2.imshow('edges 2',edges2)

    cv2.waitKey(1)

    env.render()

pyglet.clock.schedule_interval(update, 1.0 / env.unwrapped.frame_rate)

# Enter main event loop
pyglet.app.run()

env.close()
