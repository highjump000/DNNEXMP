import random
from PIL import Image
import numpy as np


imgx = 500
imgy = 500
image = Image.new("RGB", (imgx, imgy))
pixels = image.load()
mx = 50
my = 50  # width and height of the maze
maze = [[0 for x in range(mx)] for y in range(my)]
dx = [0, 1, 0, -1]
dy = [-1, 0, 1, 0]  # 4 directions to move in the maze
color = [(0, 0, 0), (255, 255, 255),(0, 255, 0),(255, 0, 0)]  # RGB colors of the maze
# start the maze from a random cell
cx = random.randint(0, mx - 1)
cy = random.randint(0, my - 1)
maze[cy][cx] = 1
stack = [(cx, cy, 0)]  # stack element: (x, y, direction)
'''
while len(stack) > 0:
    (cx, cy, cd) = stack[-1]
    # to prevent zigzags:
    # if changed direction in the last move then cannot change again
    if len(stack) > 2:
        if cd != stack[-2][2]:
            dirRange = [cd]
        else:
            dirRange = range(4)
    else:
        dirRange = range(4)

    # find a new cell to add
    nlst = []  # list of available neighborsimg
    for i in dirRange:
        nx = cx + dx[i]
        ny = cy + dy[i]
        if 0 <= nx < mx and 0 <= ny < my:
            if maze[ny][nx] == 0:
                ctr = 0  # of occupied neighbors must be 1
                for j in range(4):
                    ex = nx + dx[j]
                    ey = ny + dy[j]
                    if 0 <= ex < mx and 0 <= ey < my:
                        if maze[ey][ex] == 1:
                            ctr += 1
                if ctr == 1:
                    nlst.append(i)

    # if 1 or more neighbors available then randomly select one and move
    if len(nlst) > 0:
        ir = nlst[random.randint(0, len(nlst) - 1)]
        cx += dx[ir]
        cy += dy[ir]
        maze[cy][cx] = 1
        stack.append((cx, cy, ir))
    else:
        stack.pop()
# change it to [0,1] list
maze[0][0] = 3 # end point
maze[my-1][mx-1] = 2 # start point

for i in range(len(maze)):
    print maze[i]
'''
maze = np.load('maze.npy')

# save maze in array form
for ky in range(imgy):
    for kx in range(imgx):
       pixels[kx, ky] = color[maze[my * ky / imgy][mx * kx / imgx]]


#np.save('maze',maze)

image.show("Maze_" + str(mx) + "x" + str(my) + ".png", "PNG")

image.save("Maze_" + str(mx) + "x" + str(my) + ".png", "PNG")