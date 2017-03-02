from PIL import Image

def plot(value):
    imgx = 500
    imgy = 500
    image = Image.new("RGB", (imgx, imgy))
    pixels = image.load()
   # for ky in range(imgy):
   #     for kx in range(imgx):
           # if
           # pixels[kx, ky] = color[maze[my * ky / imgy][mx * kx / imgx]]
