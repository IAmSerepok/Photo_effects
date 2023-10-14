import pygame as pg
import pygame.gfxdraw
import cv2
import os
from effect_functions import *
from copy import deepcopy


main_folder = os.path.dirname(__file__)
img_folder = os.path.join(main_folder, "input")
save_folder = os.path.join(main_folder, "output")


class PhotoFilters:

    def __init__(self, file_name):

        pg.init()
        self.path = os.path.join(img_folder, file_name)
        self.image, self.cv2_image = self.get_image()
        self.next_image = self.image
        self.screen_size = self.screen_width, self.screen_height = self.image.shape[0], self.image.shape[1]
        self.surface = pg.display.set_mode(self.screen_size)
        self.clock = pg.time.Clock()

    def draw_converted_image(self):
        for x in range(0, self.screen_width):
            for y in range(0, self.screen_height):
                color = tuple(self.image[x, y])
                pygame.gfxdraw.box(self.surface, (x, y, 1, 1), color)

    def save_image(self):# переделать
        pygame_image = pg.surfarray.array3d(self.surface)
        cv2_img = cv2.transpose(pygame_image)
        cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
        cv2.imwrite(os.path.join(save_folder, 'ascii_image_rgb.jpg'), cv2_img)

    def get_image(self):
        cv2_image = cv2.imread(self.path)
        transposed_image = cv2.transpose(cv2_image)
        image = cv2.cvtColor(transposed_image, cv2.COLOR_BGR2RGB)

        return image, cv2_image

    def draw_cv2_image(self):
        resized_cv2_image = cv2.resize(self.cv2_image, (self.screen_width//2, self.screen_height//2),
                                       interpolation=cv2.INTER_AREA)
        cv2.imshow('img', resized_cv2_image)

    def convolution(self, function):
        for x in range(1, self.screen_width - 1):
            for y in range(1, self.screen_height - 1):
                self.next_image[x, y] = function(self.image, x, y)
        self.image = deepcopy(self.next_image)

    def draw(self):
        self.surface.fill('black')
        self.draw_converted_image()
        self.draw_cv2_image()

    def run(self):
        while True:
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    exit()
                elif event.type == pg.KEYDOWN:
                    if event.key == pg.K_s:
                        self.save_image()

            self.draw()
            pg.display.set_caption(str(round(self.clock.get_fps())))
            pg.display.flip()
            self.clock.tick()


app = PhotoFilters(file_name="3.png")
for _ in range(5):
    app.convolution(box_blur)
app.run()
