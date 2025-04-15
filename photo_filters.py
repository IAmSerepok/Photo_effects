import pygame as pg

import pygame.gfxdraw
import cv2

from copy import deepcopy

from effect_functions import *

from typing import Dict, Tuple, Optional


class PhotoFilters:
    """Класс для применения фильтров к изображениям.

    Attributes:
        input_path (str): Путь к исходному изображению.
        image (numpy.ndarray): Основное изображение в формате RGB (width, height, 3).
        next_image (numpy.ndarray): Буфер для обработки изображения.
        screen_size (Tuple[int, int]): Размеры экрана (ширина, высота).
        surface (pygame.Surface): Основная поверхность для отрисовки.
        clock (pygame.time.Clock): Таймер для контроля FPS.
        cv2_image (numpy.ndarray): Копия изображения в формате OpenCV BGR.
    """

    def __init__(self, input_path: str) -> None:
        """Инициализирует обработчик изображений.

        Args:
            input_path (str): Путь к исходному изображению.
        """
        pg.init()
        self.input_path = input_path
        self.image, self.cv2_image = self.__get_image()
        self.next_image = self.image
        self.screen_size = self.screen_width, self.screen_height = self.image.shape[0], self.image.shape[1]
        self.surface = pg.display.set_mode(self.screen_size)
        self.clock = pg.time.Clock()

    def save_image(self, output_path: str) -> None:
        """Сохраняет текущее изображение в файл.

        Args:
            output_path (str): Путь для сохранения обработанного изображения.
        """
        pygame_image = pg.surfarray.array3d(self.surface)
        cv2_img = cv2.transpose(pygame_image)
        cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
        cv2.imwrite(output_path, cv2_img)

    def convolution(self, function: str, n_iter: int = 1) -> None:
        """Применяет выбранный фильтр к изображению.

        Args:
            function (str): Название фильтра из доступных:
                - 'box_blur' - размытие по квадрату 3x3
                - 'gaussian_blur' - гауссово размытие
                - 'clarifier' - усиление доминирующего канала
                - 'dimmer' - затемнение
            n_iter (int): Количество итераций применения фильтра.

        Raises:
            KeyError: Если указано неверное имя фильтра.
        """
        name_to_func = {
            'box_blur': box_blur, 
            'gaussian_blur': gaussian_blur, 
            'clarifier': clarifier, 
            'dimmer': dimmer
        }

        for _ in range(n_iter):
            for x in range(1, self.screen_width - 1):
                for y in range(1, self.screen_height - 1):
                    self.next_image[x, y] = name_to_func[function](self.image, x, y)
            self.image = deepcopy(self.next_image)

    def run(self) -> None:
        """Запускает основной цикл приложения.
        
        Обрабатывает события:
        - Закрытие окна (выход из программы)
        - Нажатие 'S' для сохранения изображения
        """
        while True:
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    exit()
                elif event.type == pg.KEYDOWN:
                    if event.key == pg.K_s:
                        self.save_image("output.jpg")

            self.__draw()
            pg.display.set_caption(str(round(self.clock.get_fps())))
            pg.display.flip()
            self.clock.tick()

    def __draw_converted_image(self) -> None:
        """Отрисовывает текущее изображение на pygame поверхности."""
        for x in range(0, self.screen_width):
            for y in range(0, self.screen_height):
                color = tuple(self.image[x, y])
                pygame.gfxdraw.box(self.surface, (x, y, 1, 1), color)

    def __get_image(self) -> Tuple[ndarray, ndarray]:
        """Загружает и преобразует изображение.

        Returns:
            Кортеж из:
            - изображения в формате RGB (numpy array)
            - изображения в формате BGR (OpenCV format)
        """
        cv2_image = cv2.imread(self.input_path)
        transposed_image = cv2.transpose(cv2_image)
        image = cv2.cvtColor(transposed_image, cv2.COLOR_BGR2RGB)
        return image, cv2_image

    def __draw_cv2_image(self) -> None:
        """Отображает оригинальное изображение через OpenCV."""
        resized_cv2_image = cv2.resize(self.cv2_image, (self.screen_width//2, self.screen_height//2),
                                     interpolation=cv2.INTER_AREA)
        cv2.imshow('img', resized_cv2_image)

    def __draw(self) -> None:
        """Основная функция отрисовки."""
        self.surface.fill('black')
        self.__draw_converted_image()
        self.__draw_cv2_image()
