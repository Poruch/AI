# -*- coding: utf-8 -*-
import math
import time
import threading
import socket

import win32gui
import win32con
import win32api

import pygame
from pygame.locals import *
from OpenGL.GL import *

import live2d.v3 as live2d
if live2d.LIVE2D_VERSION == 3:
    from live2d.v3 import StandardParams
else:
    from live2d.v2 import StandardParams

from live2d.utils import log

live2d.enableLog(True)
live2d.setLogLevel(live2d.Live2DLogLevels.LV_DEBUG)

# ========== UDP настройки ==========
UDP_IP = "127.0.0.1"
UDP_PORT = 12345
udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
udp_socket.bind((UDP_IP, UDP_PORT))
udp_socket.setblocking(False)
command_queue = []

def udp_server_thread():
    global command_queue
    while True:
        try:
            data, addr = udp_socket.recvfrom(1024)
            if data:
                command = data.decode('utf-8').strip()
                print(f"[UDP] Получено: {command}")
                command_queue.append(command)
        except socket.error:
            time.sleep(0.01)
        except Exception as e:
            print(f"[UDP] Ошибка: {e}")
            time.sleep(0.1)

# ========== Основная функция ==========
def main():
    pygame.init()
    live2d.init()

    display = (500, 600)

    # 1. Запрашиваем альфа-канал ДО создания окна
    pygame.display.gl_set_attribute(pygame.GL_ALPHA_SIZE, 8)

    # 2. Создаём окно без рамок, с OpenGL
    screen = pygame.display.set_mode(display, DOUBLEBUF | OPENGL | NOFRAME)

    # --- Настройка прозрачного окна через WinAPI ---
    hwnd = pygame.display.get_wm_info()["window"]  # Получаем HWND окна

    # Получаем текущие расширенные стили окна и добавляем флаг WS_EX_LAYERED
    ex_style = win32gui.GetWindowLong(hwnd, win32con.GWL_EXSTYLE)
    win32gui.SetWindowLong(hwnd, win32con.GWL_EXSTYLE, ex_style | win32con.WS_EX_LAYERED)

    # Устанавливаем прозрачность для цвета-ключа. 
    # В примере используется RGB(255,0,128) - ярко-розовый. Этот цвет станет полностью прозрачным.
    win32gui.SetLayeredWindowAttributes(hwnd, win32api.RGB(255, 0, 128), 0, win32con.LWA_COLORKEY)

    pygame.display.set_caption("Live2D оверлей")

    # 3. Получаем SDL-окно для дополнительных параметров
    import pygame._sdl2.video as sdl2_video
    sdl_window = sdl2_video.Window.from_display_module()
    sdl_window.borderless = False

    # !!! Для проверки прозрачности окна раскомментируйте следующую строку !!!
    # sdl_window.opacity = 0.5   # сделает окно полупрозрачным (весь контент)

    # 4. Настраиваем OpenGL смешивание
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    # 5. Инициализация live2d OpenGL (только для версии 3)
    if live2d.LIVE2D_VERSION == 3:
        live2d.glInit()

    # 6. Загружаем модель
    model = live2d.LAppModel()
    model_path = "Usa Maid/Usa Maid.model3.json"
    if live2d.LIVE2D_VERSION == 3:
        model.LoadModelJson(model_path, maskBufferCount=100)
    else:
        model.LoadModelJson(model_path)
    model.Resize(*display)

    # Переменные для управления
    dx, dy, scale = 0.0, 0.0, 1.0
    model.SetAutoBlinkEnable(False)
    model.SetAutoBreathEnable(False)

    partIds = model.GetPartIds()
    currentTopClickedPartId = None

    def getHitFeedback(x, y):
        hitPartIds = model.HitPart(x, y, False)
        if currentTopClickedPartId is not None:
            pidx = partIds.index(currentTopClickedPartId)
            model.SetPartOpacity(pidx, 1)
            model.SetPartMultiplyColor(pidx, 1.0, 1.0, 1., 1)
        if hitPartIds:
            return hitPartIds[0]
        return None

    model.StartRandomMotion("TapBody", 300)

    radius_per_frame = math.pi * 10 / 1000 * 0.5
    deg_max = 5
    progress = 0

    print("размер холста:", model.GetCanvasSize())
    print("размер холста в пикселях:", model.GetCanvasSizePixel())
    print("пикселей на единицу:", model.GetPixelsPerUnit())

    threading.Thread(target=udp_server_thread, daemon=True).start()
    print(f"UDP-сервер слушает {UDP_IP}:{UDP_PORT}")

    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break
            if event.type == pygame.MOUSEBUTTONDOWN:
                model.SetRandomExpression()
                model.StartRandomMotion(priority=3)
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    dx -= 0.1
                elif event.key == pygame.K_RIGHT:
                    dx += 0.1
                elif event.key == pygame.K_UP:
                    dy += 0.1
                elif event.key == pygame.K_DOWN:
                    dy -= 0.1
                elif event.key == pygame.K_i:
                    scale += 0.1
                elif event.key == pygame.K_u:
                    scale -= 0.1
                elif event.key == pygame.K_r:
                    model.StopAllMotions()
                    model.ResetPose()
                elif event.key == pygame.K_e:
                    model.ResetExpression()
            if event.type == pygame.MOUSEMOTION:
                model.Drag(*pygame.mouse.get_pos())
                currentTopClickedPartId = getHitFeedback(*pygame.mouse.get_pos())

        # Обработка UDP-команд
        while command_queue:
            cmd = command_queue.pop(0)
            parts = cmd.split(':', 1)
            if len(parts) == 2:
                param_name = parts[0].strip()
                try:
                    value = float(parts[1].strip())
                    model.SetParameterValue(param_name, value, 1.0)
                    print(f"[модель] Параметр {param_name} = {value}")
                except ValueError:
                    print(f"[ошибка] Неверное значение: {parts[1]}")
            else:
                print(f"[ошибка] Неверный формат команды: {cmd}")

        # Анимация покачивания
        progress += radius_per_frame
        deg = math.sin(progress) * deg_max
        model.Rotate(deg)
        model.Update()

        # Подсветка части под курсором
        if currentTopClickedPartId is not None:
            pidx = partIds.index(currentTopClickedPartId)
            model.SetPartOpacity(pidx, 0.5)
            model.SetPartMultiplyColor(pidx, 0.0, 0.0, 1.0, 0.9)

        model.SetOffset(dx, dy)
        model.SetScale(scale)

        # ОЧИСТКА БУФЕРА С АЛЬФА-КАНАЛОМ 0 (прозрачный фон)
        glClearColor(255.0/255.0, 0.0/255.0, 128.0/255.0, 1.0)  # Альфа здесь не важна
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)  

        # Рисуем модель
        model.Draw()

        pygame.display.flip()
        pygame.time.wait(10)

    live2d.dispose()
    pygame.quit()
    quit()

if __name__ == "__main__":
    main()