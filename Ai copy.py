import websocket
import json
import socket
import speech_recognition as sr
import pyttsx3
import ollama
import threading
import time
import pygame
import os
import sys



# ---------- Настройки ----------
VTS_WS = "ws://localhost:8001"
PLUGIN_NAME = "MyAssistantPlugin"
PLUGIN_DEVELOPER = "MyName"
MODEL_NAME = "deepseek-r1:8b"
TRACKING_PARAM = "MouthOpen"

# ---------- Инициализация ----------
recognizer = sr.Recognizer()
pygame.mixer.init()

# Блокировка для синтеза речи (pyttsx3 не потокобезопасен)
tts_lock = threading.Lock()
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)


# ---------- Анимация рта ----------
def animate_mouth(duration):
    end = time.time() + duration
    while time.time() < end:
        sock.sendto(b"ParamMouthOpenY:0.8", ("127.0.0.1", 12345))
        time.sleep(0.15)
        sock.sendto(b"ParamMouthOpenY:0.0", ("127.0.0.1", 12345))           
        time.sleep(0.15)
    sock.sendto(b"ParamMouthOpenY:0.0", ("127.0.0.1", 12345))

# ---------- Запрос к Ollama ----------
def ask_ollama(prompt):
    try:
        response = ollama.chat(model=MODEL_NAME, messages=[{"role": "user", "content": prompt}])
        return response['message']['content']
    except Exception as e:
        print("Ошибка Ollama:", e)
        return "Извините, я не могу ответить."

# ---------- Озвучка и анимация ----------
def speak_and_animate(answer):
    temp_file = "temp.wav"

    # 1. Синтез речи (один поток за раз)
    with tts_lock:
        # Создаём новый движок для каждого вызова
        engine = pyttsx3.init()
        # Настроим голос (можно закрепить индекс, если нужно)
        voices = engine.getProperty('voices')
        if voices:
            engine.setProperty('voice', voices[0].id)
        try:
            engine.save_to_file(answer, temp_file)
            engine.runAndWait()
        except Exception as e:
            print("Ошибка синтеза речи:", e)
            return
        finally:
            try:
                engine.stop()
            except:
                pass

    # 2. Проверка файла
    if not os.path.exists(temp_file) or os.path.getsize(temp_file) == 0:
        print("Файл речи не создан или пуст")
        return

    # 3. Загрузка в pygame
    try:
        sound = pygame.mixer.Sound(temp_file)
    except pygame.error as e:
        print("Ошибка загрузки звука:", e)
        # Перезапустим микшер
        pygame.mixer.quit()
        time.sleep(0.2)
        pygame.mixer.init()
        try:
            sound = pygame.mixer.Sound(temp_file)
        except pygame.error as e2:
            print("Не удалось загрузить звук:", e2)
            return

    # 4. Удаляем временный файл
    try:
        os.remove(temp_file)
    except:
        pass

    # 5. Запуск анимации
    duration = max(len(answer) / 10, 1.5)
    threading.Thread(target=animate_mouth, args=(duration,), daemon=True).start()

    # 6. Воспроизведение
    channel = sound.play()
    if channel:
        while channel.get_busy():
            time.sleep(0.1)
    else:
        print("Не удалось воспроизвести звук")

# ---------- Основной цикл ----------
def main():
    print("Ассистент запущен. Введите текст (Enter для отправки).")
    while True:
        try:
            user_text = input("Вы: ")
            if not user_text:
                continue
            if user_text.lower() in ("выход", "exit", "quit"):
                break

            answer = ask_ollama(user_text)
            print(f"Ассистент: {answer}")
            threading.Thread(target=speak_and_animate, args=(answer,), daemon=True).start()

        except KeyboardInterrupt:
            break
        except Exception as e:
            print("Ошибка:", e)

if __name__ == "__main__":
    try:
        main()
    finally:
        pygame.mixer.quit()
        print("Программа завершена")