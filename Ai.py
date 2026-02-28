import websocket
import json
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
MODEL_NAME = "llama3"
TRACKING_PARAM = "MouthOpen"

# ---------- Инициализация ----------
recognizer = sr.Recognizer()
pygame.mixer.init()

# Блокировка для синтеза речи (pyttsx3 не потокобезопасен)
tts_lock = threading.Lock()

# ---------- Класс для работы с VTube Studio ----------
class VTubeController:
    def __init__(self, uri, plugin_name, plugin_developer):
        self.uri = uri
        self.plugin_name = plugin_name
        self.plugin_developer = plugin_developer
        self.ws = None
        self.authenticated = False
        self.lock = threading.Lock()

    def connect(self):
        """Подключение и аутентификация"""
        with self.lock:
            if self.ws:
                try:
                    self.ws.close()
                except:
                    pass
            try:
                self.ws = websocket.create_connection(self.uri, timeout=5)
            except Exception as e:
                print(f"Ошибка подключения: {e}")
                return False

            # Запрос токена
            token_req = {
                "apiName": "VTubeStudioPublicAPI",
                "apiVersion": "1.0",
                "requestID": "token",
                "messageType": "AuthenticationTokenRequest",
                "data": {
                    "pluginName": self.plugin_name,
                    "pluginDeveloper": self.plugin_developer
                }
            }
            self.ws.send(json.dumps(token_req))
            resp = json.loads(self.ws.recv())
            if resp.get("messageType") != "AuthenticationTokenResponse":
                print("Ошибка получения токена:", resp)
                return False
            token = resp["data"]["authenticationToken"]

            # Аутентификация
            auth_req = {
                "apiName": "VTubeStudioPublicAPI",
                "apiVersion": "1.0",
                "requestID": "auth",
                "messageType": "AuthenticationRequest",
                "data": {
                    "pluginName": self.plugin_name,
                    "pluginDeveloper": self.plugin_developer,
                    "authenticationToken": token
                }
            }
            self.ws.send(json.dumps(auth_req))
            auth_resp = json.loads(self.ws.recv())
            if auth_resp.get("data", {}).get("authenticated"):
                self.authenticated = True
                print("✅ Подключено к VTube Studio")
                return True
            else:
                print("Ошибка аутентификации:", auth_resp)
                return False

    def set_parameter(self, param_name, value):
        """Отправить значение параметра, при необходимости переподключиться"""
        with self.lock:
            if not self.ws or not self.authenticated:
                if not self.connect():
                    return False
            try:
                cmd = {
                    "apiName": "VTubeStudioPublicAPI",
                    "apiVersion": "1.0",
                    "requestID": "set",
                    "messageType": "InjectParameterDataRequest",
                    "data": {
                        "parameterValues": [{"id": param_name, "value": value}]
                    }
                }
                self.ws.send(json.dumps(cmd))
                _ = self.ws.recv()  # игнорируем ответ
                return True
            except Exception as e:
                print(f"Ошибка отправки параметра: {e}")
                self.authenticated = False
                return False

    def close(self):
        with self.lock:
            if self.ws:
                try:
                    self.ws.close()
                except:
                    pass
                self.ws = None
                self.authenticated = False

# ---------- Подключаемся к VTS ----------
vts = VTubeController(VTS_WS, PLUGIN_NAME, PLUGIN_DEVELOPER)
if not vts.connect():
    print("Не удалось подключиться к VTube Studio. Запустите VTS и включите API.")
    sys.exit(1)

# ---------- Анимация рта ----------
def animate_mouth(duration):
    end = time.time() + duration
    while time.time() < end:
        if not vts.set_parameter(TRACKING_PARAM, 0.8):
            break
        time.sleep(0.15)
        if not vts.set_parameter(TRACKING_PARAM, 0.0):
            break
        time.sleep(0.15)
    vts.set_parameter(TRACKING_PARAM, 0.0)

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
        vts.close()
        pygame.mixer.quit()
        print("Программа завершена")