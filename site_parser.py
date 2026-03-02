import requests
from bs4 import BeautifulSoup
import time
import csv
from urllib.parse import urlparse
import xml.etree.ElementTree as ET

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}
DELAY = 2  # Соблюдаем вежливость
OUTPUT_CSV = "genshin_articles.csv"
SITEMAP_INDEX = "https://wotpack.ru/sitemap.xml"
TARGET_CATEGORY = "genshin"

def fetch_url(url):
    """Безопасно загружает URL с обработкой ошибок."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        return resp
    except Exception as e:
        print(f"Ошибка загрузки {url}: {e}")
        return None

def get_all_post_urls_from_sitemaps():
    """Собирает все URL статей из всех post-sitemap.xml файлов."""
    print(f"Загружаем индексную карту: {SITEMAP_INDEX}")
    resp = fetch_url(SITEMAP_INDEX)
    if not resp:
        return []

    # Парсим индексную карту, чтобы найти все post-sitemap
    root = ET.fromstring(resp.content)
    # Пространство имен sitemap
    ns = {'sm': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
    sitemap_urls = []
    for sitemap in root.findall('sm:sitemap/sm:loc', ns):
        sitemap_urls.append(sitemap.text)

    # Фильтруем только карты с постами
    post_sitemaps = [url for url in sitemap_urls if 'post-sitemap' in url]
    print(f"Найдено карт с постами: {len(post_sitemaps)}")

    all_post_urls = []
    for sitemap_url in post_sitemaps:
        print(f"Загружаем: {sitemap_url}")
        resp_sm = fetch_url(sitemap_url)
        if not resp_sm:
            continue
        root_sm = ET.fromstring(resp_sm.content)
        for url_elem in root_sm.findall('sm:url/sm:loc', ns):
            all_post_urls.append(url_elem.text)
        time.sleep(DELAY)  # Задержка между загрузками карт
    print(f"Всего собрано URL статей: {len(all_post_urls)}")
    return all_post_urls

def parse_article(article_url):
    """Загружает статью и извлекает заголовок и текст."""
    resp = fetch_url(article_url)
    if not resp:
        return None, None
    soup = BeautifulSoup(resp.text, 'html.parser')

    # 1. Пытаемся получить заголовок из meta og:title
    og_title = soup.find('meta', property='og:title')
    title = og_title['content'] if og_title and og_title.get('content') else None

    # 2. Если нет og:title, ищем h1 с классом entry-title или просто h1
    if not title:
        title_tag = soup.select_one('h1.entry-title') or soup.find('h1')
        title = title_tag.get_text(strip=True) if title_tag else None

    # 3. Если всё ещё нет, ставим заглушку
    if not title:
        title = "Без заголовка"

    # Текст статьи (как и раньше)
    content_tag = soup.find('div', class_='entry-content')
    if content_tag:
        for unwanted in content_tag.select('.sharedaddy, .jp-relatedposts, .code-block, .yarpp-related'):
            unwanted.decompose()
        text = content_tag.get_text(separator='\n', strip=True)
    else:
        text = ""

    return title, text

def main():
    # Шаг 1: Получаем все ссылки на статьи
    all_urls = get_all_post_urls_from_sitemaps()

    # Шаг 2: Фильтруем только по Genshin Impact
    genshin_urls = [url for url in all_urls if TARGET_CATEGORY in url]
    print(f"Найдено статей по Genshin Impact: {len(genshin_urls)}")

    # Шаг 3: Сохраняем список ссылок (на случай обрыва)
    with open("genshin_links.txt", "w", encoding="utf-8") as f:
        for url in genshin_urls:
            f.write(url + "\n")

    # Шаг 4: Парсим каждую статью
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["URL", "Заголовок", "Текст"])

        for idx, url in enumerate(genshin_urls, 1):
            if(idx > 100):
                break
            print(f"[{idx}/{len(genshin_urls)}] Парсинг: {url}")
            title, text = parse_article(url)
            if title and text:
                writer.writerow([url, title, text])
            else:
                print(f"  -> Не удалось распарсить")
            time.sleep(DELAY)

    print(f"Готово! Результат в файле {OUTPUT_CSV}")

if __name__ == "__main__":
    main()