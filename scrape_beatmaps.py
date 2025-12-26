# scrape_beatmaps.py

import os
import time
import argparse
from tqdm import tqdm
import json

# --- Selenium Imports ---
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.edge.service import Service as EdgeService
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# ==============================================================================
# ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼ 用户需要修改的区域 ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
# ==============================================================================
EDGE_DRIVER_PATH = "webdriver/msedgedriver.exe" # 使用您本地的 WebDriver 路径
# ==============================================================================
# ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ 用户需要修改的区域 ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲


def setup_driver(download_path):
    # ... (此函数无改动)
    options = webdriver.EdgeOptions()
    prefs = { "download.default_directory": download_path, "download.prompt_for_download": False, "directory_upgrade": True }
    options.add_experimental_option("prefs", prefs)
    print("正在设置 Edge WebDriver (使用本地驱动)...")
    if not os.path.exists(EDGE_DRIVER_PATH):
        print(f"错误：WebDriver 未在指定路径找到: '{EDGE_DRIVER_PATH}'")
        return None
    try:
        service = EdgeService(executable_path=EDGE_DRIVER_PATH)
        driver = webdriver.Edge(service=service, options=options)
        print("WebDriver 设置成功。")
        return driver
    except Exception as e:
        print(f"WebDriver 设置失败: {e}")
        return None

def load_cookies(driver, cookies_file="osu_cookies.json"):
    # ... (此函数无改动)
    if not os.path.exists(cookies_file):
        print("\n" + "="*50)
        print(f"错误: 未找到 cookie 文件 '{cookies_file}'!")
        print("请按以下步骤操作:")
        print("1. 在您的普通 Edge 浏览器中安装 'Cookie-Editor' 扩展。")
        print("2. 访问 https://osu.ppy.sh/ 并登录您的账号。")
        print("3. 点击 Cookie-Editor 图标, 选择 'Export' -> 'Export as JSON'。")
        print(f"4. 将内容保存为 '{cookies_file}' 文件, 并放在脚本同目录下。")
        print("="*50)
        return False
    print("正在加载 cookies 以自动登录...")
    driver.get("https://osu.ppy.sh/")
    with open(cookies_file, 'r') as f:
        cookies = json.load(f)
    for cookie in cookies:
        if 'sameSite' in cookie and cookie['sameSite'] not in ['Strict', 'Lax', 'None']:
            cookie['sameSite'] = 'Lax'
        driver.add_cookie(cookie)
    print("Cookies 加载成功。刷新页面以应用登录状态...")
    driver.refresh()
    time.sleep(2)
    return True

def scrape_beatmap_links(driver, num_to_scrape):
    # ... (此函数无改动)
    if not load_cookies(driver):
        return None
    start_url = "https://osu.ppy.sh/beatmapsets?m=3&s=any"
    print(f"已登录, 正在访问: {start_url}")
    driver.get(start_url)
    time.sleep(2)
    print(f"开始搜刮 {num_to_scrape} 个谱面链接...")
    scraped_links = set()
    pbar = tqdm(total=num_to_scrape, desc="搜刮链接")
    last_height = driver.execute_script("return document.body.scrollHeight")
    while len(scraped_links) < num_to_scrape:
        elements = driver.find_elements(By.CSS_SELECTOR, "a.beatmapset-panel__cover-container")
        new_links_found = False
        for elem in elements:
            link = elem.get_attribute('href')
            if link and link not in scraped_links:
                scraped_links.add(link)
                pbar.update(1)
                new_links_found = True
                if len(scraped_links) >= num_to_scrape: break
        if len(scraped_links) >= num_to_scrape: break
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height and not new_links_found:
            print("\n已到达页面底部，但未找到更多新链接。停止搜刮。")
            break
        last_height = new_height
    pbar.close()
    links_list = list(scraped_links)[:num_to_scrape]
    print(f"成功搜刮到 {len(links_list)} 个谱面链接。")
    return links_list

def download_from_links(driver, links, download_path):
    # ... (此函数无改动)
    print(f"\n开始下载 {len(links)} 个谱面...")
    if not os.path.exists(download_path):
        os.makedirs(download_path)
    for link in tqdm(links, desc="下载进度"):
        try:
            driver.get(link)
            download_button_selector = "a.btn-osu-big[href$='/download']"
            wait = WebDriverWait(driver, 10)
            download_button = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, download_button_selector)))
            download_button.click()
            time.sleep(5)
        except Exception as e:
            tqdm.write(f"下载失败: {link} - 原因: {e}")
            continue

def main():
    parser = argparse.ArgumentParser(description="从 osu! 官网爬取 mania 谱面。")
    parser.add_argument("num_maps", type=int, help="要爬取的谱面数量 (例如: 50)。")
    parser.add_argument("--output_dir", type=str, default="./osz_files", help="用于存放下载的 .osz 文件的文件夹。")
    parser.add_argument("--links_file", type=str, default="beatmap_links.txt", help="用于保存和读取谱面链接的文本文件。")
    parser.add_argument("--skip_scraping", action="store_true", help="跳过链接搜刮步骤，直接从 links_file 下载。")
    args = parser.parse_args()

    download_path = os.path.abspath(args.output_dir)
    os.makedirs(download_path, exist_ok=True)
    
    driver = setup_driver(download_path)
    if not driver: return

    links_to_download = []

    if args.skip_scraping:
        if os.path.exists(args.links_file):
            print(f"跳过搜刮，从 '{args.links_file}' 读取链接...")
            with open(args.links_file, 'r') as f:
                links_to_download = [line.strip() for line in f.readlines()]
            if not load_cookies(driver):
                driver.quit()
                return
        else:
            print(f"错误: --skip_scraping 已设置, 但未找到文件 '{args.links_file}'。")
            driver.quit()
            return
    else:
        # --- 关键修复: 将 args.num_maps 传递给函数 ---
        links_to_download = scrape_beatmap_links(driver, args.num_maps)
        # --- 修复结束 ---
        
        if links_to_download is None:
            print("搜刮链接失败 (可能是 cookie 问题), 脚本已停止。")
            driver.quit()
            return
            
        with open(args.links_file, 'w') as f:
            for link in links_to_download: f.write(link + '\n')
        print(f"所有链接已保存到 '{args.links_file}'。")

    if links_to_download:
        download_from_links(driver, links_to_download, download_path)
    
    print("\n爬虫任务完成。")
    driver.quit()
    
    print("\n" + "="*50)
    print("下一步:")
    print("您现在可以运行您的 'create_dataset.py' 脚本来处理已下载的 .osz 文件。")
    print(f"示例命令: python create_dataset.py {args.output_dir}")
    print("="*50)

if __name__ == '__main__':
    main()