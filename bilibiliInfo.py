import requests
from bs4 import BeautifulSoup

def fetch_latest_videos():
    url = 'https://www.bilibili.com/v/popular/all'  # Bilibili 视频热门页面
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    # 发送 GET 请求获取网页内容
    response = requests.get(url, headers=headers)

    # 检查请求是否成功
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')

        # 查找视频信息
        video_items = soup.find_all('div', class_='bili-video-card__info')

        latest_videos = []

        # 遍历找到的视频并提取标题和链接
        for item in video_items:
            title = item.find('h3', class_='bili-video-card__info--tit').get_text(strip=True)
            link = 'https:' + item.find('a', class_='bili-video-card__info--tit').get('href')
            views = item.find('span', class_='bili-video-card__stats--view').get_text(strip=True)

            # 保存视频信息
            latest_videos.append({
                'title': title,
                'link': link,
                'views': views,
            })

        # 输出最新视频信息
        for video in latest_videos:
            print(f"Title: {video['title']}")
            print(f"Link: {video['link']}")
            print(f"Views: {video['views']}")
            print('-' * 40)

    else:
        print(f"Failed to retrieve the page. Status code: {response.status_code}")

if __name__ == '__main__':
    fetch_latest_videos()

# 运行函数，抓取 Bilibili 最新视频
