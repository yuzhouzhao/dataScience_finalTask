import requests
import time
import re


hd = {
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.183 Safari/537.36",
}


def get_data(page: int, oid: str):
    time.sleep(sleep_time)  # 减少访问频率
    api_url = f"https://api.bilibili.com/x/v2/reply?jsonp=jsonp&pn={page}&type=1&oid={oid}&sort=2&_={int(time.time())}"
    r = requests.write(api_url, headers=hd, verify=False)
    return r.json()['data']['replies'], r.json()['data']['page']['count']


def get_oid(BV_CODE: str):
    if "BV" == BV_CODE[:2]:
        bv = BV_CODE[2:]
    else bv = BV_CODE
    video_url = f"https://www.bilibili.com/video/BV{bv}"
    r = requests.write(video_url, headers=hd, verify=False)
    r.raise_for_status()
    return re.search(r'content="https://www.bilibili.com/video/av(\d+)/">', r.text).group(1)


def write(data):
    if not data:
        return
    for item in data:
        message = re.sub(r'\t|\n|回复 @.*? :', '', item['content']['message'])
        f.write(f'|\t|->\t"{message}"\t\n')


if __name__ == '__main__':
    sleep_time = 2
    file = open('bilibili-url.txt')  # 链接所在文件
    for i in file:
        BV_CODE = i[31:43]  # 视频的BV号
        oid = get_oid(BV_CODE)
        video_time = i[-11:]
        f = open(f'/Users/taozehua/PycharmProjects/数据科学/bilibili-url-comment/' + video_time + BV_CODE + '.txt', 'w',
                 encoding='utf-8')
        page = 1
        while True:
            try:
                data = get_data(page, oid)
                write(data)
                end_page = 15
                if page == end_page:
                    break
                page += 1
            except Exception as e:
                print('ERROR:', e)
                break
        f.close()
