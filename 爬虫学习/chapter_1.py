'''
http请求:请求行 , 请求头 , 请求体;
http响应:状态行 , 响应头 , 响应体;
状态码： 2 成功  3 重定向  4 客户端错误 5 服务器错误
'''
import requests
from bs4 import BeautifulSoup
# 有些网站为了防止数据爬取，就会检查 User-Agent属性（告诉服务器请求是由什么发出来的 浏览器或其他，版本，系统等），
# requests请求默认不带这个参数，所以我们要伪造（copy本地浏览器的User-Agent即可）
for i in range(0, 250 , 25):
    headers =  {
        'User-Agent' :'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36 Edg/114.0.1823.43'
    }
    concent = requests.get(f'https://movie.douban.com/top250?start={i}',headers=headers)
    soup = BeautifulSoup(concent.text, "html.parser")
    all_name = soup.findAll( "span" , attrs={"class":"title"})
    for name in all_name:
        # print(name) # 结果 :<span class="title">肖申克的救赎</span>
        # print(name.string) # 结果：肖申克的救赎
        name1 = name.string
        if '/' not in name1:
            print(name.string)
   