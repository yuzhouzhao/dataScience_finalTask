# 疫情下的网络社会及公众心态研究

## 目录

[TOC]

## 01-成员基本信息

---

|  姓名  |   学号    |    电话     |                   分工                   |
| :----: | :-------: | :---------: | :--------------------------------------: |
| 赵宇舟 | 191250204 | 18851865055 |  数据内容爬取、机器学习与模型训练、nlp   |
| 林正顺 | 191250088 | 15371077531 |     数据链接爬取、信息检索、概统分析     |
| 陶泽华 | 191250133 | 19850355091 | 数据内容爬取、数据筛选与分类、数据可视化 |

## 02-绪论

***

### 2.1-摘要





**关键字**：疫情，心态分析，python，数据爬取，机器学习，朴素贝叶斯，NLP，数据可视化

### 2.2-研究背景

> ​		中国社会正处在深刻而快速的转型期，其中，在社会变迁层面，社会结构的快速分化，以“撕裂”的方式强化了社会团体、阶层 之间的张力，使得整体社会结构出现紧张(李汉林、魏钦恭、张彦, 2010)，并投射在个体心理层面，进一步凸显出公众的社会认知、 情绪、信念、意向、行动等对社会治理的重要影响(王俊秀, 2014;杨宜音, 2006) 。 同时，随着互联网应用的不断普及，日益多元复杂的公众情绪，借助网络的力量传播和放大，对社会心态的塑形力量进一步增强，赋予了群体心理及集体行为的极化可能(周晓虹，2014)。 当下新型冠状病毒(COVID-19) 肆虐全球，给人们的生产和生活产生了极大影响，也形成了疫情下独特的网络社会心态 和公众情绪。 因此，立足此次新型冠状病毒(COVID-19)重大突发公共卫生事件情境，借助适宜的数据与计量手段，准确并客观地 了解公众的网络社会心态与基于此呈现出的行为规律，就可能实现公众的情绪引导，让大众以积极的心态与政府一起应对和处理公共卫生事件及其衍生问题，维护国家与社会的长治久安。
>

### 2.3-代码开源地址

**GitHub:** [dataScience_finalTask](https://github.com/yuzhouzhao/dataScience_finalTask.git) 

**数据可视化: **[https://linzs148.github.io/Visualization/](https://linzs148.github.io/Visualization/)（建议使用Chrome浏览器）

## 03-数据获取及初步处理

---

### 3.1-方法

利用python和一系列库进行新闻、b站评论和微博评论的爬取

### 3.2-过程

分三步进行：

1. 新闻链接爬取+新闻内容爬取
2. bilibili视频链接爬取+bilibili视频评论爬取
3. 微博评论爬取

#### 新闻链接爬取+新闻内容爬取：

#### bilibili视频链接爬取+bilibili视频评论爬取：

​    	根据"疫情", "新冠", "抗疫", "口罩", "病例", "钟南山", "防疫", "火神山", "雷神山“，”肺炎“等一系列关键词，同时借助requests库、BeautifulSoup库来筛选对应上述关键词的链接，并将结果存入了bilibili.txt文件中。再根据获得的链接获得视频的BV号去爬取视频的评论。

#### 微博评论爬取：

  	首先用cookies模拟登陆网页版新浪微博，然后根据指定的官方微博和指定的微博时间对应的微博链接来抓取该时间段的前15条微博的评论。

### 3.3-代码解释

#### 新闻链接爬取+新闻内容爬取代码：

#### bilibili视频链接爬取+bilibili视频评论爬取代码：

```python
# b站视频链接爬取代码
keywords = ["疫情", "新冠", "抗疫", "口罩", "病例", "钟南山", "防疫", "火神山", "雷神山", "肺炎"]

# 根据设定的关键词来获得b站相关视频的链接
def getUrls(keyword, page):
    url = "https://search.bilibili.com/all?order=click"
    # 设定关键词
    params = {'keyword': keyword, 'page': page}
    # 爬虫常规代码
    response = requests.get(url, params=params, headers=headers)
    response.raise_for_status()
    response.encoding = response.apparent_encoding
    soup = BeautifulSoup(response.text, "html.parser")
    lis = soup.find_all('li', {'class': "video-item matrix"})
    
    for li in lis:
        attributes = li.a.attrs
        spans = li.find_all('span')
        # 将得到的链接和时间写入文件
        with open('bilibili-url.txt', 'a') as f:
            f.writelines('https:' + attributes['href'] + " " + spans[5].text.strip() + '\n')
```

```python
# b站视频评论爬取代码
if __name__ == '__main__':
    sleep_time = 2
    file = open('bilibili-url.txt')  # 视频链接所在文件
    for i in file:
        BV_CODE = i[31:43]  # 从链接中截取得到视频的BV号
        oid = get_oid(BV_CODE) # 获得视频的oid号
        video_time = i[-11:] # 从链接中截取得到视频发表的时间
        f = open(f'/Users/taozehua/PycharmProjects/数据科学/bilibili-url-comment/' + video_time + BV_CODE + '.txt', 'w',
                 encoding='utf-8')
        page = 1
        while True:
            try:
                data = get_data(page, oid)
                write(data) #向文件写入数据
                end_page = 15 # 设定连续读取15页的评论
                if page == end_page:
                    break
                page += 1
            except Exception as e:
                print('ERROR:', e)
                break
        f.close()
```

#### 微博评论爬取代码：

```python
# 设置cookies 模拟登陆网页版微博，cookies由本机登录时获得
self.headers = {
            "cookie": "login_sid_t=95e31daa102176d1debb61e844641c26; cross_origin_proto=SSL; _s_tentry=passport.weibo.com; Apache=8896751903602.45.1611205480375; SINAGLOBAL=8896751903602.45.1611205480375; ULV=1611205480379:1:1:1:8896751903602.45.1611205480375:; wb_view_log_5688140475=1440*9002; wb_view_log=1440*9002; WBStorage=8daec78e6a891122|undefined; crossidccode=CODE-tc-1L2uul-2O9Vrn-3NXFKeQ59aniHK8bfdc35; SSOLoginState=1611214485; SUB=_2A25NDV7FDeRhGeNI41oQ9C7IzDmIHXVuDmKNrDV8PUJbkNAKLWTskW1NSDShAmgWg52xu-LVa7tA1onXiM57xVOs; SUBP=0033WrSXqPxfM725Ws9jqgMF55529P9D9WW0TQIzrxRpmDardhs5o9p15NHD95QfSonReKB7ShMfWs4DqcjQi--ciK.RiKLsi--Ri-8si-82i--fi-isiKn0i--ciKnXi-isxsHLM5tt; wvr=6; UOR=,,graph.qq.com; webim_unReadCount=%7B%22time%22%3A1611214550618%2C%22dm_pub_total%22%3A4%2C%22chat_group_client%22%3A0%2C%22chat_group_notice%22%3A0%2C%22allcountNum%22%3A42%2C%22msgbox%22%3A0%7D",
            "user-agent": "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/72.0.3626.96 Safari/537.36",
        }



def parse_comment_info(self, url):
    # 爬取评论和评论的时间
    res = requests.write(url, headers=self.headers)
    response = res.json()
    count = response['data']['count']
    html = etree.HTML(response['data']['html'])
    name = html.xpath("//div[@class='list_li S_line1 clearfix']/div[@class='WB_face W_fl']/a/img/@alt")  # 解析评论人的姓名
    info = html.xpath("//div[@node-type='replywrap']/div[@class='WB_text']/text()")  # 解析评论信息
    info = "".join(info).replace(" ", "").split("\n")
    comment_time = html.xpath("//div[@class='WB_from S_txt2']/text()")  # 解析评论时间
    comment_info_list = []
    # 将评论和时间存入
    for i in range(len(name)):
        item = {}
        # 控制存进去的评论数量，过滤不需要的信息
        if info[i] != "：转发微博" and info[i] != "：" and i <= 6:
            item["comment"] = info[i]  # 存储评论的信息
            item["comment_time"] = comment_time[i]  # 存储评论时间
            comment_info_list.append(item)
        else:   break
    return count, comment_info_list
```

### 3.4-成果展示

#### 新闻内容：

​		获得了新浪网2019年12月至2020年12月之间的所有与新冠疫情有关的新闻，并且以日期为划分存储。

![新闻](./finalReportImages/新闻.png)

#### b站评论：

​		获得了b站2019年12月至2020年12月之间的所有与新冠疫情有关的视频的评论，并且以日期为划分存储。

![b站评论](./finalReportImages/b站评论.png)

#### 微博评论：

​		获得了新浪微博2019年12月至2020年6月之间的一些官方微博的关于疫情报道的评论。

![微博评论](./finalReportImages/微博评论.png)

## 04-数据分析及模型训练

---

### 4.1-方法

​		文本情感分析是指用自然语言处理（NLP）、文本挖掘以及计算机语言学等方法对带有情感色彩的主观性文本进行分析、处理、归纳和推理的过程。 

> **维基中文百科：**文本情感分析（也称为意见挖掘）是指用自然语言处理、文本挖掘以及计算机语言学等方法来识别和提取原素材中的主观信息。

​	    我们在本次作业中做的是**基于机器学习的情感分析**，我们将情感分析视为一个二分类的问题，采用机器学习的方法识别。先将文本分词后，选择特征词后转化为词向量，进而将文本矩阵化，利用朴素贝叶斯（Naive Bayes），支持向量机（SVM）等算法进行分类。最终训练得到一个基于机器学习的分类器模型，而模型的分类效果取决于训练文本的选择以及正确的情感标注。

#### 用到的工具：

> - NLTK库（natural language toolkit）：是一套基于python的自然语言处理工具集。
> - Sklearn库（ Scikit-learn ）：机器学习中最简单高效的数据挖掘和数据分析工具。
> - Snownlp库：一个处理中文文本的 Python 类库。

### 4.2-过程

一个完整的机器学习项目一般流程包括：

1. 数据获取及分析
2. 数据预处理
3. 特征工程
4. 训练模型选择与调优
5. 模型评估

#### 分词

​		分词即将书面文本分割成有意义单位的过程，这里的有意义单位即“词”，中文与英文不同，英文天然地以单词为单位组成句子，每个单词之间有空格为分割，而中文则需要另外的处理。中文分词方法：

- jieba库：一个基于Python的中文分词的组件
- Snownlp.seg：Snownlp库也有分词的功能**（我们使用的）**

#### 停用词处理

> **维基中文百科：**在信息检索中，为节省存储空间和提高搜索效率，在处理自然语言数据（或文本）之前或之后会自动过滤掉某些字或词，这些字或词即被称为Stop Words(停用词)。

​		语言包含很多功能词。与其他词相比，功能词没有什么实际含义。如中文中的呢，呐，阿，哎等，这些词很少单独表达文档相关程度的信息，因此，在训练模型前，可以将此类词语去除。
​		常用的中文停用词表：哈工大停用词词库，四川大学机器学习智能实验室停用词库，百度停用词表等，每张停用词表都大同小异，随意选择一份即可

#### 标签

​		情感分析是文本训练的一种，属于**监督学习**，所以需要整理样本，即人工为训练集打标签。根据大作业需求，我们确定了样本的标签，样本标签为整数的原因是为了方便量化。在大作业中，我们的心态分级是：

- 积极（-1）

- 中立（0）

- 消极（+1）

  值得注意的是，我们在机器学习的训练过程中只使用了正样本和负样本，即`pos.txt`和`neg.txt` 

#### 机器学习模型选择

​		我们将本次大作业的情感预测视为一个二分问题，因此最适用的两个机器学习算法便是**支持向量机（SVM）**以及**朴素贝叶斯（Naive Bayes）**。而在sklearn和nltk库中，均有支持向量机和朴素贝叶斯模型的接口，为找到最佳的机器学习算法，我们在实验过程中同时实验了两种不同的模型，并最终选择了**朴素贝叶斯**。主要的参考资料有：《机器学习》周志华著，各类blog、知乎、CSDN等。

##### 支持向量机（SVM）

> **支持向量机**（Support Vector Machine, SVM）是一类按监督学习方式对数据进行二元分类的广义线性分类器。其被广泛应用于机器学习(Machine Learning), 计算机视觉(Computer Vision) 和数据挖掘(Data Mining)当中。

​	支持向量机是一类经典的**监督学习分类器**，SVM有三种学习模型（如下图，由上到下由简到繁），由于还是机器学习的初学者，我们选择学习了解最简单的一部分的SVM，即线性可分支持向量机：硬间隔。

<img src="./finalReportImages/svm.png" alt="svm" style="zoom:60%;" />

**支持向量机的原理**

​		给定训练样本集D={(x1,y1),(x2,y2),...,(xm,ym)}，yi∈{−1,+1}，分类学习最基本的想法就是**基于训练集D在样本空间中找到一个划分超平面**，将不同类别的样本分开。但能将训练样本分开的划分**超平面**可能有很多，如图所示，那么应该选取哪一个呢？直观来看，我们选择的就是“最中间”的那一条，SVM便可以帮助我们找到这条划分。
​		而对于一维数据的划分，我们只需要找到一个点；对于二维数据的划分，我们需要找到一条线；对于三维数据的划分，我们需要找到一个面；而对于多维的数据，我们则需要找到一个“超平面”，超平面是平面中的直线、空间中的平面之推广，是纯粹的数学概念，不是现实的物理概念。

<img src="./finalReportImages/svm2.png" alt="svm2" style="zoom:50%;" />

> **超平面**：超平面是n维欧氏空间中余维度等于一的线性子空间，也就是必须是(n-1)维度。因为是子空间，所以超平面一定经过原点。

​		通过给定的线性可分训练集T，学习得到分类的超平面：$wx+b=0$ ，从而得到决策函数：$f(x)=sign(w^Tx+b)$ ，称决策函数为线性可分支持向量机。假设超平面能训练样本正确分类，则对于 $(x_i,y_i)\in D$ 有：

- 如果 $w^Tx_i+b\geq+1,y_i=+1$ 
- 如果 $w^Tx_i+b\leq-1,y_i=-1$ 

​        我们通过求“间隔”最大的超平面来确定参数w, b：即在$y_i(wx_i+b)\geq1$（等价于上面的不等式组）的约束条件下，$max_{(w,b)}\frac1{||w||}$的最大值。

**支持向量机的优缺点：**

- 优点

  - SVM在小样本训练集上能够得到比其它算法好很多的结果。
  - 算法原理简单

- 缺点

  - SVM在大样本训练集上的表现不尽人意。
  - 如果数据量很大，SVM的训练时间就会比较长。

  - 应用在二元分类表现最好，其他预测问题表现不是太好

`sklearn`库中对SVM的算法实现在包`sklearn.svm`里

```python
from sklearn import svm
```

##### 朴素贝叶斯（Naive Bayes）

> **朴素贝叶斯**（Naive Bayesian Model，NBM）是基于贝叶斯定理与特征条件独立假设的分类方法

​		贝叶斯方法是以贝叶斯原理为基础，使用概率统计的知识对样本数据集进行分类。由于其有着坚实的数学基础，贝叶斯分类算法的误判率是很低的。贝叶斯方法的特点是结合先验概率和后验概率，即避免了只使用先验概率的主观偏见，也避免了单独使用样本信息的过拟合现象。贝叶斯分类算法在数据集较大的情况下表现出较高的准确率，同时算法本身也比较简单。

**贝叶斯原理**

$P(A|B)=P(A)\frac{P(B|A)}{P(B)}$ 
其中：
		P(A)称为"先验概率"（Prior probability），即在B事件发生之前，我们对A事件概率的一个判断。
		P(A|B)称为"后验概率"（Posterior probability），即在B事件发生之后，我们对A事件概率的重新评估。
		P(B|A)/P(B)称为"可能性函数"（Likelyhood），这是一个调整因子，使得预估概率更接近真实概率。
贝叶斯定理之所以有用，是因为我们在生活中经常遇到这种情况：我们可以很容易直接得出P(A|B)，P(B|A)则很难直接得出，但我们更关心P(B|A)，贝叶斯定理就为我们打通从P(A|B)获得P(B|A)的道路。

**算法原理**

假设现在我们有一个数据集，它由两类数据组成，数据分布如下图所示：

<img src="./finalReportImages/bayes.png" alt="bayes" style="zoom:50%;" />

用 P1(x, y)表示数据点 (x, y)属于类别1(图中红色圆点表示的类别)的概率
用 P2(x, y)表示数据点 (x, y)属于类别2(图中蓝色三角形表示的类别)的概率
那么对于一个新数据点(x, y)，可以用下面的规则来判断它的类别：

- 如果P1(x, y) > P2(x ,y)，那么类别为1
- 如果P1(x, y) < P2(x ,y)，那么类别为2

​        也就是说，我们会选择**高概率**对应的类别。这就是贝叶斯决策理论的核心思想，即选择**具有最高概率的决策**。而朴素贝叶斯之所以称为“朴素”，是因为在整个过程中都假设特征之间是相互独立的以及每一个特征都是同等重要的。
​        **贝叶斯模型的训练过程实质上是在统计每一个特征出现的频次，其核心代码如下：** 

```python
def train(self, data):
    # data 中既包含正样本，也包含负样本
    for d in data: # data中是list
        # d[0]:分词的结果，list
        # d[1]:正/负样本的标记
        c = d[1]
        if c not in self.d:
            self.d[c] = AddOneProb() # 类的初始化
        for word in d[0]: # 分词结果中的每一个词
            self.d[c].add(word, 1)
    # 返回的是正类和负类之和
    self.total = sum(map(lambda x: self.d[x].getsum(), self.d.keys())) # 取得所有的d中的sum之和
```

**朴素贝叶斯的优缺点：**

- 优点：
  - 生成式模型，通过计算概率来进行分类，可以用来处理多分类问题。
  - 对小规模的数据表现很好，适合多分类任务，算法也比较简单。
- 缺点：
  - 由于朴素贝叶斯的“朴素”特点，所以会带来一些准确率上的损失。
  - 需要计算先验概率，分类决策存在错误率。
  - 朴素贝叶斯的准确率，依赖于训练语料。

`sklearn`库中对朴素贝叶斯的算法实现在包`sklearn.naive_bayes`里

##### Snownlp

> **SnowNLP**是一个python写的类库，可以方便的处理中文文本内容，是受到了TextBlob的启发而写的，由于现在大部分的自然语言处理库基本都是针对英文的，于是写了一个方便处理中文的类库，并且和TextBlob不同的是，这里没有用NLTK，所有的算法都是自己实现的，并且自带了一些训练好的字典。注意本程序都是处理的unicode编码，所以使用时请自行decode成unicode。

​		Snownlp是我们最终选择调用的主库，之所以选择Snownlp，是因为其集成了包括分词，去除停用词、情感分析等多种功能。
​		Snownlp中的情感分析（Sentiment）调用的机器学习算法是上面提到**朴素贝叶斯（Naive Bayes）**，有关情感分析的最核心算法在**Sentiment类**中，主要的步骤是：

1. 初始化贝叶斯模型
2. 分词以及去停用词的操作 
3. 读入正负样本组成训练集
4. 调用Bayes模型的训练方法
5. 得到训练好的模型
6. 保存最终的模型

**注**：具体的细节详见代码解释

​		通过调用Sentiment.train()后，我们训练好的贝叶斯模型被保存为一个名为`sentiment.marshal.3`的文件，由于snownlp本身有训练好的模型（但该模型的训练语料来自商品购买评价，不适用本次的作业），因此我们需要修改`snownlp/seg/__init__.py`里的`data_path`指向刚训练好的文件，之后便可以简单的调用训练好的分类器来进行判断了：

```python
# coding:utf-8
from snownlp import sentiment
print(sentiment.classify("xxxxxx"))
```

​		Snownlp的情感分析，即`sentiment.classify("”)`函数，会给我们返回一个积极情绪的概率，此概率为是一个 0~1 的浮点数。概率越接近 1，表示是**积极心态**的概率越高，越靠近 0，表示是**消极心态**的概率越高，而对于接近 0.5 的样本，我们则认为其是**中立心态**，经过简单的尝试，我们将大于 0.8 和小于 0.2 的才视为强烈的积极和消极，而对于 0.2~0.8 的样本，我们则认为其在积极和消极的置信度之外。

### 4.3-代码解释

##### Sentiment类

```python
class Sentiment(object):

    def __init__(self):
        self.classifier = Bayes() # 使用的是Bayes的模型

    def save(self, fname, iszip=True):
        self.classifier.save(fname, iszip) # 保存最终的模型

    def load(self, fname=data_path, iszip=True):
        self.classifier.load(fname, iszip) # 加载贝叶斯模型

    # 分词以及去停用词的操作    
    def handle(self, doc):
        words = seg.seg(doc) # 分词
        words = normal.filter_stop(words) # 去停用词
        return words # 返回分词后的结果

    def train(self, neg_docs, pos_docs):
        data = []
        # 读入负样本
        for sent in neg_docs:
            data.append([self.handle(sent), 'neg'])
        # 读入正样本
        for sent in pos_docs:
            data.append([self.handle(sent), 'pos'])
        # 调用的是Bayes模型的训练方法
        self.classifier.train(data)

    def classify(self, sent):
        # 1、调用sentiment类中的handle方法
        # 2、调用Bayes类中的classify方法
        ret, prob = self.classifier.classify(self.handle(sent)) # 调用贝叶斯中的classify方法
        if ret == 'pos':
            return prob
        return 1-probclass Sentiment(object):
```

##### \__init__.py文件 

```python
...
classifier = Sentiment() # 初始化类
classifier.load()

# 训练新模型的接口函数
def train(neg_file, pos_file):
    neg = codecs.open(neg_file, 'r', 'utf-8').readlines()
    pos = codecs.open(pos_file, 'r', 'utf-8').readlines()
    neg_docs = []
    pos_docs = []
    for line in neg:
        neg_docs.append(line.rstrip("\r\n"))
    for line in pos:
        pos_docs.append(line.rstrip("\r\n"))
    global classifier
    classifier = Sentiment()
    classifier.train(neg_docs, pos_docs)


# 保存模型的接口函数
def save(fname, iszip=True):
    classifier.save(fname, iszip)

def load(fname, iszip=True):
    classifier.load(fname, iszip)

# 心态分析函数
def classify(sent):
    return classifier.classify(sent)
```

## 05-总结

---

### 数据可视化

## 06-后记

---

### 感谢与感想

### 引用

### 不足与改进