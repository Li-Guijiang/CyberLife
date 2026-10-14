"""第二轮灌数据脚本：删除旧生活相册、灌入真实照片墙、说说文案、丰富文章

用法：python seed_v2.py
功能：
  1. 删除旧的"生活相册"及其 3 张占位照片
  2. 灌入 55 张真实生活照片（来自 public/images/photos/）
  3. 清空旧说说，灌入 15 条真实说说文案（分散在最近 15 天，每天一条）
  4. 灌入更多文章（技术/生活/随笔/项目分类，共 8 篇）
幂等可重跑。
"""

import os
import json
from datetime import datetime, timedelta
from sqlmodel import Session, select, delete

from app.database import engine
from app.models import (
    Category, Post, Chatter, Album, Photo,
)

# 55 张照片的映射（文件名 -> 横竖）
PHOTOS_MAP = os.path.join(os.path.dirname(__file__), "photos_map.json")

# 说说文案（15 条，按时间从早到晚）
CHATTERS = [
    "又和服务器斗智斗勇，宝塔端口报错折腾半天，总算找到问题在哪，搞服务器真的到处都是小坑😂",
    "一直在筹备我们的技术社团，想做成类似游戏开发部那种氛围，大家一起自由做项目、学技术。",
    "今天在 Wallpaper Engine 挑背景素材，看中了好几个同人动态壁纸，可惜场景类的没法直接导出视频，只能录屏处理。",
    "Dream-Space 慢慢成型，备案搞定，域名解析完成，离正式上线越来越近了。",
    "一边上课当班助，一边写代码改项目，时间被拆得很碎，挤时间开发已经成常态。",
    "和团队伙伴划分好了岗位，大家都是新手，边做边学，慢慢把平台维护起来。",
    "调后端数据库的时候又踩依赖版本的坑，Python 环境兼容真的很折磨人，慢慢一点点排错。",
    "空闲的时候打两把 CS，跳蹲落地总卡一下，习惯真难改。",
    "想把网站首页换成爱丽丝的动态视频背景，不过 4K 原文件太大，还要压缩转格式才能放到网页。",
    "从本地写代码，到买服务器、走 ICP 备案 + 公安备案，完整走一遍上线流程，学到好多课本没有的东西。",
    "社团材料还在慢慢整理，写介绍、准备申报，希望能顺利把社团办起来。",
    "敲代码累了就看看动漫，算是自己的放松方式。",
    "项目不会一次性完美，上线只是开始，后面会持续迭代，慢慢完善平台所有功能。",
    "今天测试网站访问，安全组没开端口，直接拒绝连接，又是学到新知识点的一天。",
    "一群人一起做项目的感觉挺好，各司其职，互相配合，一起打磨 Dream-Space。",
]

# 丰富文章（8 篇）
POSTS = [
    {
        "title": "从零上线一个个人网站：我的完整部署笔记",
        "slug": "deploy-personal-site-from-scratch",
        "description": "从本地写代码到买服务器、备案、配置 Nginx、申请 SSL 证书，完整记录一次个人网站上线全过程。",
        "content": """# 从零上线一个个人网站

这篇文章记录我完整走一遍网站上线流程的经验。

## 1. 买服务器

我选择的是阿里云 ECS，地域选成都（离得近，延迟低）。刚买来是 1 核 2G 的入门配置。

## 2. 配置环境

```bash
# 安装 Node.js、Python、Nginx、MySQL
yum install -y nginx mysql-server
```

## 3. 备案

国内服务器必须备案：ICP 备案 + 公安备案两步，周期较长，要提前准备材料。

## 4. 部署

用 PM2 托管前后端进程，Nginx 做反向代理，acme.sh 申请免费 SSL。

## 5. 踩坑总结

- 安全组端口没开 → 直接拒绝连接
- 低内存服务器 npm 安装容易 OOM → 加 swap
- Let's Encrypt 国内超时 → 换 ZeroSSL

> 上线只是开始，持续迭代才是常态。
""",
        "category": "技术",
        "pinned": True,
    },
    {
        "title": "我为什么想做「游戏开发部」式的技术社团",
        "slug": "why-game-dev-club-style-community",
        "description": "想打造一个像游戏开发部那样，大家自由做项目、学技术、互相配合的技术社团。",
        "content": """# 我为什么想做「游戏开发部」式的技术社团

一直很向往游戏开发部那种氛围：一群人因为热爱聚在一起，自由地做项目、学技术。

## 核心理念

- **自由**：不设硬性考核，凭兴趣驱动
- **协作**：各司其职，互相配合
- **成长**：边做边学，从新手慢慢变熟手

## 目前进展

社团材料还在整理，写介绍、准备申报，希望能顺利办起来。

## 未来计划

等社团成立，想组织大家一起做几个有意思的项目，把 Dream-Space 平台也维护得更好。
""",
        "category": "生活",
        "pinned": False,
    },
    {
        "title": "Python 依赖版本冲突的排查思路",
        "slug": "python-dependency-conflict",
        "description": "记录一次后端数据库驱动版本冲突的排查过程，以及 Python 环境兼容的通用解法。",
        "content": """# Python 依赖版本冲突的排查思路

调后端数据库时踩了依赖版本的坑，记录一下排查过程。

## 问题现象

项目从 PostgreSQL 迁移到 MySQL 后，模型定义报错 `VARCHAR requires a length`。

## 排查过程

1. 定位到 SQLModel 在 MySQL 下对 `str` 类型字段会生成 VARCHAR
2. 需要显式指定 max_length 或改用 Text 类型
3. 逐个模型检查，把无长度限制的 str 字段改成 Text

## 通用解法

- 用虚拟环境隔离依赖
- 用 requirements.txt 锁定版本
- 遇到兼容问题，先看驱动文档

> Python 环境兼容真的很折磨人，慢慢一点点排错就好。
""",
        "category": "技术",
        "pinned": False,
    },
    {
        "title": "网页动态壁纸的优化：4K 视频怎么塞进网页",
        "slug": "optimize-video-background-for-web",
        "description": "4K 动态壁纸原文件太大，如何压缩转格式后流畅地用在网页背景。",
        "content": """# 网页动态壁纸的优化

想把爱丽丝的动态视频背景放到网页，但 4K 原文件动辄几百 MB，直接塞进网页会卡死。

## 优化思路

1. **降分辨率**：4K → 1080p 甚至 720p，网页背景够用
2. **压码率**：用 CRF 25 左右，肉眼几乎无损
3. **去音轨**：背景视频不需要声音
4. **faststart**：把 moov 元数据移到文件头，浏览器能边下边播

## ffmpeg 命令

```bash
ffmpeg -i input.mp4 \
  -vf "scale='min(1280,iw)':-2" \
  -c:v libx264 -preset fast -crf 25 \
  -an -movflags +faststart output.mp4
```

实测一个 9.5MB 的视频能压到 400KB，加载快 20 倍。

> 场景类的动态壁纸没法直接导出视频，只能录屏处理，再压缩。
""",
        "category": "技术",
        "pinned": False,
    },
    {
        "title": "上班助与写代码：时间被拆碎的日子",
        "slug": "time-management-fragmented",
        "description": "一边当班助一边写代码，时间被拆得很碎，记录挤时间开发的生活常态。",
        "content": """# 上班助与写代码：时间被拆碎的日子

这学期一边当班助，一边写代码改项目，时间被拆得很碎。

## 时间碎片化

白天上课、处理班助事务，晚上才有大块时间写代码。挤时间开发已经成了常态。

## 一些小方法

- 把任务拆小，碎片时间也能推进一点
- 用待办清单记下所有 TODO
- 通勤时想好下一步要写什么

## 心态

虽然累，但看着项目一点点成型，还是很值得的。

> 时间被拆碎不可怕，可怕的是碎片时间被浪费。
""",
        "category": "生活",
        "pinned": False,
    },
    {
        "title": "给个人网站加上 ICP 和公安备案",
        "slug": "icp-and-public-security-filing",
        "description": "国内个人网站备案的完整流程：ICP 备案 + 公安联网备案，以及备案号如何展示。",
        "content": """# 给个人网站加上 ICP 和公安备案

国内服务器上线网站，备案是绕不开的一步。

## ICP 备案

在接入商（如阿里云）提交备案，审核通过后获得 ICP 备案号。

## 公安联网备案

ICP 通过后，再到公安备案系统提交，获得公安备案号和警徽图标。

## 网站展示

- ICP 备案号：放在页脚，链接到 beian.miit.gov.cn
- 公安备案号：放在页脚，配警徽图标，链接到 beian.mps.gov.cn

## 小提示

备案号要放在网站显眼位置，别人才能知道你的网站是正规备案的。

> 完整走一遍上线流程，学到好多课本没有的东西。
""",
        "category": "随笔",
        "pinned": False,
    },
    {
        "title": "CS 跳蹲落地总卡一下：聊聊游戏里的小习惯",
        "slug": "cs-gaming-habits",
        "description": "空闲打两把 CS，跳蹲落地总卡一下，习惯真难改。记录一点游戏和生活的小感想。",
        "content": """# CS 跳蹲落地总卡一下

空闲的时候打两把 CS，发现跳蹲落地总卡一下，这个小习惯真难改。

## 关于习惯

游戏里的小毛病，往往映射着生活里的小毛病。改习惯需要刻意练习。

## 放松方式

敲代码累了就看看动漫、打打游戏，算是自己的放松方式。

> 劳逸结合，才能走得更远。
""",
        "category": "生活",
        "pinned": False,
    },
    {
        "title": "Dream-Space 技术架构总览",
        "slug": "dream-space-architecture",
        "description": "介绍一下 Dream-Space 的全栈架构：Next.js 前端、FastAPI 后端、MySQL 数据库、PM2 部署。",
        "content": """# Dream-Space 技术架构总览

Dream-Space 是一个全栈个人博客与创意社区，记录一下它的架构。

## 前端

- **Next.js 16** + React 19 + Tailwind CSS 4
- 玻璃拟态设计风格
- 支持动态壁纸、音乐播放、照片墙、说说、小说、时间河流等模块

## 后端

- **FastAPI** + SQLModel
- MySQL 数据库
- JWT 认证

## 部署

- Nginx 反向代理 + SSL
- PM2 进程守护 + 开机自启

## 模块一览

文章、说说、照片墙、友链、收藏夹、音乐、小说、留言、时间河流、小工具。

> 项目不会一次性完美，上线只是开始，会持续迭代。
""",
        "category": "项目",
        "pinned": False,
    },
]


def main() -> None:
    with Session(engine) as session:
        # ============ 1. 删除旧"生活相册"及照片 ============
        old_album = session.exec(select(Album).where(Album.title == "生活相册")).first()
        if old_album:
            session.exec(delete(Photo).where(Photo.album_id == old_album.id))
            session.delete(old_album)
            session.commit()
            print("[OK] 已删除旧'生活相册'及其占位照片")

        # ============ 2. 灌入真实照片墙 ============
        photos_map = []
        if os.path.exists(PHOTOS_MAP):
            with open(PHOTOS_MAP, "r", encoding="utf-8") as f:
                photos_map = json.load(f)

        if photos_map:
            # 检查相册是否已存在
            album = session.exec(select(Album).where(Album.title == "我们的日常")).first()
            if album is None:
                album = Album(title="我们的日常", description="日常生活的记录，每一张都是珍贵的回忆", photo_count=len(photos_map), sort=0)
                session.add(album)
                session.flush()
                # 用前三张作为封面
                covers = photos_map[:3]
                for i, (fname, orient) in enumerate(covers):
                    session.add(Photo(
                        album_id=album.id,
                        url=f"/images/photos/{fname}",
                        caption="",
                        orientation=orient,
                        sort=i,
                    ))
            # 灌入全部照片
            existing_urls = {p.url for p in session.exec(select(Photo).where(Photo.album_id == album.id))}
            for i, (fname, orient) in enumerate(photos_map):
                url = f"/images/photos/{fname}"
                if url in existing_urls:
                    continue
                session.add(Photo(album_id=album.id, url=url, caption="", orientation=orient, sort=i))
            session.commit()
            # 更新相册封面（用第一张）
            album.cover = f"/images/photos/{photos_map[0][0]}"
            album.photo_count = len(photos_map)
            session.add(album)
            session.commit()
            print(f"[OK] 照片墙'我们的日常'已灌入 {len(photos_map)} 张照片")
        else:
            print("[WARN] 未找到 photos_map.json，跳过照片灌入")

        # ============ 3. 清空旧说说，灌入 15 条 ============
        old_chatters = session.exec(select(Chatter)).all()
        for c in old_chatters:
            session.delete(c)
        session.commit()

        base = datetime.now() - timedelta(days=len(CHATTERS) - 1)
        moods = ["happy", "thinking", "working", "happy", "busy", "working", "working", "playing", "thinking", "happy", "working", "relaxing", "thinking", "working", "happy"]
        for i, text in enumerate(CHATTERS):
            session.add(Chatter(
                content=text,
                mood=moods[i % len(moods)],
                status="published",
                likes=0,
                created_at=base + timedelta(days=i),
                updated_at=base + timedelta(days=i),
            ))
        session.commit()
        print(f"[OK] 说说已重置为 {len(CHATTERS)} 条（分散在最近 {len(CHATTERS)} 天）")

        # ============ 4. 灌入丰富文章 ============
        # 确保分类存在
        cat_map = {}
        for name, slug, desc in [
            ("技术", "tech", "技术分享与踩坑记录"),
            ("生活", "life", "日常随想与生活记录"),
            ("随笔", "essay", "杂谈与思考"),
            ("项目", "project", "项目开发与复盘"),
        ]:
            c = session.exec(select(Category).where(Category.slug == slug)).first()
            if c is None:
                c = Category(name=name, slug=slug, description=desc, sort=len(cat_map))
                session.add(c)
                session.flush()
            cat_map[slug] = c.id

        for p in POSTS:
            exists = session.exec(select(Post).where(Post.slug == p["slug"])).first()
            if exists:
                continue
            post = Post(
                title=p["title"],
                slug=p["slug"],
                description=p["description"],
                content=p["content"],
                category_id=cat_map.get(p["category"]),
                status="published",
                is_pinned=p["pinned"],
                views=100,
                likes=10,
                word_count=len(p["content"]),
                reading_time=max(1, len(p["content"]) // 300),
                published_at=datetime.now(),
            )
            session.add(post)
            print(f"[OK] 文章 {p['title']}")
        session.commit()

        print("\n[完成] 第二轮数据灌入完毕")


if __name__ == "__main__":
    main()
