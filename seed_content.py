"""灌入演示内容数据（MySQL）

用法：python seed_content.py
功能：插入示例分类、文章、说说、友链、相册、收藏站点，
     让网站"文字/说说/友链/相册"等模块有内容可展示。
所有插入均为幂等（已存在则跳过）。
"""

from datetime import datetime
from sqlmodel import Session, select

from app.database import engine
from app.models import (
    Category, Post, Chatter, FriendLink, Album, Photo,
    BookmarkCategory, BookmarkSite,
)


def main() -> None:
    with Session(engine) as session:
        # ============ 分类 ============
        categories = [
            ("技术", "tech", "技术分享与踩坑记录"),
            ("生活", "life", "日常随想与生活记录"),
            ("随笔", "essay", "杂谈与思考"),
        ]
        cat_map = {}
        for name, slug, desc in categories:
            c = session.exec(select(Category).where(Category.slug == slug)).first()
            if c is None:
                c = Category(name=name, slug=slug, description=desc, sort=len(cat_map))
                session.add(c)
                session.flush()
                print(f"[OK] 分类 {name}")
            cat_map[slug] = c.id

        # ============ 文章 ============
        posts = [
            {
                "title": "欢迎来到梦境空间",
                "slug": "welcome-to-dream-space",
                "description": "Dream-Space 梦境空间正式上线，记录技术、生活与思考。",
                "content": "# 欢迎来到梦境空间\n\n这是 Dream-Space 的第一篇文章。\n\n在这里，我会分享：\n\n- 技术踩坑与经验\n- 日常生活的碎碎念\n- 一些有趣的思考\n\n> 梦想，从此刻开始。",
                "category": "随笔",
                "pinned": True,
            },
            {
                "title": "用 Next.js 16 构建个人博客的实践",
                "slug": "nextjs-16-blog-practice",
                "description": "从零搭建一个基于 Next.js 16 + FastAPI 的个人博客全流程记录。",
                "content": "# 用 Next.js 16 构建个人博客\n\n本文记录本站的技术架构：\n\n## 前端\n\n- Next.js 16（App Router）\n- React 19 + Tailwind CSS 4\n- 玻璃拟态设计风格\n\n## 后端\n\n- FastAPI + SQLModel\n- MySQL 数据库\n\n## 部署\n\n- Nginx 反向代理\n- PM2 进程守护\n\n希望对你有所帮助。",
                "category": "技术",
                "pinned": False,
            },
            {
                "title": "我的 2026 年计划",
                "slug": "my-2026-plan",
                "description": "新的一年，新的目标，记录一些想做的事。",
                "content": "# 我的 2026 年计划\n\n1. 坚持写博客\n2. 深入学习后端开发\n3. 保持运动习惯\n4. 读 12 本书\n\n加油！",
                "category": "生活",
                "pinned": False,
            },
        ]
        for p in posts:
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

        # ============ 说说 ============
        chatters = [
            "今天把博客部署上线了，开心！🎉",
            "研究了一下动态壁纸背景，网站看起来更酷了。",
            "成都的天气不错，适合写代码。",
            "给网站加上了公安备案，更正规了。",
            "欢迎新朋友来访，欢迎留言交流～",
        ]
        for text in chatters:
            exists = session.exec(select(Chatter).where(Chatter.content == text)).first()
            if exists:
                continue
            session.add(Chatter(content=text, mood="happy", status="published", likes=5))
        print(f"[OK] 说说 {len(chatters)} 条")

        # ============ 友链 ============
        friend_links = [
            ("GitHub", "https://github.com", "全球最大的代码托管平台"),
            ("Next.js", "https://nextjs.org", "React 框架"),
            ("FastAPI", "https://fastapi.tiangolo.com", "高性能 Python Web 框架"),
            ("阿里云", "https://www.aliyun.com", "云计算服务平台"),
        ]
        for name, url, desc in friend_links:
            exists = session.exec(select(FriendLink).where(FriendLink.url == url)).first()
            if exists:
                continue
            session.add(FriendLink(name=name, url=url, description=desc, sort=0, is_approved=True))
        print(f"[OK] 友链 {len(friend_links)} 条")

        # ============ 相册 ============
        album = session.exec(select(Album).where(Album.title == "生活相册")).first()
        if album is None:
            album = Album(title="生活相册", description="记录生活的点滴", photo_count=3, sort=0)
            session.add(album)
            session.flush()
            session.add(Photo(album_id=album.id, url="/images/1.webp", caption="日常", sort=0))
            session.add(Photo(album_id=album.id, url="/images/20.webp", caption="风景", sort=1))
            session.add(Photo(album_id=album.id, url="/images/36.webp", caption="随手拍", sort=2))
            print("[OK] 相册 生活相册（3 张）")

        # ============ 收藏站点 ============
        bcat = session.exec(select(BookmarkCategory).where(BookmarkCategory.name == "开发工具")).first()
        if bcat is None:
            bcat = BookmarkCategory(name="开发工具", icon="🛠️", description="常用开发工具", sort=0)
            session.add(bcat)
            session.flush()
            sites = [
                ("GitHub", "https://github.com", "代码托管"),
                ("Stack Overflow", "https://stackoverflow.com", "技术问答"),
                ("MDN", "https://developer.mozilla.org", "Web 文档"),
            ]
            for name, url, desc in sites:
                session.add(BookmarkSite(category_id=bcat.id, name=name, url=url, description=desc, sort=0))
            print(f"[OK] 收藏站点 {len(sites)} 个")

        session.commit()
        print("\n[完成] 演示内容数据灌入完毕")


if __name__ == "__main__":
    main()
