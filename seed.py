"""初始化脚本（MySQL）

用法：
    python seed.py

功能：
    1. 自动建表（若不存在）
    2. 创建默认管理员账号 admin / admin123
    3. 写入默认站点配置
"""

from sqlmodel import Session, select

from app.database import init_db, engine
from app.models import User, SiteConfig
from app.utils.auth import hash_password

DEFAULT_SITE_CONFIG = {
    "site_title": '"Dream-Space 梦境空间"',
    "site_description": '"李贵江的个人博客 —— 梦境空间"',
    "icp_number": '""',
    "icp_link": '""',
}


def main() -> None:
    init_db()

    with Session(engine) as session:
        # 1. 默认管理员
        admin = session.exec(select(User).where(User.username == "admin")).first()
        if admin is None:
            session.add(
                User(
                    username="admin",
                    hashed_password=hash_password("admin123"),
                    nickname="李贵江",
                    is_admin=True,
                )
            )
            print("[OK] 已创建管理员账号 admin / admin123")
        else:
            print("[SKIP] 管理员账号已存在")

        # 2. 默认站点配置
        for key, value in DEFAULT_SITE_CONFIG.items():
            row = session.exec(select(SiteConfig).where(SiteConfig.key == key)).first()
            if row is None:
                session.add(SiteConfig(key=key, value=value))
        session.commit()
        print("[OK] 站点配置初始化完成")


if __name__ == "__main__":
    main()
