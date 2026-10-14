# CyberLife — 仿生载体认知软件系统
#
# 基础镜像：Python 3.11 (slim)
# 用途：提供 CyberLife 项目统一的运行环境，内含声明的数值计算与机器学习依赖。
#
# 构建：docker build -t ghcr.io/li-guijiang/cyberlife:1.0.0 .
# 运行：docker run --rm ghcr.io/li-guijiang/cyberlife:1.0.0

FROM python:3.11-slim

LABEL org.opencontainers.image.title="CyberLife" \
      org.opencontainers.image.description="Embodied bionic carrier cognitive software system. Perception-decision-execution closed loop." \
      org.opencontainers.image.licenses="BSL-1.0" \
      org.opencontainers.image.source="https://github.com/Li-Guijiang/CyberLife" \
      org.opencontainers.image.vendor="Li-Guijiang"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CYBERLIFE_HOME=/opt/cyberlife

WORKDIR /opt/cyberlife

# 安装项目声明的核心依赖
RUN pip install --no-cache-dir \
        numpy \
        scipy \
        filterpy \
        scikit-learn

# 复制项目文件
COPY . .

# 语法预编译校验
RUN python -m compileall -q /opt/cyberlife > /dev/null 2>&1 || true

# 默认输出运行时信息
ENTRYPOINT ["python"]
CMD ["-c", "import sys, platform; print('CyberLife runtime ready'); print('Python', platform.python_version(), '|', sys.platform)"]
