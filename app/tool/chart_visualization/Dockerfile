FROM python:3.12-slim
RUN sed -i 's@deb.debian.org@mirrors.aliyun.com@g' /etc/apt/sources.list.d/debian.sources
RUN apt-get update && apt-get install -y --no-install-recommends \
    apt-transport-https \
    curl \
    gnupg \
    ca-certificates \
    build-essential \
    libexpat1 \
    chromium \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js LTS version and npm
RUN curl -fsSL https://deb.nodesource.com/setup_lts.x | bash - && \
    apt-get install -y nodejs && \
    node -v && npm -v

RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple
RUN pip install --no-cache-dir plotly pandas matplotlib seaborn kaleido numpy


# Install chart_visualization and required dependencies
COPY ./src /chart_visualization/src
COPY ./package.json /chart_visualization/package.json
COPY ./tsconfig.json /chart_visualization/tsconfig.json
ENV PUPPETEER_EXECUTABLE_PATH=/usr/bin/chromium
RUN cd /chart_visualization && npm install

