#! /bin/bash

docker build -t python:3.12-slim-test -f app/tool/chart_visualization/Dockerfile app/tool/chart_visualization
