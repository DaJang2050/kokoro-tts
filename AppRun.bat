@echo off
chcp 65001 > nul

echo 第一步：从本地安装uv
cd uv
REM 设置 INSTALLER_DOWNLOAD_URL 环境变量 实现从本地安装，避免Github网络问题
for /f "delims=" %%a in ('powershell -command "(Get-Location).Path"') do set "INSTALLER_DOWNLOAD_URL=%%a"
REM 执行 uv-installer.ps1 脚本 (需要 PowerShell 环境)
powershell -ExecutionPolicy ByPass -File .\uv-installer.ps1
REM 返回上一级目录，进入 python 目录
cd ..\python

echo 第二步：开始安装Python环境
REM 安装 Python 3.11.9，指定本地镜像源，避免Github网络问题
for /f "delims=" %%a in ('powershell -command "(Get-Location).Path -replace '\\','/'"') do set "LOCAL_MIRROR=file:///%%a"
uv python install 3.11.9 --mirror "%LOCAL_MIRROR%"

echo 第三步：使用uv恢复环境
REM 同步依赖，指定清华镜像源
uv sync --default-index "https://pypi.tuna.tsinghua.edu.cn/simple"
echo 请问是否需要安装 CUDA 版本的 PyTorch（2.3 GiB）？(y/n) 它下载往往很慢（尽管使用南京大学安装源） 
set /p "choice=请输入您的选择: "
if /i "%choice%"=="y" (
  echo 正在安装 CUDA 版本的 PyTorch...
  uv pip install torch torchvision torchaudio --index-url https://mirror.nju.edu.cn/pytorch/whl/cu126
) else (
  echo 跳过安装 CUDA 版本的 PyTorch。
  echo 如果需要使用 CPU 版本的 PyTorch，请自行安装。
)
uv sync

echo 第四步：运行 app.py
uv run app.py
echo 请按回车键退出!
pause
