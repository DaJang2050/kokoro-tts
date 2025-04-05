import time
import threading
import pyperclip  # 用于监听剪贴板
import sounddevice as sd
import re  # 用于检测中文字符
import sys
import torch
import os
import pathlib
# 在初始化TTS引擎前添加这段代码
import subprocess
import sys
import spacy

# 获取当前脚本所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 获取当前脚本所在目录的父目录
parent_dir = os.path.dirname(current_dir)

try:
    print("正在从本地安装 spaCy 英文模型...")
    whl_path = os.path.join(current_dir, "en_core_web_sm-3.8.0-py3-none-any.whl")
    if os.path.exists(whl_path):
        subprocess.check_call([sys.executable, "-m", "pip", "install", whl_path])
        print("spaCy 英文模型安装成功!")
    else:
        print(f"错误: 未找到本地 spaCy 模型文件: {whl_path}")
except Exception as e:
    print(f"安装 spaCy 模型时出错: {e}")

# 在安装模型后添加这段代码
try:
    nlp = spacy.load('en_core_web_sm')
    print("成功加载 spaCy 英文模型")
except Exception as e:
    print(f"加载 spaCy 模型失败: {e}")

# 构建本地模型路径
local_model_path = os.path.join(current_dir, "models--hexgrad--Kokoro-82M")

# 查看是否启用了cuda
if torch.cuda.is_available():
    print("已启用GPU（cuda）加速...")
else:
    print("正在使用CPU执行...")

# 检查模型目录是否存在
if os.path.exists(local_model_path):
    print(f"找到本地模型路径: {local_model_path}")
else:
    print(f"错误: 本地模型路径不存在: {local_model_path}")
    sys.exit(1)

# 在导入KPipeline之前先重写huggingface_hub的hf_hub_download函数
# 这是一个更直接的解决方案，强制使用本地文件
from huggingface_hub import hf_hub_download as original_hf_hub_download

def custom_hf_hub_download(repo_id, filename, **kwargs):
    """重写hf_hub_download函数，强制使用本地文件"""
    print(f"拦截下载请求: {repo_id}/{filename}，使用本地模型")
    
    # 处理voices文件夹下的文件请求
    if filename.startswith('voices/'):
        voice_file = os.path.join(local_model_path, filename)
        if os.path.exists(voice_file):
            print(f"使用本地语音文件: {voice_file}")
            return voice_file
    
    # 处理其他文件请求
    # 常见的请求包括config.json, pytorch_model.bin等
    file_path = os.path.join(local_model_path, filename)
    if os.path.exists(file_path):
        print(f"使用本地文件: {file_path}")
        return file_path
    
    # 查找可能的snapshots文件夹
    snapshots_dir = os.path.join(local_model_path, "snapshots")
    if os.path.exists(snapshots_dir):
        # 查找最新的快照
        snapshot_folders = [f for f in os.listdir(snapshots_dir) if os.path.isdir(os.path.join(snapshots_dir, f))]
        if snapshot_folders:
            latest_snapshot = max(snapshot_folders)
            snapshot_file = os.path.join(snapshots_dir, latest_snapshot, filename)
            if os.path.exists(snapshot_file):
                print(f"使用快照中的文件: {snapshot_file}")
                return snapshot_file
    
    # 如果未找到文件，尝试查找直接存放在models--hexgrad--Kokoro-82M中的文件
    files_in_model_dir = os.listdir(local_model_path)
    matching_files = [f for f in files_in_model_dir if f == filename or f.endswith(os.path.basename(filename))]
    if matching_files:
        file_path = os.path.join(local_model_path, matching_files[0])
        print(f"使用本地文件(直接匹配): {file_path}")
        return file_path
    
    # 如果仍然找不到，打印错误并返回原始路径，但这可能会导致程序崩溃
    print(f"错误: 无法在本地找到文件: {filename}")
    print(f"请确保您已下载完整的模型，或者提供正确的本地模型路径")
    
    # 由于我们已设置为离线模式，这应该会失败，但至少提供更清晰的错误信息
    raise FileNotFoundError(f"在本地模型目录中未找到所需文件: {filename}")

# 替换huggingface_hub库中的函数
import huggingface_hub
huggingface_hub.hf_hub_download = custom_hf_hub_download

# 设置离线模式，防止任何网络请求
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_HOME'] = current_dir

# 现在导入KPipeline
from kokoro import KPipeline

# 初始化 TTS 引擎 - 只初始化一次
try:
    print("正在初始化 TTS 引擎...")
    # 使用原始的参数调用，但由于我们修改了hf_hub_download函数，会使用本地模型
    pipeline = KPipeline(lang_code='a', repo_id='hexgrad/Kokoro-82M')
    print("TTS 引擎初始化成功")
except Exception as e:
    print(f"TTS 引擎初始化失败: {e}")
    # 打印完整的错误追踪信息，帮助调试
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 存储上一次剪贴板内容，用于对比检测变化
last_clipboard_content = ""
playing = False  # 标记是否正在播放

# 配置参数
REPEAT_COUNT = 3  # 重复播放次数
REPEAT_INTERVAL = 0  # 每次重复播放之间的间隔时间(秒)

def is_chinese_content(text):
    """检测文本是否包含中文字符"""
    # \u4e00-\u9fff 是中文字符的 Unicode 范围
    chinese_pattern = re.compile(r'[\u4e00-\u9fff]')
    return bool(chinese_pattern.search(text))

def play_text(text):
    """播放文本指定次数"""
    global playing
    playing = True
    try:
        for repeat in range(REPEAT_COUNT):
            print(f"正在播放第 {repeat+1} 遍: {text[:30]}..." if len(text) > 30 else f"正在播放第 {repeat+1} 遍: {text}")
            
            try:
                generator = pipeline(text, voice='af_heart')
                
                audio_played = False
                for i, (gs, ps, audio) in enumerate(generator):
                    # print(f"生成的音频片段 {i+1}, 准备播放...")
                    sd.play(audio, 24000)
                    sd.wait()
                    audio_played = True
                
                if not audio_played:
                    print("警告: 没有生成任何音频片段")
                    
            except Exception as e:
                print(f"播放过程中发生错误: {e}")
                # 打印更详细的错误信息用于调试
                import traceback
                traceback.print_exc()
            
            # 如果不是最后一次播放，添加短暂间隔
            if repeat < REPEAT_COUNT - 1:
                # print(f"等待 {REPEAT_INTERVAL} 秒后播放下一遍...")
                time.sleep(REPEAT_INTERVAL)
    finally:
        playing = False
        print("播放完成！")
        print('正在监听剪贴板...')

def monitor_clipboard():
    """监听剪贴板变化"""
    global last_clipboard_content, playing
    
    print(f"剪贴板监听已启动，复制非中文文本后将自动朗读{REPEAT_COUNT}次...")
    print("每次朗读间隔: {:.1f}秒".format(REPEAT_INTERVAL))
    
    # 测试声音设备
    try:
        print("测试声音设备...")
        sd.check_output_settings(device=None, channels=1, dtype='float32', samplerate=24000)
        print("声音设备测试成功")
        print('正在监听剪贴板...')
    except Exception as e:
        print(f"声音设备测试失败: {e}")
        print("程序将继续运行，但可能无法播放声音")
    
    while True:
        try:
            current_content = pyperclip.paste()
            
            # 检测剪贴板内容是否发生变化且不为空
            if current_content != last_clipboard_content and current_content.strip() != "":
                print(f"检测到剪贴板变化 [{len(current_content)} 字符]")
                
                # 更新上次剪贴板内容
                last_clipboard_content = current_content
                
                # 检查是否包含中文
                if is_chinese_content(current_content):
                    print(f"检测到中文内容，已忽略: {current_content[:50]}..." if len(current_content) > 50 else f"检测到中文内容，已忽略: {current_content}")
                    continue
                
                print(f"准备播放非中文内容: {current_content[:50]}..." if len(current_content) > 50 else f"准备播放非中文内容: {current_content}")
                
                # 如果正在播放，等待播放结束
                if playing:
                    print("等待当前播放结束...")
                    while playing:
                        time.sleep(0.2)  # 减少等待检查间隔
                
                # 创建新线程播放文本，避免阻塞剪贴板监听
                play_thread = threading.Thread(target=play_text, args=(current_content,))
                play_thread.daemon = True
                play_thread.start()
            
            # 缩短剪贴板轮询间隔，提高响应速度
            time.sleep(0.3)
            
        except Exception as e:
            print(f"监听剪贴板时发生错误: {e}")
            time.sleep(1)  # 出错时稍微延长休眠时间
        
if __name__ == "__main__":
    print("程序启动中...")
    
    # 获取初始剪贴板内容
    try:
        last_clipboard_content = pyperclip.paste()
        print(f"初始剪贴板内容: {last_clipboard_content[:30]}..." if len(last_clipboard_content) > 30 else f"初始剪贴板内容: {last_clipboard_content}")
    except Exception as e:
        print(f"获取剪贴板内容失败: {e}")
        last_clipboard_content = ""
    
    try:
        # 启动剪贴板监听
        monitor_clipboard()
    except KeyboardInterrupt:
        print("\n程序已退出")
    except Exception as e:
        print(f"程序运行时发生错误: {e}")
