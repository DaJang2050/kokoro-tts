import os
import sys
import time
import threading
import pyperclip  # 用于监听剪贴板
import re  # 用于检测中文字符
import subprocess

# 全局变量声明
pipeline = None
nlp = None
init_completed = False
init_failed = False
init_error = None

# 存储上一次剪贴板内容，用于对比检测变化
last_clipboard_content = ""
playing = False  # 标记是否正在播放

# 配置参数
REPEAT_COUNT = 3  # 重复播放次数
REPEAT_INTERVAL = 0  # 每次重复播放之间的间隔时间(秒)

# 显示进度指示的计数器
progress_counter = 0

# 获取当前脚本所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

# 构建本地模型路径
local_model_path = os.path.join(current_dir, "models--hexgrad--Kokoro-82M")

def spinner():
    """显示加载进度动画"""
    global progress_counter, init_completed, init_failed
    spinner_chars = ["⣾", "⣽", "⣻", "⢿", "⡿", "⣟", "⣯", "⣷"]
    status_messages = [
        "正在初始化...",
        "加载模型中...", 
        "准备语音引擎...",
        "加载词汇库...",
        "配置神经网络..."
    ]
    
    while not (init_completed or init_failed):
        idx = progress_counter % len(spinner_chars)
        msg_idx = (progress_counter // 10) % len(status_messages)
        sys.stdout.write(f"\r{spinner_chars[idx]} {status_messages[msg_idx]}")
        sys.stdout.flush()
        progress_counter += 1
        time.sleep(0.1)
    
    if init_completed:
        sys.stdout.write("\r✓ 初始化完成!                 \n")
    else:
        sys.stdout.write(f"\r✗ 初始化失败! {init_error}       \n")
    sys.stdout.flush()

def is_chinese_content(text):
    """检测文本是否包含中文字符"""
    # \u4e00-\u9fff 是中文字符的 Unicode 范围
    chinese_pattern = re.compile(r'[\u4e00-\u9fff]')
    return bool(chinese_pattern.search(text))

def initialize_system():
    """初始化系统，包括模型加载等"""
    global pipeline, nlp, init_completed, init_failed, init_error
    
    try:
        # 1. 安装spaCy模型
        try:
            print("检查 spaCy 英文模型...")
            whl_path = os.path.join(current_dir, "en_core_web_sm-3.8.0-py3-none-any.whl")
            if os.path.exists(whl_path):
                subprocess.check_call([sys.executable, "-m", "pip", "install", whl_path], 
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as e:
            print(f"注意: spaCy 模型安装可选项: {e}")
        
        # 2. 检测GPU
        print("检测硬件加速...")
        import torch
        if torch.cuda.is_available():
            print("已启用GPU（CUDA）加速")
        else:
            print("使用CPU执行")
        
        # 3. 重写huggingface_hub的下载函数
        print("配置模型路径...")
        from huggingface_hub import hf_hub_download as original_hf_hub_download

        def custom_hf_hub_download(repo_id, filename, **kwargs):
            """重写hf_hub_download函数，使用本地文件缓存加速"""
            # 创建路径缓存，避免重复查找
            if not hasattr(custom_hf_hub_download, "path_cache"):
                custom_hf_hub_download.path_cache = {}
                
            cache_key = f"{repo_id}/{filename}"
            if cache_key in custom_hf_hub_download.path_cache:
                return custom_hf_hub_download.path_cache[cache_key]
            
            # 预定义可能的路径
            possible_paths = [
                os.path.join(local_model_path, filename),
                os.path.join(local_model_path, os.path.basename(filename)),
            ]
            
            # 处理voices文件夹下的文件请求
            if filename.startswith('voices/'):
                possible_paths.append(os.path.join(local_model_path, filename))
            
            # 快速检查snapshots文件夹
            snapshots_dir = os.path.join(local_model_path, "snapshots")
            if os.path.exists(snapshots_dir):
                snapshot_folders = [f for f in os.listdir(snapshots_dir) if os.path.isdir(os.path.join(snapshots_dir, f))]
                if snapshot_folders:
                    latest_snapshot = max(snapshot_folders)
                    possible_paths.append(os.path.join(snapshots_dir, latest_snapshot, filename))
            
            # 查找文件
            for path in possible_paths:
                if os.path.exists(path):
                    custom_hf_hub_download.path_cache[cache_key] = path
                    return path
            
            # 如果未找到，尝试直接匹配
            files_in_model_dir = os.listdir(local_model_path)
            matching_files = [f for f in files_in_model_dir if f == filename or f.endswith(os.path.basename(filename))]
            if matching_files:
                file_path = os.path.join(local_model_path, matching_files[0])
                custom_hf_hub_download.path_cache[cache_key] = file_path
                return file_path
            
            raise FileNotFoundError(f"在本地模型目录中未找到所需文件: {filename}")

        # 替换huggingface_hub库中的函数
        import huggingface_hub
        huggingface_hub.hf_hub_download = custom_hf_hub_download

        # 设置离线模式，防止任何网络请求
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['HF_HOME'] = current_dir
        
        # 4. 加载spaCy模型（可选）
        try:
            import spacy
            nlp = spacy.load('en_core_web_sm')
        except Exception as e:
            print(f"注意: spaCy 模型加载可选项: {e}")
        
        # 5. 初始化TTS引擎
        from kokoro import KPipeline
        pipeline = KPipeline(lang_code='a', repo_id='hexgrad/Kokoro-82M')
        
        # 6. 模型预热
        print("正在预热模型...")
        import sounddevice as sd
        warmup_text = "This is a test."
        generator = pipeline(warmup_text, voice='af_heart')
        first_chunk = True
        for _, _, audio in generator:
            if first_chunk:
                # 只播放第一个片段，不等待，只是预热
                sd.play(audio[0:100], 24000)
                first_chunk = False
                break
        
        # 7. 测试声音设备
        sd.check_output_settings(device=None, channels=1, dtype='float32', samplerate=24000)
        
        # 初始化完成
        init_completed = True
        
    except Exception as e:
        import traceback
        init_error = str(e)
        init_failed = True
        # 记录详细错误信息到文件
        with open("error_log.txt", "a") as f:
            f.write(f"\n--- Error at {time.ctime()} ---\n")
            traceback.print_exc(file=f)
        print(f"初始化错误: {e}")

def play_text(text):
    """播放文本指定次数"""
    global playing, pipeline, init_completed, init_failed
    
    # 如果初始化失败，直接返回
    if init_failed:
        print(f"无法播放，初始化失败: {init_error}")
        return
    
    # 如果初始化还没完成，等待
    if not init_completed:
        print("等待系统初始化完成...")
        while not (init_completed or init_failed):
            time.sleep(0.5)
        if init_failed:
            print(f"无法播放，初始化失败: {init_error}")
            return
    
    playing = True
    try:
        import sounddevice as sd
        for repeat in range(REPEAT_COUNT):
            print(f"正在播放第 {repeat+1} 遍: {text[:30]}..." if len(text) > 30 else f"正在播放第 {repeat+1} 遍: {text}")
            
            try:
                generator = pipeline(text, voice='af_heart')
                
                audio_played = False
                for i, (gs, ps, audio) in enumerate(generator):
                    sd.play(audio, 24000)
                    sd.wait()
                    audio_played = True
                
                if not audio_played:
                    print("警告: 没有生成任何音频片段")
                    
            except Exception as e:
                print(f"播放过程中发生错误: {e}")
            
            # 如果不是最后一次播放，添加短暂间隔
            if repeat < REPEAT_COUNT - 1:
                time.sleep(REPEAT_INTERVAL)
    finally:
        playing = False
        print("播放完成！")
        print('正在监听剪贴板...')

def monitor_clipboard():
    """监听剪贴板变化"""
    global last_clipboard_content, playing, init_completed, init_failed
    
    print(f"剪贴板监听已启动，复制非中文文本后将自动朗读{REPEAT_COUNT}次...")
    print("每次朗读间隔: {:.1f}秒".format(REPEAT_INTERVAL))
    
    # 获取初始剪贴板内容
    try:
        last_clipboard_content = pyperclip.paste()
        print(f"初始剪贴板内容:\n {last_clipboard_content[:30]}..." if len(last_clipboard_content) > 30 else f"初始剪贴板内容:\n {last_clipboard_content}")
    except Exception as e:
        print(f"获取剪贴板内容失败: {e}")
        last_clipboard_content = ""
    
    print('正在监听剪贴板...')
    
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
    print("-------------------")
    print("英文文本朗读助手")
    print("功能: 监听剪贴板中的英文文本并朗读")
    print("操作: 复制任何英文文本即可自动朗读")
    print("-------------------")
    
    # 检查模型路径
    if not os.path.exists(local_model_path):
        print(f"错误: 本地模型路径不存在: {local_model_path}")
        print("请确保模型文件夹放置在正确位置")
        sys.exit(1)
    else:
        print(f"已找到模型路径: {os.path.basename(local_model_path)}")
    
    # 创建并启动初始化线程
    init_thread = threading.Thread(target=initialize_system)
    init_thread.daemon = True
    init_thread.start()
    
    # 创建并启动进度指示器线程
    spinner_thread = threading.Thread(target=spinner)
    spinner_thread.daemon = True
    spinner_thread.start()
    
    try:
        # 启动剪贴板监听（不用等待初始化完成）
        monitor_clipboard()
    except KeyboardInterrupt:
        print("\n程序已退出")
    except Exception as e:
        print(f"程序运行时发生错误: {e}")
        import traceback
        traceback.print_exc()
