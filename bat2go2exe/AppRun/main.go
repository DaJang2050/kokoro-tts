package main

import (
	"bufio"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
)

func main() {
	// 设置UTF-8编码
	cmd := exec.Command("chcp", "65001")
	cmd.Stdout = io.Discard // 丢弃标准输出
	cmd.Stderr = io.Discard // 丢弃标准错误输出
	cmd.Run()

	// 获取用户名
	username := os.Getenv("USERNAME")
	if username == "" && runtime.GOOS == "windows" {
		username = os.Getenv("USERPROFILE")
		if username != "" {
			username = filepath.Base(username)
		}
	}

	// 检查 uv.exe 是否存在
	uvPath := filepath.Join("C:", "Users", username, ".local", "bin", "uv.exe")
	_, err := os.Stat(uvPath)

	if err == nil {
		fmt.Printf("uv.exe 已存在于 %s，直接执行第四步\n", uvPath)
		err = os.Chdir("python")
		if err != nil {
			fmt.Printf("无法切换到python目录: %v\n", err)
			os.Exit(1)
		}
		// 执行第四步
		runStep4()
	} else {
		fmt.Println("uv.exe 不存在，将依次执行全部步骤")

		// 第一步：从本地安装uv
		fmt.Println("第一步：从本地安装uv")
		err = os.Chdir("uv")
		if err != nil {
			fmt.Printf("无法切换到uv目录: %v\n", err)
			os.Exit(1)
		}

		// 获取当前工作目录作为INSTALLER_DOWNLOAD_URL
		currentDir, err := os.Getwd()
		if err != nil {
			fmt.Printf("无法获取当前目录: %v\n", err)
			os.Exit(1)
		}

		// 设置环境变量 INSTALLER_DOWNLOAD_URL
		os.Setenv("INSTALLER_DOWNLOAD_URL", currentDir)

		// 执行 uv-installer.ps1 脚本
		cmd = exec.Command("powershell", "-ExecutionPolicy", "ByPass", "-File", ".\\uv-installer.ps1")
		cmd.Stdout = os.Stdout
		cmd.Stderr = os.Stderr
		err = cmd.Run()
		if err != nil {
			fmt.Printf("执行uv-installer.ps1失败: %v\n", err)
			os.Exit(1)
		}

		// 返回上一级目录，进入 python 目录
		err = os.Chdir("..\\python")
		if err != nil {
			fmt.Printf("无法切换到python目录: %v\n", err)
			os.Exit(1)
		}

		// 第二步：安装Python环境
		fmt.Println("第二步：开始安装Python环境")
		// 获取当前工作目录并转换路径格式
		currentDir, err = os.Getwd()
		if err != nil {
			fmt.Printf("无法获取当前目录: %v\n", err)
			os.Exit(1)
		}
		localMirror := "file:///" + strings.ReplaceAll(currentDir, "\\", "/")

		// 安装Python 3.11.9
		cmd = exec.Command("uv", "python", "install", "3.11.9", "--mirror", localMirror)
		cmd.Stdout = os.Stdout
		cmd.Stderr = os.Stderr
		err = cmd.Run()
		if err != nil {
			fmt.Printf("安装Python失败: %v\n", err)
			os.Exit(1)
		}

		// 第三步：使用uv恢复环境
		fmt.Println("第三步：使用uv恢复环境")
		// 同步依赖，指定清华镜像源
		cmd = exec.Command("uv", "sync", "--default-index", "https://pypi.tuna.tsinghua.edu.cn/simple")
		cmd.Stdout = os.Stdout
		cmd.Stderr = os.Stderr
		err = cmd.Run()
		if err != nil {
			fmt.Printf("同步依赖失败: %v\n", err)
			os.Exit(1)
		}

		// 询问是否安装CUDA版本的PyTorch
		fmt.Println("请问是否需要安装 CUDA 版本的 PyTorch（2.3 GiB）？(y/n) 它下载往往很慢（尽管使用南京大学安装源）")
		reader := bufio.NewReader(os.Stdin)
		choice, _ := reader.ReadString('\n')
		choice = strings.TrimSpace(strings.ToLower(choice))

		if choice == "y" {
			fmt.Println("正在安装 CUDA 版本的 PyTorch...")
			cmd = exec.Command("uv", "pip", "install", "torch", "torchvision", "torchaudio",
				"--index-url", "https://mirror.nju.edu.cn/pytorch/whl/cu126")
			cmd.Stdout = os.Stdout
			cmd.Stderr = os.Stderr
			err = cmd.Run()
			if err != nil {
				fmt.Printf("安装CUDA版本PyTorch失败: %v\n", err)
				os.Exit(1)
			}
		} else {
			fmt.Println("跳过安装 CUDA 版本的 PyTorch。")
			fmt.Println("如果需要使用 CPU 版本的 PyTorch，请自行安装。")
		}

		// 再次同步依赖
		cmd = exec.Command("uv", "sync")
		cmd.Stdout = os.Stdout
		cmd.Stderr = os.Stderr
		err = cmd.Run()
		if err != nil {
			fmt.Printf("同步依赖失败: %v\n", err)
			os.Exit(1)
		}

		// 执行第四步
		runStep4()
	}

	fmt.Println("请按回车键退出!")
	bufio.NewReader(os.Stdin).ReadBytes('\n')
}

func runStep4() {
	fmt.Println("第四步：运行 app.py")
	cmd := exec.Command("uv", "run", "app.py")
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	err := cmd.Run()
	if err != nil {
		fmt.Printf("运行app.py失败: %v\n", err)
		os.Exit(1)
	}
}
