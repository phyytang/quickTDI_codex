#!/usr/bin/env python3
"""
Launcher script for TJ_GUI with system compatibility checks. TJ_GUI启动脚本，包含系统兼容性检查。
"""

import sys
import platform
from pathlib import Path

# Make sure this script's directory (for TJ_GUI) and the src/ directory
# (for the TJ_*.py simulation modules) are importable regardless of cwd.
_GUI_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_GUI_DIR))
sys.path.insert(0, str(_GUI_DIR.parent / "src"))

def check_system():
    """Check system compatibility 检查系统兼容性"""
    print("=" * 60)
    print("Taiji/LISA TDI Simulator - System Check 太极/LISA TDI仿真器 - 系统检查")
    print("=" * 60)

    # Python version Python版本
    py_version = sys.version_info
    print(f"Python version Python版本: {py_version.major}.{py_version.minor}.{py_version.micro}")

    if py_version < (3, 9):
        print("ERROR 错误: Python 3.9 or later required! 需要Python 3.9或更高版本!")
        return False

    # macOS version macOS版本
    if platform.system() == 'Darwin':
        mac_ver = platform.mac_ver()[0]
        print(f"macOS version macOS版本: {mac_ver}")

    # Check tkinter 检查tkinter
    try:
        import tkinter as tk
        root = tk.Tk()
        tk_version = root.tk.call('info', 'patchlevel')
        root.destroy()
        print(f"Tkinter version Tkinter版本: {tk_version}")
    except Exception as e:
        print(f"ERROR 错误: Tkinter not available Tkinter不可用: {e}")
        return False

    # Check required packages 检查必需的包
    required_packages = ['numpy', 'scipy', 'matplotlib']
    missing = []

    for package in required_packages:
        try:
            mod = __import__(package)
            version = getattr(mod, '__version__', 'unknown')
            print(f"{package} version {package}版本: {version}")
        except ImportError:
            missing.append(package)

    if missing:
        print(f"\nERROR 错误: Missing packages 缺少包: {', '.join(missing)}")
        print(f"Install with 安装命令: python3 -m pip install {' '.join(missing)}")
        return False

    print("=" * 60)
    print("All checks passed! Starting GUI... 所有检查通过！启动GUI...")
    print("=" * 60)
    return True

def main():
    """Main entry point 主入口"""
    if not check_system():
        sys.exit(1)

    try:
        import TJ_GUI
        TJ_GUI.main()
    except Exception as e:
        import traceback
        print("\n" + "=" * 60)
        print("ERROR 错误: GUI failed to start GUI启动失败")
        print("=" * 60)
        print(f"Error 错误: {e}")
        print("\nFull traceback 完整追溯:")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
