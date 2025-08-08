import AutoROM
import os

if __name__ == "__main__":
    # 默认下载地址（AutoROM会自动下载）
    source_file = None  # None表示自动处理
    # 安装目录（比如环境site-packages的roms目录）
    install_dir = os.path.join(os.path.dirname(AutoROM.__file__), "roms")
    # 是否静默安装
    quiet = True

    AutoROM.main(source_file, install_dir, quiet)