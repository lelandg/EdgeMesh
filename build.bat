pip install nuitka
nuitka --standalone --remove-output --mingw64 --onefile --show-progress --show-scons --show-modules --enable-plugin=pyqt6 --enable-plugin=upx edge_mesh.py
