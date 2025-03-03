pip install nuitka
# Or debug better build with one of these builds:
# nuitka --standalone --show-progress --show-scons --show-modules --show-scons --onefile --debug edge_mesh.py

# This build for Linux
# nuitka --standalone --remove-output --onefile --show-progress --show-scons --show-modules --show-scons edge_mesh.py
# nuitka --standalone --onefile --show-progress --show-scons --show-modules --show-scons edge_mesh.py

# This builds for Windows:
nuitka --standalone --remove-output --mingw64 --onefile \
       --show-progress --show-scons --show-modules \
       --enable-plugin=pyqt6 \
       --enable-plugin=numpy \
       --enable-plugin=tensorflow \
       --enable-plugin=multiprocessing \
       --enable-plugin=transformers \
       --enable-plugin=matplotlib \
       --enable-plugin=upx edge_mesh.py
