pip install nuitka
@REM Or debug better build with one of these builds:
@REM nuitka --standalone --show-progress --show-scons --show-modules --show-scons --debug edge_mesh.py
@REM nuitka --standalone --show-progress --show-scons --show-modules --show-scons --onefile --debug edge_mesh.py
nuitka --standalone --show-progress --show-scons --show-modules --show-scons --onefile --enable-plugin=tk-inter --enable-plugin=pyqt5 edge_mesh.py
