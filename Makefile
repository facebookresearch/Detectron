# Don't use the --user flag for setup.py develop mode with virtualenv.
DEV_USER_FLAG=$(shell python2 -c "import sys; print('' if hasattr(sys, 'real_prefix') else '--user')")

.PHONY: default
default: dev

.PHONY: install
install:
	python2 setup.py install

.PHONY: ops
ops:
	mkdir -p build && cd build && cmake .. && make -j$(shell nproc)

.PHONY: dev
dev:
	python2 setup.py develop $(DEV_USER_FLAG)

.PHONY: clean
clean:
	python2 setup.py develop --uninstall $(DEV_USER_FLAG)
	rm -rf build
