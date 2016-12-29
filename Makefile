SUBDIRS=xclass examples
include toplevel.mk

all: $(SUBDIRS)
	for subdir in $(SUBDIRS); do \
		$(MAKE) -C $$subdir -j 12 all ; \
	done

clean: $(SUBDIRS)
	for subdir in $(SUBDIRS); do \
		$(MAKE) -C $$subdir clean ; \
	done

