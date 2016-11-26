SUBDIRS=Example02

.PHONY: clean all

all: $(SUBDIRS)
	for subdir in $(SUBDIRS); do \
		$(MAKE) -C $$subdir -j 12 all ; \
	done

clean: $(SUBDIRS)
	for subdir in $(SUBDIRS); do \
		$(MAKE) -C $$subdir -j 12 clean ; \
	done

