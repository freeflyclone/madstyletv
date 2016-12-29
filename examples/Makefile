SUBDIRS=Example02 OpenALTest MavlinkDev PhysXTest

.PHONY: clean all

all: $(SUBDIRS)
	for subdir in $(SUBDIRS); do \
                $(MAKE) -C $$subdir all ; \
	done

clean: $(SUBDIRS)
	for subdir in $(SUBDIRS); do \
                $(MAKE) -C $$subdir clean ; \
	done

