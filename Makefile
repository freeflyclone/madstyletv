SUBDIRS=xclass/3rdParty xclass examples

include topheader.mk

all: $(SUBDIRS)
	for subdir in $(SUBDIRS); do \
		$(MAKE) -C $$subdir -j 12 all ; \
	done

clean: $(SUBDIRS)
	for subdir in $(SUBDIRS); do \
		$(MAKE) -C $$subdir clean ; \
	done
