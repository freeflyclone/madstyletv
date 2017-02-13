SUBDIRS=Example02 \
	OpenALTest \
	OpenCVTest \
	OpenCVWebcam \
	MavlinkDev \
	MqttDev \
	PhysXTest \
	ComputeShaderTest \
	SDFTest \
	GUIDev

.PHONY: clean all

all: $(SUBDIRS)
	for subdir in $(SUBDIRS); do \
                $(MAKE) -C $$subdir all ; \
	done

clean: $(SUBDIRS)
	for subdir in $(SUBDIRS); do \
                $(MAKE) -C $$subdir clean ; \
	done

