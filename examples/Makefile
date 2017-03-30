SUBDIRS=Example02 \
	OpenALTest \
	OpenCVTest \
	OpenCVWebcam \
	MavlinkDev \
	MqttDev \
	MqttMav \
	PhysXTest \
	ComputeShaderTest \
	SDFTest \
	GUIDev \
	ImuDev \
	XAVTest


.PHONY: clean all

all: $(SUBDIRS)
	for subdir in $(SUBDIRS); do \
                $(MAKE) -C $$subdir all ; \
	done

clean: $(SUBDIRS)
	for subdir in $(SUBDIRS); do \
                $(MAKE) -C $$subdir clean ; \
	done

