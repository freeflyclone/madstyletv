all: ${PROGRAM}

${PROGRAM}: ${PROGRAM_OBJS}
	$(call PROGRAM_BUILD)

clean:
	@-rm -rf $(PROGRAM)
	@-rm -rf $(PROGRAM_OBJS)

test:
	$(call ECHO_VARS)
