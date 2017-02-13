#include "xclasses.h"

static int run = 1;
void handle_signal(int s) {
	run = 0;
}

int main(int argc, char *argv[]) {
	XMqtt mqtt;
	XMavlink mavlink("/dev/arduinoMega");
	char buffer[64];

	mavlink.AddListener(MAVLINK_MSG_ID_ATTITUDE, [&](mavlink_message_t msg){
		mavlink_attitude_t attitude;
		mavlink_msg_attitude_decode(&msg, &attitude);

		sprintf(buffer, "%5.3f, %5.3f, %5.3f", attitude.yaw, attitude.pitch, attitude.roll);

		mqtt.Publish("/mavlink/attitude", strlen(buffer)+1, buffer, 0);
	});

	while(run)
		std::this_thread::sleep_for(std::chrono::duration<int, std::milli>(100));

	return 0;
}
