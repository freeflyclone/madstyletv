#pragma once
// MESSAGE SET_POSITION_TARGET_GLOBAL_INT PACKING

#define MAVLINK_MSG_ID_SET_POSITION_TARGET_GLOBAL_INT 86

MAVPACKED(
typedef struct __mavlink_set_position_target_global_int_t {
 uint32_t time_boot_ms; /*< Timestamp in milliseconds since system boot. The rationale for the timestamp in the setpoint is to allow the system to compensate for the transport delay of the setpoint. This allows the system to compensate processing latency.*/
 uint8_t target_system; /*< System ID*/
 uint8_t target_component; /*< Component ID*/
 uint8_t coordinate_frame; /*< Valid options are: MAV_FRAME_GLOBAL_INT = 5, MAV_FRAME_GLOBAL_RELATIVE_ALT_INT = 6, MAV_FRAME_GLOBAL_TERRAIN_ALT_INT = 11*/
 uint16_t type_mask; /*< Bitmask to indicate which dimensions should be ignored by the vehicle: a value of 0b0000000000000000 or 0b0000001000000000 indicates that none of the setpoint dimensions should be ignored. If bit 10 is set the floats afx afy afz should be interpreted as force instead of acceleration. Mapping: bit 1: x, bit 2: y, bit 3: z, bit 4: vx, bit 5: vy, bit 6: vz, bit 7: ax, bit 8: ay, bit 9: az, bit 10: is force setpoint, bit 11: yaw, bit 12: yaw rate*/
 int32_t lat_int; /*< X Position in WGS84 frame in 1e7 * meters*/
 int32_t lon_int; /*< Y Position in WGS84 frame in 1e7 * meters*/
 float alt; /*< Altitude in meters in AMSL altitude, not WGS84 if absolute or relative, above terrain if GLOBAL_TERRAIN_ALT_INT*/
 float vx; /*< X velocity in NED frame in meter / s*/
 float vy; /*< Y velocity in NED frame in meter / s*/
 float vz; /*< Z velocity in NED frame in meter / s*/
 float afx; /*< X acceleration or force (if bit 10 of type_mask is set) in NED frame in meter / s^2 or N*/
 float afy; /*< Y acceleration or force (if bit 10 of type_mask is set) in NED frame in meter / s^2 or N*/
 float afz; /*< Z acceleration or force (if bit 10 of type_mask is set) in NED frame in meter / s^2 or N*/
 float yaw; /*< yaw setpoint in rad*/
 float yaw_rate; /*< yaw rate setpoint in rad/s*/
}) mavlink_set_position_target_global_int_t;

#define MAVLINK_MSG_ID_SET_POSITION_TARGET_GLOBAL_INT_LEN 53
#define MAVLINK_MSG_ID_SET_POSITION_TARGET_GLOBAL_INT_MIN_LEN 53
#define MAVLINK_MSG_ID_86_LEN 53
#define MAVLINK_MSG_ID_86_MIN_LEN 53

#define MAVLINK_MSG_ID_SET_POSITION_TARGET_GLOBAL_INT_CRC 219
#define MAVLINK_MSG_ID_86_CRC 219



#if MAVLINK_COMMAND_24BIT
#define MAVLINK_MESSAGE_INFO_SET_POSITION_TARGET_GLOBAL_INT { \
    86, \
    "SET_POSITION_TARGET_GLOBAL_INT", \
    16, \
    {  { "time_boot_ms", NULL, MAVLINK_TYPE_UINT32_T, 0, 0, offsetof(mavlink_set_position_target_global_int_t, time_boot_ms) }, \
         { "target_system", NULL, MAVLINK_TYPE_UINT8_T, 0, 4, offsetof(mavlink_set_position_target_global_int_t, target_system) }, \
         { "target_component", NULL, MAVLINK_TYPE_UINT8_T, 0, 5, offsetof(mavlink_set_position_target_global_int_t, target_component) }, \
         { "coordinate_frame", NULL, MAVLINK_TYPE_UINT8_T, 0, 6, offsetof(mavlink_set_position_target_global_int_t, coordinate_frame) }, \
         { "type_mask", NULL, MAVLINK_TYPE_UINT16_T, 0, 7, offsetof(mavlink_set_position_target_global_int_t, type_mask) }, \
         { "lat_int", NULL, MAVLINK_TYPE_INT32_T, 0, 9, offsetof(mavlink_set_position_target_global_int_t, lat_int) }, \
         { "lon_int", NULL, MAVLINK_TYPE_INT32_T, 0, 13, offsetof(mavlink_set_position_target_global_int_t, lon_int) }, \
         { "alt", NULL, MAVLINK_TYPE_FLOAT, 0, 17, offsetof(mavlink_set_position_target_global_int_t, alt) }, \
         { "vx", NULL, MAVLINK_TYPE_FLOAT, 0, 21, offsetof(mavlink_set_position_target_global_int_t, vx) }, \
         { "vy", NULL, MAVLINK_TYPE_FLOAT, 0, 25, offsetof(mavlink_set_position_target_global_int_t, vy) }, \
         { "vz", NULL, MAVLINK_TYPE_FLOAT, 0, 29, offsetof(mavlink_set_position_target_global_int_t, vz) }, \
         { "afx", NULL, MAVLINK_TYPE_FLOAT, 0, 33, offsetof(mavlink_set_position_target_global_int_t, afx) }, \
         { "afy", NULL, MAVLINK_TYPE_FLOAT, 0, 37, offsetof(mavlink_set_position_target_global_int_t, afy) }, \
         { "afz", NULL, MAVLINK_TYPE_FLOAT, 0, 41, offsetof(mavlink_set_position_target_global_int_t, afz) }, \
         { "yaw", NULL, MAVLINK_TYPE_FLOAT, 0, 45, offsetof(mavlink_set_position_target_global_int_t, yaw) }, \
         { "yaw_rate", NULL, MAVLINK_TYPE_FLOAT, 0, 49, offsetof(mavlink_set_position_target_global_int_t, yaw_rate) }, \
         } \
}
#else
#define MAVLINK_MESSAGE_INFO_SET_POSITION_TARGET_GLOBAL_INT { \
    "SET_POSITION_TARGET_GLOBAL_INT", \
    16, \
    {  { "time_boot_ms", NULL, MAVLINK_TYPE_UINT32_T, 0, 0, offsetof(mavlink_set_position_target_global_int_t, time_boot_ms) }, \
         { "target_system", NULL, MAVLINK_TYPE_UINT8_T, 0, 4, offsetof(mavlink_set_position_target_global_int_t, target_system) }, \
         { "target_component", NULL, MAVLINK_TYPE_UINT8_T, 0, 5, offsetof(mavlink_set_position_target_global_int_t, target_component) }, \
         { "coordinate_frame", NULL, MAVLINK_TYPE_UINT8_T, 0, 6, offsetof(mavlink_set_position_target_global_int_t, coordinate_frame) }, \
         { "type_mask", NULL, MAVLINK_TYPE_UINT16_T, 0, 7, offsetof(mavlink_set_position_target_global_int_t, type_mask) }, \
         { "lat_int", NULL, MAVLINK_TYPE_INT32_T, 0, 9, offsetof(mavlink_set_position_target_global_int_t, lat_int) }, \
         { "lon_int", NULL, MAVLINK_TYPE_INT32_T, 0, 13, offsetof(mavlink_set_position_target_global_int_t, lon_int) }, \
         { "alt", NULL, MAVLINK_TYPE_FLOAT, 0, 17, offsetof(mavlink_set_position_target_global_int_t, alt) }, \
         { "vx", NULL, MAVLINK_TYPE_FLOAT, 0, 21, offsetof(mavlink_set_position_target_global_int_t, vx) }, \
         { "vy", NULL, MAVLINK_TYPE_FLOAT, 0, 25, offsetof(mavlink_set_position_target_global_int_t, vy) }, \
         { "vz", NULL, MAVLINK_TYPE_FLOAT, 0, 29, offsetof(mavlink_set_position_target_global_int_t, vz) }, \
         { "afx", NULL, MAVLINK_TYPE_FLOAT, 0, 33, offsetof(mavlink_set_position_target_global_int_t, afx) }, \
         { "afy", NULL, MAVLINK_TYPE_FLOAT, 0, 37, offsetof(mavlink_set_position_target_global_int_t, afy) }, \
         { "afz", NULL, MAVLINK_TYPE_FLOAT, 0, 41, offsetof(mavlink_set_position_target_global_int_t, afz) }, \
         { "yaw", NULL, MAVLINK_TYPE_FLOAT, 0, 45, offsetof(mavlink_set_position_target_global_int_t, yaw) }, \
         { "yaw_rate", NULL, MAVLINK_TYPE_FLOAT, 0, 49, offsetof(mavlink_set_position_target_global_int_t, yaw_rate) }, \
         } \
}
#endif

/**
 * @brief Pack a set_position_target_global_int message
 * @param system_id ID of this system
 * @param component_id ID of this component (e.g. 200 for IMU)
 * @param msg The MAVLink message to compress the data into
 *
 * @param time_boot_ms Timestamp in milliseconds since system boot. The rationale for the timestamp in the setpoint is to allow the system to compensate for the transport delay of the setpoint. This allows the system to compensate processing latency.
 * @param target_system System ID
 * @param target_component Component ID
 * @param coordinate_frame Valid options are: MAV_FRAME_GLOBAL_INT = 5, MAV_FRAME_GLOBAL_RELATIVE_ALT_INT = 6, MAV_FRAME_GLOBAL_TERRAIN_ALT_INT = 11
 * @param type_mask Bitmask to indicate which dimensions should be ignored by the vehicle: a value of 0b0000000000000000 or 0b0000001000000000 indicates that none of the setpoint dimensions should be ignored. If bit 10 is set the floats afx afy afz should be interpreted as force instead of acceleration. Mapping: bit 1: x, bit 2: y, bit 3: z, bit 4: vx, bit 5: vy, bit 6: vz, bit 7: ax, bit 8: ay, bit 9: az, bit 10: is force setpoint, bit 11: yaw, bit 12: yaw rate
 * @param lat_int X Position in WGS84 frame in 1e7 * meters
 * @param lon_int Y Position in WGS84 frame in 1e7 * meters
 * @param alt Altitude in meters in AMSL altitude, not WGS84 if absolute or relative, above terrain if GLOBAL_TERRAIN_ALT_INT
 * @param vx X velocity in NED frame in meter / s
 * @param vy Y velocity in NED frame in meter / s
 * @param vz Z velocity in NED frame in meter / s
 * @param afx X acceleration or force (if bit 10 of type_mask is set) in NED frame in meter / s^2 or N
 * @param afy Y acceleration or force (if bit 10 of type_mask is set) in NED frame in meter / s^2 or N
 * @param afz Z acceleration or force (if bit 10 of type_mask is set) in NED frame in meter / s^2 or N
 * @param yaw yaw setpoint in rad
 * @param yaw_rate yaw rate setpoint in rad/s
 * @return length of the message in bytes (excluding serial stream start sign)
 */
static inline uint16_t mavlink_msg_set_position_target_global_int_pack(uint8_t system_id, uint8_t component_id, mavlink_message_t* msg,
                               uint32_t time_boot_ms, uint8_t target_system, uint8_t target_component, uint8_t coordinate_frame, uint16_t type_mask, int32_t lat_int, int32_t lon_int, float alt, float vx, float vy, float vz, float afx, float afy, float afz, float yaw, float yaw_rate)
{
#if MAVLINK_NEED_BYTE_SWAP || !MAVLINK_ALIGNED_FIELDS
    char buf[MAVLINK_MSG_ID_SET_POSITION_TARGET_GLOBAL_INT_LEN];
    _mav_put_uint32_t(buf, 0, time_boot_ms);
    _mav_put_uint8_t(buf, 4, target_system);
    _mav_put_uint8_t(buf, 5, target_component);
    _mav_put_uint8_t(buf, 6, coordinate_frame);
    _mav_put_uint16_t(buf, 7, type_mask);
    _mav_put_int32_t(buf, 9, lat_int);
    _mav_put_int32_t(buf, 13, lon_int);
    _mav_put_float(buf, 17, alt);
    _mav_put_float(buf, 21, vx);
    _mav_put_float(buf, 25, vy);
    _mav_put_float(buf, 29, vz);
    _mav_put_float(buf, 33, afx);
    _mav_put_float(buf, 37, afy);
    _mav_put_float(buf, 41, afz);
    _mav_put_float(buf, 45, yaw);
    _mav_put_float(buf, 49, yaw_rate);

        memcpy(_MAV_PAYLOAD_NON_CONST(msg), buf, MAVLINK_MSG_ID_SET_POSITION_TARGET_GLOBAL_INT_LEN);
#else
    mavlink_set_position_target_global_int_t packet;
    packet.time_boot_ms = time_boot_ms;
    packet.target_system = target_system;
    packet.target_component = target_component;
    packet.coordinate_frame = coordinate_frame;
    packet.type_mask = type_mask;
    packet.lat_int = lat_int;
    packet.lon_int = lon_int;
    packet.alt = alt;
    packet.vx = vx;
    packet.vy = vy;
    packet.vz = vz;
    packet.afx = afx;
    packet.afy = afy;
    packet.afz = afz;
    packet.yaw = yaw;
    packet.yaw_rate = yaw_rate;

        memcpy(_MAV_PAYLOAD_NON_CONST(msg), &packet, MAVLINK_MSG_ID_SET_POSITION_TARGET_GLOBAL_INT_LEN);
#endif

    msg->msgid = MAVLINK_MSG_ID_SET_POSITION_TARGET_GLOBAL_INT;
    return mavlink_finalize_message(msg, system_id, component_id, MAVLINK_MSG_ID_SET_POSITION_TARGET_GLOBAL_INT_MIN_LEN, MAVLINK_MSG_ID_SET_POSITION_TARGET_GLOBAL_INT_LEN, MAVLINK_MSG_ID_SET_POSITION_TARGET_GLOBAL_INT_CRC);
}

/**
 * @brief Pack a set_position_target_global_int message on a channel
 * @param system_id ID of this system
 * @param component_id ID of this component (e.g. 200 for IMU)
 * @param chan The MAVLink channel this message will be sent over
 * @param msg The MAVLink message to compress the data into
 * @param time_boot_ms Timestamp in milliseconds since system boot. The rationale for the timestamp in the setpoint is to allow the system to compensate for the transport delay of the setpoint. This allows the system to compensate processing latency.
 * @param target_system System ID
 * @param target_component Component ID
 * @param coordinate_frame Valid options are: MAV_FRAME_GLOBAL_INT = 5, MAV_FRAME_GLOBAL_RELATIVE_ALT_INT = 6, MAV_FRAME_GLOBAL_TERRAIN_ALT_INT = 11
 * @param type_mask Bitmask to indicate which dimensions should be ignored by the vehicle: a value of 0b0000000000000000 or 0b0000001000000000 indicates that none of the setpoint dimensions should be ignored. If bit 10 is set the floats afx afy afz should be interpreted as force instead of acceleration. Mapping: bit 1: x, bit 2: y, bit 3: z, bit 4: vx, bit 5: vy, bit 6: vz, bit 7: ax, bit 8: ay, bit 9: az, bit 10: is force setpoint, bit 11: yaw, bit 12: yaw rate
 * @param lat_int X Position in WGS84 frame in 1e7 * meters
 * @param lon_int Y Position in WGS84 frame in 1e7 * meters
 * @param alt Altitude in meters in AMSL altitude, not WGS84 if absolute or relative, above terrain if GLOBAL_TERRAIN_ALT_INT
 * @param vx X velocity in NED frame in meter / s
 * @param vy Y velocity in NED frame in meter / s
 * @param vz Z velocity in NED frame in meter / s
 * @param afx X acceleration or force (if bit 10 of type_mask is set) in NED frame in meter / s^2 or N
 * @param afy Y acceleration or force (if bit 10 of type_mask is set) in NED frame in meter / s^2 or N
 * @param afz Z acceleration or force (if bit 10 of type_mask is set) in NED frame in meter / s^2 or N
 * @param yaw yaw setpoint in rad
 * @param yaw_rate yaw rate setpoint in rad/s
 * @return length of the message in bytes (excluding serial stream start sign)
 */
static inline uint16_t mavlink_msg_set_position_target_global_int_pack_chan(uint8_t system_id, uint8_t component_id, uint8_t chan,
                               mavlink_message_t* msg,
                                   uint32_t time_boot_ms,uint8_t target_system,uint8_t target_component,uint8_t coordinate_frame,uint16_t type_mask,int32_t lat_int,int32_t lon_int,float alt,float vx,float vy,float vz,float afx,float afy,float afz,float yaw,float yaw_rate)
{
#if MAVLINK_NEED_BYTE_SWAP || !MAVLINK_ALIGNED_FIELDS
    char buf[MAVLINK_MSG_ID_SET_POSITION_TARGET_GLOBAL_INT_LEN];
    _mav_put_uint32_t(buf, 0, time_boot_ms);
    _mav_put_uint8_t(buf, 4, target_system);
    _mav_put_uint8_t(buf, 5, target_component);
    _mav_put_uint8_t(buf, 6, coordinate_frame);
    _mav_put_uint16_t(buf, 7, type_mask);
    _mav_put_int32_t(buf, 9, lat_int);
    _mav_put_int32_t(buf, 13, lon_int);
    _mav_put_float(buf, 17, alt);
    _mav_put_float(buf, 21, vx);
    _mav_put_float(buf, 25, vy);
    _mav_put_float(buf, 29, vz);
    _mav_put_float(buf, 33, afx);
    _mav_put_float(buf, 37, afy);
    _mav_put_float(buf, 41, afz);
    _mav_put_float(buf, 45, yaw);
    _mav_put_float(buf, 49, yaw_rate);

        memcpy(_MAV_PAYLOAD_NON_CONST(msg), buf, MAVLINK_MSG_ID_SET_POSITION_TARGET_GLOBAL_INT_LEN);
#else
    mavlink_set_position_target_global_int_t packet;
    packet.time_boot_ms = time_boot_ms;
    packet.target_system = target_system;
    packet.target_component = target_component;
    packet.coordinate_frame = coordinate_frame;
    packet.type_mask = type_mask;
    packet.lat_int = lat_int;
    packet.lon_int = lon_int;
    packet.alt = alt;
    packet.vx = vx;
    packet.vy = vy;
    packet.vz = vz;
    packet.afx = afx;
    packet.afy = afy;
    packet.afz = afz;
    packet.yaw = yaw;
    packet.yaw_rate = yaw_rate;

        memcpy(_MAV_PAYLOAD_NON_CONST(msg), &packet, MAVLINK_MSG_ID_SET_POSITION_TARGET_GLOBAL_INT_LEN);
#endif

    msg->msgid = MAVLINK_MSG_ID_SET_POSITION_TARGET_GLOBAL_INT;
    return mavlink_finalize_message_chan(msg, system_id, component_id, chan, MAVLINK_MSG_ID_SET_POSITION_TARGET_GLOBAL_INT_MIN_LEN, MAVLINK_MSG_ID_SET_POSITION_TARGET_GLOBAL_INT_LEN, MAVLINK_MSG_ID_SET_POSITION_TARGET_GLOBAL_INT_CRC);
}

/**
 * @brief Encode a set_position_target_global_int struct
 *
 * @param system_id ID of this system
 * @param component_id ID of this component (e.g. 200 for IMU)
 * @param msg The MAVLink message to compress the data into
 * @param set_position_target_global_int C-struct to read the message contents from
 */
static inline uint16_t mavlink_msg_set_position_target_global_int_encode(uint8_t system_id, uint8_t component_id, mavlink_message_t* msg, const mavlink_set_position_target_global_int_t* set_position_target_global_int)
{
    return mavlink_msg_set_position_target_global_int_pack(system_id, component_id, msg, set_position_target_global_int->time_boot_ms, set_position_target_global_int->target_system, set_position_target_global_int->target_component, set_position_target_global_int->coordinate_frame, set_position_target_global_int->type_mask, set_position_target_global_int->lat_int, set_position_target_global_int->lon_int, set_position_target_global_int->alt, set_position_target_global_int->vx, set_position_target_global_int->vy, set_position_target_global_int->vz, set_position_target_global_int->afx, set_position_target_global_int->afy, set_position_target_global_int->afz, set_position_target_global_int->yaw, set_position_target_global_int->yaw_rate);
}

/**
 * @brief Encode a set_position_target_global_int struct on a channel
 *
 * @param system_id ID of this system
 * @param component_id ID of this component (e.g. 200 for IMU)
 * @param chan The MAVLink channel this message will be sent over
 * @param msg The MAVLink message to compress the data into
 * @param set_position_target_global_int C-struct to read the message contents from
 */
static inline uint16_t mavlink_msg_set_position_target_global_int_encode_chan(uint8_t system_id, uint8_t component_id, uint8_t chan, mavlink_message_t* msg, const mavlink_set_position_target_global_int_t* set_position_target_global_int)
{
    return mavlink_msg_set_position_target_global_int_pack_chan(system_id, component_id, chan, msg, set_position_target_global_int->time_boot_ms, set_position_target_global_int->target_system, set_position_target_global_int->target_component, set_position_target_global_int->coordinate_frame, set_position_target_global_int->type_mask, set_position_target_global_int->lat_int, set_position_target_global_int->lon_int, set_position_target_global_int->alt, set_position_target_global_int->vx, set_position_target_global_int->vy, set_position_target_global_int->vz, set_position_target_global_int->afx, set_position_target_global_int->afy, set_position_target_global_int->afz, set_position_target_global_int->yaw, set_position_target_global_int->yaw_rate);
}

/**
 * @brief Send a set_position_target_global_int message
 * @param chan MAVLink channel to send the message
 *
 * @param time_boot_ms Timestamp in milliseconds since system boot. The rationale for the timestamp in the setpoint is to allow the system to compensate for the transport delay of the setpoint. This allows the system to compensate processing latency.
 * @param target_system System ID
 * @param target_component Component ID
 * @param coordinate_frame Valid options are: MAV_FRAME_GLOBAL_INT = 5, MAV_FRAME_GLOBAL_RELATIVE_ALT_INT = 6, MAV_FRAME_GLOBAL_TERRAIN_ALT_INT = 11
 * @param type_mask Bitmask to indicate which dimensions should be ignored by the vehicle: a value of 0b0000000000000000 or 0b0000001000000000 indicates that none of the setpoint dimensions should be ignored. If bit 10 is set the floats afx afy afz should be interpreted as force instead of acceleration. Mapping: bit 1: x, bit 2: y, bit 3: z, bit 4: vx, bit 5: vy, bit 6: vz, bit 7: ax, bit 8: ay, bit 9: az, bit 10: is force setpoint, bit 11: yaw, bit 12: yaw rate
 * @param lat_int X Position in WGS84 frame in 1e7 * meters
 * @param lon_int Y Position in WGS84 frame in 1e7 * meters
 * @param alt Altitude in meters in AMSL altitude, not WGS84 if absolute or relative, above terrain if GLOBAL_TERRAIN_ALT_INT
 * @param vx X velocity in NED frame in meter / s
 * @param vy Y velocity in NED frame in meter / s
 * @param vz Z velocity in NED frame in meter / s
 * @param afx X acceleration or force (if bit 10 of type_mask is set) in NED frame in meter / s^2 or N
 * @param afy Y acceleration or force (if bit 10 of type_mask is set) in NED frame in meter / s^2 or N
 * @param afz Z acceleration or force (if bit 10 of type_mask is set) in NED frame in meter / s^2 or N
 * @param yaw yaw setpoint in rad
 * @param yaw_rate yaw rate setpoint in rad/s
 */
#ifdef MAVLINK_USE_CONVENIENCE_FUNCTIONS

static inline void mavlink_msg_set_position_target_global_int_send(mavlink_channel_t chan, uint32_t time_boot_ms, uint8_t target_system, uint8_t target_component, uint8_t coordinate_frame, uint16_t type_mask, int32_t lat_int, int32_t lon_int, float alt, float vx, float vy, float vz, float afx, float afy, float afz, float yaw, float yaw_rate)
{
#if MAVLINK_NEED_BYTE_SWAP || !MAVLINK_ALIGNED_FIELDS
    char buf[MAVLINK_MSG_ID_SET_POSITION_TARGET_GLOBAL_INT_LEN];
    _mav_put_uint32_t(buf, 0, time_boot_ms);
    _mav_put_uint8_t(buf, 4, target_system);
    _mav_put_uint8_t(buf, 5, target_component);
    _mav_put_uint8_t(buf, 6, coordinate_frame);
    _mav_put_uint16_t(buf, 7, type_mask);
    _mav_put_int32_t(buf, 9, lat_int);
    _mav_put_int32_t(buf, 13, lon_int);
    _mav_put_float(buf, 17, alt);
    _mav_put_float(buf, 21, vx);
    _mav_put_float(buf, 25, vy);
    _mav_put_float(buf, 29, vz);
    _mav_put_float(buf, 33, afx);
    _mav_put_float(buf, 37, afy);
    _mav_put_float(buf, 41, afz);
    _mav_put_float(buf, 45, yaw);
    _mav_put_float(buf, 49, yaw_rate);

    _mav_finalize_message_chan_send(chan, MAVLINK_MSG_ID_SET_POSITION_TARGET_GLOBAL_INT, buf, MAVLINK_MSG_ID_SET_POSITION_TARGET_GLOBAL_INT_MIN_LEN, MAVLINK_MSG_ID_SET_POSITION_TARGET_GLOBAL_INT_LEN, MAVLINK_MSG_ID_SET_POSITION_TARGET_GLOBAL_INT_CRC);
#else
    mavlink_set_position_target_global_int_t packet;
    packet.time_boot_ms = time_boot_ms;
    packet.target_system = target_system;
    packet.target_component = target_component;
    packet.coordinate_frame = coordinate_frame;
    packet.type_mask = type_mask;
    packet.lat_int = lat_int;
    packet.lon_int = lon_int;
    packet.alt = alt;
    packet.vx = vx;
    packet.vy = vy;
    packet.vz = vz;
    packet.afx = afx;
    packet.afy = afy;
    packet.afz = afz;
    packet.yaw = yaw;
    packet.yaw_rate = yaw_rate;

    _mav_finalize_message_chan_send(chan, MAVLINK_MSG_ID_SET_POSITION_TARGET_GLOBAL_INT, (const char *)&packet, MAVLINK_MSG_ID_SET_POSITION_TARGET_GLOBAL_INT_MIN_LEN, MAVLINK_MSG_ID_SET_POSITION_TARGET_GLOBAL_INT_LEN, MAVLINK_MSG_ID_SET_POSITION_TARGET_GLOBAL_INT_CRC);
#endif
}

/**
 * @brief Send a set_position_target_global_int message
 * @param chan MAVLink channel to send the message
 * @param struct The MAVLink struct to serialize
 */
static inline void mavlink_msg_set_position_target_global_int_send_struct(mavlink_channel_t chan, const mavlink_set_position_target_global_int_t* set_position_target_global_int)
{
#if MAVLINK_NEED_BYTE_SWAP || !MAVLINK_ALIGNED_FIELDS
    mavlink_msg_set_position_target_global_int_send(chan, set_position_target_global_int->time_boot_ms, set_position_target_global_int->target_system, set_position_target_global_int->target_component, set_position_target_global_int->coordinate_frame, set_position_target_global_int->type_mask, set_position_target_global_int->lat_int, set_position_target_global_int->lon_int, set_position_target_global_int->alt, set_position_target_global_int->vx, set_position_target_global_int->vy, set_position_target_global_int->vz, set_position_target_global_int->afx, set_position_target_global_int->afy, set_position_target_global_int->afz, set_position_target_global_int->yaw, set_position_target_global_int->yaw_rate);
#else
    _mav_finalize_message_chan_send(chan, MAVLINK_MSG_ID_SET_POSITION_TARGET_GLOBAL_INT, (const char *)set_position_target_global_int, MAVLINK_MSG_ID_SET_POSITION_TARGET_GLOBAL_INT_MIN_LEN, MAVLINK_MSG_ID_SET_POSITION_TARGET_GLOBAL_INT_LEN, MAVLINK_MSG_ID_SET_POSITION_TARGET_GLOBAL_INT_CRC);
#endif
}

#if MAVLINK_MSG_ID_SET_POSITION_TARGET_GLOBAL_INT_LEN <= MAVLINK_MAX_PAYLOAD_LEN
/*
  This varient of _send() can be used to save stack space by re-using
  memory from the receive buffer.  The caller provides a
  mavlink_message_t which is the size of a full mavlink message. This
  is usually the receive buffer for the channel, and allows a reply to an
  incoming message with minimum stack space usage.
 */
static inline void mavlink_msg_set_position_target_global_int_send_buf(mavlink_message_t *msgbuf, mavlink_channel_t chan,  uint32_t time_boot_ms, uint8_t target_system, uint8_t target_component, uint8_t coordinate_frame, uint16_t type_mask, int32_t lat_int, int32_t lon_int, float alt, float vx, float vy, float vz, float afx, float afy, float afz, float yaw, float yaw_rate)
{
#if MAVLINK_NEED_BYTE_SWAP || !MAVLINK_ALIGNED_FIELDS
    char *buf = (char *)msgbuf;
    _mav_put_uint32_t(buf, 0, time_boot_ms);
    _mav_put_uint8_t(buf, 4, target_system);
    _mav_put_uint8_t(buf, 5, target_component);
    _mav_put_uint8_t(buf, 6, coordinate_frame);
    _mav_put_uint16_t(buf, 7, type_mask);
    _mav_put_int32_t(buf, 9, lat_int);
    _mav_put_int32_t(buf, 13, lon_int);
    _mav_put_float(buf, 17, alt);
    _mav_put_float(buf, 21, vx);
    _mav_put_float(buf, 25, vy);
    _mav_put_float(buf, 29, vz);
    _mav_put_float(buf, 33, afx);
    _mav_put_float(buf, 37, afy);
    _mav_put_float(buf, 41, afz);
    _mav_put_float(buf, 45, yaw);
    _mav_put_float(buf, 49, yaw_rate);

    _mav_finalize_message_chan_send(chan, MAVLINK_MSG_ID_SET_POSITION_TARGET_GLOBAL_INT, buf, MAVLINK_MSG_ID_SET_POSITION_TARGET_GLOBAL_INT_MIN_LEN, MAVLINK_MSG_ID_SET_POSITION_TARGET_GLOBAL_INT_LEN, MAVLINK_MSG_ID_SET_POSITION_TARGET_GLOBAL_INT_CRC);
#else
    mavlink_set_position_target_global_int_t *packet = (mavlink_set_position_target_global_int_t *)msgbuf;
    packet->time_boot_ms = time_boot_ms;
    packet->target_system = target_system;
    packet->target_component = target_component;
    packet->coordinate_frame = coordinate_frame;
    packet->type_mask = type_mask;
    packet->lat_int = lat_int;
    packet->lon_int = lon_int;
    packet->alt = alt;
    packet->vx = vx;
    packet->vy = vy;
    packet->vz = vz;
    packet->afx = afx;
    packet->afy = afy;
    packet->afz = afz;
    packet->yaw = yaw;
    packet->yaw_rate = yaw_rate;

    _mav_finalize_message_chan_send(chan, MAVLINK_MSG_ID_SET_POSITION_TARGET_GLOBAL_INT, (const char *)packet, MAVLINK_MSG_ID_SET_POSITION_TARGET_GLOBAL_INT_MIN_LEN, MAVLINK_MSG_ID_SET_POSITION_TARGET_GLOBAL_INT_LEN, MAVLINK_MSG_ID_SET_POSITION_TARGET_GLOBAL_INT_CRC);
#endif
}
#endif

#endif

// MESSAGE SET_POSITION_TARGET_GLOBAL_INT UNPACKING


/**
 * @brief Get field time_boot_ms from set_position_target_global_int message
 *
 * @return Timestamp in milliseconds since system boot. The rationale for the timestamp in the setpoint is to allow the system to compensate for the transport delay of the setpoint. This allows the system to compensate processing latency.
 */
static inline uint32_t mavlink_msg_set_position_target_global_int_get_time_boot_ms(const mavlink_message_t* msg)
{
    return _MAV_RETURN_uint32_t(msg,  0);
}

/**
 * @brief Get field target_system from set_position_target_global_int message
 *
 * @return System ID
 */
static inline uint8_t mavlink_msg_set_position_target_global_int_get_target_system(const mavlink_message_t* msg)
{
    return _MAV_RETURN_uint8_t(msg,  4);
}

/**
 * @brief Get field target_component from set_position_target_global_int message
 *
 * @return Component ID
 */
static inline uint8_t mavlink_msg_set_position_target_global_int_get_target_component(const mavlink_message_t* msg)
{
    return _MAV_RETURN_uint8_t(msg,  5);
}

/**
 * @brief Get field coordinate_frame from set_position_target_global_int message
 *
 * @return Valid options are: MAV_FRAME_GLOBAL_INT = 5, MAV_FRAME_GLOBAL_RELATIVE_ALT_INT = 6, MAV_FRAME_GLOBAL_TERRAIN_ALT_INT = 11
 */
static inline uint8_t mavlink_msg_set_position_target_global_int_get_coordinate_frame(const mavlink_message_t* msg)
{
    return _MAV_RETURN_uint8_t(msg,  6);
}

/**
 * @brief Get field type_mask from set_position_target_global_int message
 *
 * @return Bitmask to indicate which dimensions should be ignored by the vehicle: a value of 0b0000000000000000 or 0b0000001000000000 indicates that none of the setpoint dimensions should be ignored. If bit 10 is set the floats afx afy afz should be interpreted as force instead of acceleration. Mapping: bit 1: x, bit 2: y, bit 3: z, bit 4: vx, bit 5: vy, bit 6: vz, bit 7: ax, bit 8: ay, bit 9: az, bit 10: is force setpoint, bit 11: yaw, bit 12: yaw rate
 */
static inline uint16_t mavlink_msg_set_position_target_global_int_get_type_mask(const mavlink_message_t* msg)
{
    return _MAV_RETURN_uint16_t(msg,  7);
}

/**
 * @brief Get field lat_int from set_position_target_global_int message
 *
 * @return X Position in WGS84 frame in 1e7 * meters
 */
static inline int32_t mavlink_msg_set_position_target_global_int_get_lat_int(const mavlink_message_t* msg)
{
    return _MAV_RETURN_int32_t(msg,  9);
}

/**
 * @brief Get field lon_int from set_position_target_global_int message
 *
 * @return Y Position in WGS84 frame in 1e7 * meters
 */
static inline int32_t mavlink_msg_set_position_target_global_int_get_lon_int(const mavlink_message_t* msg)
{
    return _MAV_RETURN_int32_t(msg,  13);
}

/**
 * @brief Get field alt from set_position_target_global_int message
 *
 * @return Altitude in meters in AMSL altitude, not WGS84 if absolute or relative, above terrain if GLOBAL_TERRAIN_ALT_INT
 */
static inline float mavlink_msg_set_position_target_global_int_get_alt(const mavlink_message_t* msg)
{
    return _MAV_RETURN_float(msg,  17);
}

/**
 * @brief Get field vx from set_position_target_global_int message
 *
 * @return X velocity in NED frame in meter / s
 */
static inline float mavlink_msg_set_position_target_global_int_get_vx(const mavlink_message_t* msg)
{
    return _MAV_RETURN_float(msg,  21);
}

/**
 * @brief Get field vy from set_position_target_global_int message
 *
 * @return Y velocity in NED frame in meter / s
 */
static inline float mavlink_msg_set_position_target_global_int_get_vy(const mavlink_message_t* msg)
{
    return _MAV_RETURN_float(msg,  25);
}

/**
 * @brief Get field vz from set_position_target_global_int message
 *
 * @return Z velocity in NED frame in meter / s
 */
static inline float mavlink_msg_set_position_target_global_int_get_vz(const mavlink_message_t* msg)
{
    return _MAV_RETURN_float(msg,  29);
}

/**
 * @brief Get field afx from set_position_target_global_int message
 *
 * @return X acceleration or force (if bit 10 of type_mask is set) in NED frame in meter / s^2 or N
 */
static inline float mavlink_msg_set_position_target_global_int_get_afx(const mavlink_message_t* msg)
{
    return _MAV_RETURN_float(msg,  33);
}

/**
 * @brief Get field afy from set_position_target_global_int message
 *
 * @return Y acceleration or force (if bit 10 of type_mask is set) in NED frame in meter / s^2 or N
 */
static inline float mavlink_msg_set_position_target_global_int_get_afy(const mavlink_message_t* msg)
{
    return _MAV_RETURN_float(msg,  37);
}

/**
 * @brief Get field afz from set_position_target_global_int message
 *
 * @return Z acceleration or force (if bit 10 of type_mask is set) in NED frame in meter / s^2 or N
 */
static inline float mavlink_msg_set_position_target_global_int_get_afz(const mavlink_message_t* msg)
{
    return _MAV_RETURN_float(msg,  41);
}

/**
 * @brief Get field yaw from set_position_target_global_int message
 *
 * @return yaw setpoint in rad
 */
static inline float mavlink_msg_set_position_target_global_int_get_yaw(const mavlink_message_t* msg)
{
    return _MAV_RETURN_float(msg,  45);
}

/**
 * @brief Get field yaw_rate from set_position_target_global_int message
 *
 * @return yaw rate setpoint in rad/s
 */
static inline float mavlink_msg_set_position_target_global_int_get_yaw_rate(const mavlink_message_t* msg)
{
    return _MAV_RETURN_float(msg,  49);
}

/**
 * @brief Decode a set_position_target_global_int message into a struct
 *
 * @param msg The message to decode
 * @param set_position_target_global_int C-struct to decode the message contents into
 */
static inline void mavlink_msg_set_position_target_global_int_decode(const mavlink_message_t* msg, mavlink_set_position_target_global_int_t* set_position_target_global_int)
{
#if MAVLINK_NEED_BYTE_SWAP || !MAVLINK_ALIGNED_FIELDS
    set_position_target_global_int->time_boot_ms = mavlink_msg_set_position_target_global_int_get_time_boot_ms(msg);
    set_position_target_global_int->target_system = mavlink_msg_set_position_target_global_int_get_target_system(msg);
    set_position_target_global_int->target_component = mavlink_msg_set_position_target_global_int_get_target_component(msg);
    set_position_target_global_int->coordinate_frame = mavlink_msg_set_position_target_global_int_get_coordinate_frame(msg);
    set_position_target_global_int->type_mask = mavlink_msg_set_position_target_global_int_get_type_mask(msg);
    set_position_target_global_int->lat_int = mavlink_msg_set_position_target_global_int_get_lat_int(msg);
    set_position_target_global_int->lon_int = mavlink_msg_set_position_target_global_int_get_lon_int(msg);
    set_position_target_global_int->alt = mavlink_msg_set_position_target_global_int_get_alt(msg);
    set_position_target_global_int->vx = mavlink_msg_set_position_target_global_int_get_vx(msg);
    set_position_target_global_int->vy = mavlink_msg_set_position_target_global_int_get_vy(msg);
    set_position_target_global_int->vz = mavlink_msg_set_position_target_global_int_get_vz(msg);
    set_position_target_global_int->afx = mavlink_msg_set_position_target_global_int_get_afx(msg);
    set_position_target_global_int->afy = mavlink_msg_set_position_target_global_int_get_afy(msg);
    set_position_target_global_int->afz = mavlink_msg_set_position_target_global_int_get_afz(msg);
    set_position_target_global_int->yaw = mavlink_msg_set_position_target_global_int_get_yaw(msg);
    set_position_target_global_int->yaw_rate = mavlink_msg_set_position_target_global_int_get_yaw_rate(msg);
#else
        uint8_t len = msg->len < MAVLINK_MSG_ID_SET_POSITION_TARGET_GLOBAL_INT_LEN? msg->len : MAVLINK_MSG_ID_SET_POSITION_TARGET_GLOBAL_INT_LEN;
        memset(set_position_target_global_int, 0, MAVLINK_MSG_ID_SET_POSITION_TARGET_GLOBAL_INT_LEN);
    memcpy(set_position_target_global_int, _MAV_PAYLOAD(msg), len);
#endif
}
