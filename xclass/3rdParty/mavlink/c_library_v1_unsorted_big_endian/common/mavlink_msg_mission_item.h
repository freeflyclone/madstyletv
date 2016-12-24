#pragma once
// MESSAGE MISSION_ITEM PACKING

#define MAVLINK_MSG_ID_MISSION_ITEM 39

MAVPACKED(
typedef struct __mavlink_mission_item_t {
 uint8_t target_system; /*< System ID*/
 uint8_t target_component; /*< Component ID*/
 uint16_t seq; /*< Sequence*/
 uint8_t frame; /*< The coordinate system of the MISSION. see MAV_FRAME in mavlink_types.h*/
 uint16_t command; /*< The scheduled action for the MISSION. see MAV_CMD in common.xml MAVLink specs*/
 uint8_t current; /*< false:0, true:1*/
 uint8_t autocontinue; /*< autocontinue to next wp*/
 float param1; /*< PARAM1, see MAV_CMD enum*/
 float param2; /*< PARAM2, see MAV_CMD enum*/
 float param3; /*< PARAM3, see MAV_CMD enum*/
 float param4; /*< PARAM4, see MAV_CMD enum*/
 float x; /*< PARAM5 / local: x position, global: latitude*/
 float y; /*< PARAM6 / y position: global: longitude*/
 float z; /*< PARAM7 / z position: global: altitude (relative or absolute, depending on frame.*/
}) mavlink_mission_item_t;

#define MAVLINK_MSG_ID_MISSION_ITEM_LEN 37
#define MAVLINK_MSG_ID_MISSION_ITEM_MIN_LEN 37
#define MAVLINK_MSG_ID_39_LEN 37
#define MAVLINK_MSG_ID_39_MIN_LEN 37

#define MAVLINK_MSG_ID_MISSION_ITEM_CRC 148
#define MAVLINK_MSG_ID_39_CRC 148



#if MAVLINK_COMMAND_24BIT
#define MAVLINK_MESSAGE_INFO_MISSION_ITEM { \
    39, \
    "MISSION_ITEM", \
    14, \
    {  { "target_system", NULL, MAVLINK_TYPE_UINT8_T, 0, 0, offsetof(mavlink_mission_item_t, target_system) }, \
         { "target_component", NULL, MAVLINK_TYPE_UINT8_T, 0, 1, offsetof(mavlink_mission_item_t, target_component) }, \
         { "seq", NULL, MAVLINK_TYPE_UINT16_T, 0, 2, offsetof(mavlink_mission_item_t, seq) }, \
         { "frame", NULL, MAVLINK_TYPE_UINT8_T, 0, 4, offsetof(mavlink_mission_item_t, frame) }, \
         { "command", NULL, MAVLINK_TYPE_UINT16_T, 0, 5, offsetof(mavlink_mission_item_t, command) }, \
         { "current", NULL, MAVLINK_TYPE_UINT8_T, 0, 7, offsetof(mavlink_mission_item_t, current) }, \
         { "autocontinue", NULL, MAVLINK_TYPE_UINT8_T, 0, 8, offsetof(mavlink_mission_item_t, autocontinue) }, \
         { "param1", NULL, MAVLINK_TYPE_FLOAT, 0, 9, offsetof(mavlink_mission_item_t, param1) }, \
         { "param2", NULL, MAVLINK_TYPE_FLOAT, 0, 13, offsetof(mavlink_mission_item_t, param2) }, \
         { "param3", NULL, MAVLINK_TYPE_FLOAT, 0, 17, offsetof(mavlink_mission_item_t, param3) }, \
         { "param4", NULL, MAVLINK_TYPE_FLOAT, 0, 21, offsetof(mavlink_mission_item_t, param4) }, \
         { "x", NULL, MAVLINK_TYPE_FLOAT, 0, 25, offsetof(mavlink_mission_item_t, x) }, \
         { "y", NULL, MAVLINK_TYPE_FLOAT, 0, 29, offsetof(mavlink_mission_item_t, y) }, \
         { "z", NULL, MAVLINK_TYPE_FLOAT, 0, 33, offsetof(mavlink_mission_item_t, z) }, \
         } \
}
#else
#define MAVLINK_MESSAGE_INFO_MISSION_ITEM { \
    "MISSION_ITEM", \
    14, \
    {  { "target_system", NULL, MAVLINK_TYPE_UINT8_T, 0, 0, offsetof(mavlink_mission_item_t, target_system) }, \
         { "target_component", NULL, MAVLINK_TYPE_UINT8_T, 0, 1, offsetof(mavlink_mission_item_t, target_component) }, \
         { "seq", NULL, MAVLINK_TYPE_UINT16_T, 0, 2, offsetof(mavlink_mission_item_t, seq) }, \
         { "frame", NULL, MAVLINK_TYPE_UINT8_T, 0, 4, offsetof(mavlink_mission_item_t, frame) }, \
         { "command", NULL, MAVLINK_TYPE_UINT16_T, 0, 5, offsetof(mavlink_mission_item_t, command) }, \
         { "current", NULL, MAVLINK_TYPE_UINT8_T, 0, 7, offsetof(mavlink_mission_item_t, current) }, \
         { "autocontinue", NULL, MAVLINK_TYPE_UINT8_T, 0, 8, offsetof(mavlink_mission_item_t, autocontinue) }, \
         { "param1", NULL, MAVLINK_TYPE_FLOAT, 0, 9, offsetof(mavlink_mission_item_t, param1) }, \
         { "param2", NULL, MAVLINK_TYPE_FLOAT, 0, 13, offsetof(mavlink_mission_item_t, param2) }, \
         { "param3", NULL, MAVLINK_TYPE_FLOAT, 0, 17, offsetof(mavlink_mission_item_t, param3) }, \
         { "param4", NULL, MAVLINK_TYPE_FLOAT, 0, 21, offsetof(mavlink_mission_item_t, param4) }, \
         { "x", NULL, MAVLINK_TYPE_FLOAT, 0, 25, offsetof(mavlink_mission_item_t, x) }, \
         { "y", NULL, MAVLINK_TYPE_FLOAT, 0, 29, offsetof(mavlink_mission_item_t, y) }, \
         { "z", NULL, MAVLINK_TYPE_FLOAT, 0, 33, offsetof(mavlink_mission_item_t, z) }, \
         } \
}
#endif

/**
 * @brief Pack a mission_item message
 * @param system_id ID of this system
 * @param component_id ID of this component (e.g. 200 for IMU)
 * @param msg The MAVLink message to compress the data into
 *
 * @param target_system System ID
 * @param target_component Component ID
 * @param seq Sequence
 * @param frame The coordinate system of the MISSION. see MAV_FRAME in mavlink_types.h
 * @param command The scheduled action for the MISSION. see MAV_CMD in common.xml MAVLink specs
 * @param current false:0, true:1
 * @param autocontinue autocontinue to next wp
 * @param param1 PARAM1, see MAV_CMD enum
 * @param param2 PARAM2, see MAV_CMD enum
 * @param param3 PARAM3, see MAV_CMD enum
 * @param param4 PARAM4, see MAV_CMD enum
 * @param x PARAM5 / local: x position, global: latitude
 * @param y PARAM6 / y position: global: longitude
 * @param z PARAM7 / z position: global: altitude (relative or absolute, depending on frame.
 * @return length of the message in bytes (excluding serial stream start sign)
 */
static inline uint16_t mavlink_msg_mission_item_pack(uint8_t system_id, uint8_t component_id, mavlink_message_t* msg,
                               uint8_t target_system, uint8_t target_component, uint16_t seq, uint8_t frame, uint16_t command, uint8_t current, uint8_t autocontinue, float param1, float param2, float param3, float param4, float x, float y, float z)
{
#if MAVLINK_NEED_BYTE_SWAP || !MAVLINK_ALIGNED_FIELDS
    char buf[MAVLINK_MSG_ID_MISSION_ITEM_LEN];
    _mav_put_uint8_t(buf, 0, target_system);
    _mav_put_uint8_t(buf, 1, target_component);
    _mav_put_uint16_t(buf, 2, seq);
    _mav_put_uint8_t(buf, 4, frame);
    _mav_put_uint16_t(buf, 5, command);
    _mav_put_uint8_t(buf, 7, current);
    _mav_put_uint8_t(buf, 8, autocontinue);
    _mav_put_float(buf, 9, param1);
    _mav_put_float(buf, 13, param2);
    _mav_put_float(buf, 17, param3);
    _mav_put_float(buf, 21, param4);
    _mav_put_float(buf, 25, x);
    _mav_put_float(buf, 29, y);
    _mav_put_float(buf, 33, z);

        memcpy(_MAV_PAYLOAD_NON_CONST(msg), buf, MAVLINK_MSG_ID_MISSION_ITEM_LEN);
#else
    mavlink_mission_item_t packet;
    packet.target_system = target_system;
    packet.target_component = target_component;
    packet.seq = seq;
    packet.frame = frame;
    packet.command = command;
    packet.current = current;
    packet.autocontinue = autocontinue;
    packet.param1 = param1;
    packet.param2 = param2;
    packet.param3 = param3;
    packet.param4 = param4;
    packet.x = x;
    packet.y = y;
    packet.z = z;

        memcpy(_MAV_PAYLOAD_NON_CONST(msg), &packet, MAVLINK_MSG_ID_MISSION_ITEM_LEN);
#endif

    msg->msgid = MAVLINK_MSG_ID_MISSION_ITEM;
    return mavlink_finalize_message(msg, system_id, component_id, MAVLINK_MSG_ID_MISSION_ITEM_MIN_LEN, MAVLINK_MSG_ID_MISSION_ITEM_LEN, MAVLINK_MSG_ID_MISSION_ITEM_CRC);
}

/**
 * @brief Pack a mission_item message on a channel
 * @param system_id ID of this system
 * @param component_id ID of this component (e.g. 200 for IMU)
 * @param chan The MAVLink channel this message will be sent over
 * @param msg The MAVLink message to compress the data into
 * @param target_system System ID
 * @param target_component Component ID
 * @param seq Sequence
 * @param frame The coordinate system of the MISSION. see MAV_FRAME in mavlink_types.h
 * @param command The scheduled action for the MISSION. see MAV_CMD in common.xml MAVLink specs
 * @param current false:0, true:1
 * @param autocontinue autocontinue to next wp
 * @param param1 PARAM1, see MAV_CMD enum
 * @param param2 PARAM2, see MAV_CMD enum
 * @param param3 PARAM3, see MAV_CMD enum
 * @param param4 PARAM4, see MAV_CMD enum
 * @param x PARAM5 / local: x position, global: latitude
 * @param y PARAM6 / y position: global: longitude
 * @param z PARAM7 / z position: global: altitude (relative or absolute, depending on frame.
 * @return length of the message in bytes (excluding serial stream start sign)
 */
static inline uint16_t mavlink_msg_mission_item_pack_chan(uint8_t system_id, uint8_t component_id, uint8_t chan,
                               mavlink_message_t* msg,
                                   uint8_t target_system,uint8_t target_component,uint16_t seq,uint8_t frame,uint16_t command,uint8_t current,uint8_t autocontinue,float param1,float param2,float param3,float param4,float x,float y,float z)
{
#if MAVLINK_NEED_BYTE_SWAP || !MAVLINK_ALIGNED_FIELDS
    char buf[MAVLINK_MSG_ID_MISSION_ITEM_LEN];
    _mav_put_uint8_t(buf, 0, target_system);
    _mav_put_uint8_t(buf, 1, target_component);
    _mav_put_uint16_t(buf, 2, seq);
    _mav_put_uint8_t(buf, 4, frame);
    _mav_put_uint16_t(buf, 5, command);
    _mav_put_uint8_t(buf, 7, current);
    _mav_put_uint8_t(buf, 8, autocontinue);
    _mav_put_float(buf, 9, param1);
    _mav_put_float(buf, 13, param2);
    _mav_put_float(buf, 17, param3);
    _mav_put_float(buf, 21, param4);
    _mav_put_float(buf, 25, x);
    _mav_put_float(buf, 29, y);
    _mav_put_float(buf, 33, z);

        memcpy(_MAV_PAYLOAD_NON_CONST(msg), buf, MAVLINK_MSG_ID_MISSION_ITEM_LEN);
#else
    mavlink_mission_item_t packet;
    packet.target_system = target_system;
    packet.target_component = target_component;
    packet.seq = seq;
    packet.frame = frame;
    packet.command = command;
    packet.current = current;
    packet.autocontinue = autocontinue;
    packet.param1 = param1;
    packet.param2 = param2;
    packet.param3 = param3;
    packet.param4 = param4;
    packet.x = x;
    packet.y = y;
    packet.z = z;

        memcpy(_MAV_PAYLOAD_NON_CONST(msg), &packet, MAVLINK_MSG_ID_MISSION_ITEM_LEN);
#endif

    msg->msgid = MAVLINK_MSG_ID_MISSION_ITEM;
    return mavlink_finalize_message_chan(msg, system_id, component_id, chan, MAVLINK_MSG_ID_MISSION_ITEM_MIN_LEN, MAVLINK_MSG_ID_MISSION_ITEM_LEN, MAVLINK_MSG_ID_MISSION_ITEM_CRC);
}

/**
 * @brief Encode a mission_item struct
 *
 * @param system_id ID of this system
 * @param component_id ID of this component (e.g. 200 for IMU)
 * @param msg The MAVLink message to compress the data into
 * @param mission_item C-struct to read the message contents from
 */
static inline uint16_t mavlink_msg_mission_item_encode(uint8_t system_id, uint8_t component_id, mavlink_message_t* msg, const mavlink_mission_item_t* mission_item)
{
    return mavlink_msg_mission_item_pack(system_id, component_id, msg, mission_item->target_system, mission_item->target_component, mission_item->seq, mission_item->frame, mission_item->command, mission_item->current, mission_item->autocontinue, mission_item->param1, mission_item->param2, mission_item->param3, mission_item->param4, mission_item->x, mission_item->y, mission_item->z);
}

/**
 * @brief Encode a mission_item struct on a channel
 *
 * @param system_id ID of this system
 * @param component_id ID of this component (e.g. 200 for IMU)
 * @param chan The MAVLink channel this message will be sent over
 * @param msg The MAVLink message to compress the data into
 * @param mission_item C-struct to read the message contents from
 */
static inline uint16_t mavlink_msg_mission_item_encode_chan(uint8_t system_id, uint8_t component_id, uint8_t chan, mavlink_message_t* msg, const mavlink_mission_item_t* mission_item)
{
    return mavlink_msg_mission_item_pack_chan(system_id, component_id, chan, msg, mission_item->target_system, mission_item->target_component, mission_item->seq, mission_item->frame, mission_item->command, mission_item->current, mission_item->autocontinue, mission_item->param1, mission_item->param2, mission_item->param3, mission_item->param4, mission_item->x, mission_item->y, mission_item->z);
}

/**
 * @brief Send a mission_item message
 * @param chan MAVLink channel to send the message
 *
 * @param target_system System ID
 * @param target_component Component ID
 * @param seq Sequence
 * @param frame The coordinate system of the MISSION. see MAV_FRAME in mavlink_types.h
 * @param command The scheduled action for the MISSION. see MAV_CMD in common.xml MAVLink specs
 * @param current false:0, true:1
 * @param autocontinue autocontinue to next wp
 * @param param1 PARAM1, see MAV_CMD enum
 * @param param2 PARAM2, see MAV_CMD enum
 * @param param3 PARAM3, see MAV_CMD enum
 * @param param4 PARAM4, see MAV_CMD enum
 * @param x PARAM5 / local: x position, global: latitude
 * @param y PARAM6 / y position: global: longitude
 * @param z PARAM7 / z position: global: altitude (relative or absolute, depending on frame.
 */
#ifdef MAVLINK_USE_CONVENIENCE_FUNCTIONS

static inline void mavlink_msg_mission_item_send(mavlink_channel_t chan, uint8_t target_system, uint8_t target_component, uint16_t seq, uint8_t frame, uint16_t command, uint8_t current, uint8_t autocontinue, float param1, float param2, float param3, float param4, float x, float y, float z)
{
#if MAVLINK_NEED_BYTE_SWAP || !MAVLINK_ALIGNED_FIELDS
    char buf[MAVLINK_MSG_ID_MISSION_ITEM_LEN];
    _mav_put_uint8_t(buf, 0, target_system);
    _mav_put_uint8_t(buf, 1, target_component);
    _mav_put_uint16_t(buf, 2, seq);
    _mav_put_uint8_t(buf, 4, frame);
    _mav_put_uint16_t(buf, 5, command);
    _mav_put_uint8_t(buf, 7, current);
    _mav_put_uint8_t(buf, 8, autocontinue);
    _mav_put_float(buf, 9, param1);
    _mav_put_float(buf, 13, param2);
    _mav_put_float(buf, 17, param3);
    _mav_put_float(buf, 21, param4);
    _mav_put_float(buf, 25, x);
    _mav_put_float(buf, 29, y);
    _mav_put_float(buf, 33, z);

    _mav_finalize_message_chan_send(chan, MAVLINK_MSG_ID_MISSION_ITEM, buf, MAVLINK_MSG_ID_MISSION_ITEM_MIN_LEN, MAVLINK_MSG_ID_MISSION_ITEM_LEN, MAVLINK_MSG_ID_MISSION_ITEM_CRC);
#else
    mavlink_mission_item_t packet;
    packet.target_system = target_system;
    packet.target_component = target_component;
    packet.seq = seq;
    packet.frame = frame;
    packet.command = command;
    packet.current = current;
    packet.autocontinue = autocontinue;
    packet.param1 = param1;
    packet.param2 = param2;
    packet.param3 = param3;
    packet.param4 = param4;
    packet.x = x;
    packet.y = y;
    packet.z = z;

    _mav_finalize_message_chan_send(chan, MAVLINK_MSG_ID_MISSION_ITEM, (const char *)&packet, MAVLINK_MSG_ID_MISSION_ITEM_MIN_LEN, MAVLINK_MSG_ID_MISSION_ITEM_LEN, MAVLINK_MSG_ID_MISSION_ITEM_CRC);
#endif
}

/**
 * @brief Send a mission_item message
 * @param chan MAVLink channel to send the message
 * @param struct The MAVLink struct to serialize
 */
static inline void mavlink_msg_mission_item_send_struct(mavlink_channel_t chan, const mavlink_mission_item_t* mission_item)
{
#if MAVLINK_NEED_BYTE_SWAP || !MAVLINK_ALIGNED_FIELDS
    mavlink_msg_mission_item_send(chan, mission_item->target_system, mission_item->target_component, mission_item->seq, mission_item->frame, mission_item->command, mission_item->current, mission_item->autocontinue, mission_item->param1, mission_item->param2, mission_item->param3, mission_item->param4, mission_item->x, mission_item->y, mission_item->z);
#else
    _mav_finalize_message_chan_send(chan, MAVLINK_MSG_ID_MISSION_ITEM, (const char *)mission_item, MAVLINK_MSG_ID_MISSION_ITEM_MIN_LEN, MAVLINK_MSG_ID_MISSION_ITEM_LEN, MAVLINK_MSG_ID_MISSION_ITEM_CRC);
#endif
}

#if MAVLINK_MSG_ID_MISSION_ITEM_LEN <= MAVLINK_MAX_PAYLOAD_LEN
/*
  This varient of _send() can be used to save stack space by re-using
  memory from the receive buffer.  The caller provides a
  mavlink_message_t which is the size of a full mavlink message. This
  is usually the receive buffer for the channel, and allows a reply to an
  incoming message with minimum stack space usage.
 */
static inline void mavlink_msg_mission_item_send_buf(mavlink_message_t *msgbuf, mavlink_channel_t chan,  uint8_t target_system, uint8_t target_component, uint16_t seq, uint8_t frame, uint16_t command, uint8_t current, uint8_t autocontinue, float param1, float param2, float param3, float param4, float x, float y, float z)
{
#if MAVLINK_NEED_BYTE_SWAP || !MAVLINK_ALIGNED_FIELDS
    char *buf = (char *)msgbuf;
    _mav_put_uint8_t(buf, 0, target_system);
    _mav_put_uint8_t(buf, 1, target_component);
    _mav_put_uint16_t(buf, 2, seq);
    _mav_put_uint8_t(buf, 4, frame);
    _mav_put_uint16_t(buf, 5, command);
    _mav_put_uint8_t(buf, 7, current);
    _mav_put_uint8_t(buf, 8, autocontinue);
    _mav_put_float(buf, 9, param1);
    _mav_put_float(buf, 13, param2);
    _mav_put_float(buf, 17, param3);
    _mav_put_float(buf, 21, param4);
    _mav_put_float(buf, 25, x);
    _mav_put_float(buf, 29, y);
    _mav_put_float(buf, 33, z);

    _mav_finalize_message_chan_send(chan, MAVLINK_MSG_ID_MISSION_ITEM, buf, MAVLINK_MSG_ID_MISSION_ITEM_MIN_LEN, MAVLINK_MSG_ID_MISSION_ITEM_LEN, MAVLINK_MSG_ID_MISSION_ITEM_CRC);
#else
    mavlink_mission_item_t *packet = (mavlink_mission_item_t *)msgbuf;
    packet->target_system = target_system;
    packet->target_component = target_component;
    packet->seq = seq;
    packet->frame = frame;
    packet->command = command;
    packet->current = current;
    packet->autocontinue = autocontinue;
    packet->param1 = param1;
    packet->param2 = param2;
    packet->param3 = param3;
    packet->param4 = param4;
    packet->x = x;
    packet->y = y;
    packet->z = z;

    _mav_finalize_message_chan_send(chan, MAVLINK_MSG_ID_MISSION_ITEM, (const char *)packet, MAVLINK_MSG_ID_MISSION_ITEM_MIN_LEN, MAVLINK_MSG_ID_MISSION_ITEM_LEN, MAVLINK_MSG_ID_MISSION_ITEM_CRC);
#endif
}
#endif

#endif

// MESSAGE MISSION_ITEM UNPACKING


/**
 * @brief Get field target_system from mission_item message
 *
 * @return System ID
 */
static inline uint8_t mavlink_msg_mission_item_get_target_system(const mavlink_message_t* msg)
{
    return _MAV_RETURN_uint8_t(msg,  0);
}

/**
 * @brief Get field target_component from mission_item message
 *
 * @return Component ID
 */
static inline uint8_t mavlink_msg_mission_item_get_target_component(const mavlink_message_t* msg)
{
    return _MAV_RETURN_uint8_t(msg,  1);
}

/**
 * @brief Get field seq from mission_item message
 *
 * @return Sequence
 */
static inline uint16_t mavlink_msg_mission_item_get_seq(const mavlink_message_t* msg)
{
    return _MAV_RETURN_uint16_t(msg,  2);
}

/**
 * @brief Get field frame from mission_item message
 *
 * @return The coordinate system of the MISSION. see MAV_FRAME in mavlink_types.h
 */
static inline uint8_t mavlink_msg_mission_item_get_frame(const mavlink_message_t* msg)
{
    return _MAV_RETURN_uint8_t(msg,  4);
}

/**
 * @brief Get field command from mission_item message
 *
 * @return The scheduled action for the MISSION. see MAV_CMD in common.xml MAVLink specs
 */
static inline uint16_t mavlink_msg_mission_item_get_command(const mavlink_message_t* msg)
{
    return _MAV_RETURN_uint16_t(msg,  5);
}

/**
 * @brief Get field current from mission_item message
 *
 * @return false:0, true:1
 */
static inline uint8_t mavlink_msg_mission_item_get_current(const mavlink_message_t* msg)
{
    return _MAV_RETURN_uint8_t(msg,  7);
}

/**
 * @brief Get field autocontinue from mission_item message
 *
 * @return autocontinue to next wp
 */
static inline uint8_t mavlink_msg_mission_item_get_autocontinue(const mavlink_message_t* msg)
{
    return _MAV_RETURN_uint8_t(msg,  8);
}

/**
 * @brief Get field param1 from mission_item message
 *
 * @return PARAM1, see MAV_CMD enum
 */
static inline float mavlink_msg_mission_item_get_param1(const mavlink_message_t* msg)
{
    return _MAV_RETURN_float(msg,  9);
}

/**
 * @brief Get field param2 from mission_item message
 *
 * @return PARAM2, see MAV_CMD enum
 */
static inline float mavlink_msg_mission_item_get_param2(const mavlink_message_t* msg)
{
    return _MAV_RETURN_float(msg,  13);
}

/**
 * @brief Get field param3 from mission_item message
 *
 * @return PARAM3, see MAV_CMD enum
 */
static inline float mavlink_msg_mission_item_get_param3(const mavlink_message_t* msg)
{
    return _MAV_RETURN_float(msg,  17);
}

/**
 * @brief Get field param4 from mission_item message
 *
 * @return PARAM4, see MAV_CMD enum
 */
static inline float mavlink_msg_mission_item_get_param4(const mavlink_message_t* msg)
{
    return _MAV_RETURN_float(msg,  21);
}

/**
 * @brief Get field x from mission_item message
 *
 * @return PARAM5 / local: x position, global: latitude
 */
static inline float mavlink_msg_mission_item_get_x(const mavlink_message_t* msg)
{
    return _MAV_RETURN_float(msg,  25);
}

/**
 * @brief Get field y from mission_item message
 *
 * @return PARAM6 / y position: global: longitude
 */
static inline float mavlink_msg_mission_item_get_y(const mavlink_message_t* msg)
{
    return _MAV_RETURN_float(msg,  29);
}

/**
 * @brief Get field z from mission_item message
 *
 * @return PARAM7 / z position: global: altitude (relative or absolute, depending on frame.
 */
static inline float mavlink_msg_mission_item_get_z(const mavlink_message_t* msg)
{
    return _MAV_RETURN_float(msg,  33);
}

/**
 * @brief Decode a mission_item message into a struct
 *
 * @param msg The message to decode
 * @param mission_item C-struct to decode the message contents into
 */
static inline void mavlink_msg_mission_item_decode(const mavlink_message_t* msg, mavlink_mission_item_t* mission_item)
{
#if MAVLINK_NEED_BYTE_SWAP || !MAVLINK_ALIGNED_FIELDS
    mission_item->target_system = mavlink_msg_mission_item_get_target_system(msg);
    mission_item->target_component = mavlink_msg_mission_item_get_target_component(msg);
    mission_item->seq = mavlink_msg_mission_item_get_seq(msg);
    mission_item->frame = mavlink_msg_mission_item_get_frame(msg);
    mission_item->command = mavlink_msg_mission_item_get_command(msg);
    mission_item->current = mavlink_msg_mission_item_get_current(msg);
    mission_item->autocontinue = mavlink_msg_mission_item_get_autocontinue(msg);
    mission_item->param1 = mavlink_msg_mission_item_get_param1(msg);
    mission_item->param2 = mavlink_msg_mission_item_get_param2(msg);
    mission_item->param3 = mavlink_msg_mission_item_get_param3(msg);
    mission_item->param4 = mavlink_msg_mission_item_get_param4(msg);
    mission_item->x = mavlink_msg_mission_item_get_x(msg);
    mission_item->y = mavlink_msg_mission_item_get_y(msg);
    mission_item->z = mavlink_msg_mission_item_get_z(msg);
#else
        uint8_t len = msg->len < MAVLINK_MSG_ID_MISSION_ITEM_LEN? msg->len : MAVLINK_MSG_ID_MISSION_ITEM_LEN;
        memset(mission_item, 0, MAVLINK_MSG_ID_MISSION_ITEM_LEN);
    memcpy(mission_item, _MAV_PAYLOAD(msg), len);
#endif
}
