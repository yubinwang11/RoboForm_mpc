// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from turtlebot3_msgs:msg/Sound.idl
// generated code does not contain a copyright notice

#ifndef TURTLEBOT3_MSGS__MSG__SOUND__FUNCTIONS_H_
#define TURTLEBOT3_MSGS__MSG__SOUND__FUNCTIONS_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stdlib.h>

#include "rosidl_generator_c/visibility_control.h"
#include "turtlebot3_msgs/msg/rosidl_generator_c__visibility_control.h"

#include "turtlebot3_msgs/msg/sound__struct.h"

/// Initialize msg/Sound message.
/**
 * If the init function is called twice for the same message without
 * calling fini inbetween previously allocated memory will be leaked.
 * \param[in,out] msg The previously allocated message pointer.
 * Fields without a default value will not be initialized by this function.
 * You might want to call memset(msg, 0, sizeof(
 * turtlebot3_msgs__msg__Sound
 * )) before or use
 * turtlebot3_msgs__msg__Sound__create()
 * to allocate and initialize the message.
 * \return true if initialization was successful, otherwise false
 */
ROSIDL_GENERATOR_C_PUBLIC_turtlebot3_msgs
bool
turtlebot3_msgs__msg__Sound__init(turtlebot3_msgs__msg__Sound * msg);

/// Finalize msg/Sound message.
/**
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_turtlebot3_msgs
void
turtlebot3_msgs__msg__Sound__fini(turtlebot3_msgs__msg__Sound * msg);

/// Create msg/Sound message.
/**
 * It allocates the memory for the message, sets the memory to zero, and
 * calls
 * turtlebot3_msgs__msg__Sound__init().
 * \return The pointer to the initialized message if successful,
 * otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_turtlebot3_msgs
turtlebot3_msgs__msg__Sound *
turtlebot3_msgs__msg__Sound__create();

/// Destroy msg/Sound message.
/**
 * It calls
 * turtlebot3_msgs__msg__Sound__fini()
 * and frees the memory of the message.
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_turtlebot3_msgs
void
turtlebot3_msgs__msg__Sound__destroy(turtlebot3_msgs__msg__Sound * msg);


/// Initialize array of msg/Sound messages.
/**
 * It allocates the memory for the number of elements and calls
 * turtlebot3_msgs__msg__Sound__init()
 * for each element of the array.
 * \param[in,out] array The allocated array pointer.
 * \param[in] size The size / capacity of the array.
 * \return true if initialization was successful, otherwise false
 * If the array pointer is valid and the size is zero it is guaranteed
 # to return true.
 */
ROSIDL_GENERATOR_C_PUBLIC_turtlebot3_msgs
bool
turtlebot3_msgs__msg__Sound__Sequence__init(turtlebot3_msgs__msg__Sound__Sequence * array, size_t size);

/// Finalize array of msg/Sound messages.
/**
 * It calls
 * turtlebot3_msgs__msg__Sound__fini()
 * for each element of the array and frees the memory for the number of
 * elements.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_turtlebot3_msgs
void
turtlebot3_msgs__msg__Sound__Sequence__fini(turtlebot3_msgs__msg__Sound__Sequence * array);

/// Create array of msg/Sound messages.
/**
 * It allocates the memory for the array and calls
 * turtlebot3_msgs__msg__Sound__Sequence__init().
 * \param[in] size The size / capacity of the array.
 * \return The pointer to the initialized array if successful, otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_turtlebot3_msgs
turtlebot3_msgs__msg__Sound__Sequence *
turtlebot3_msgs__msg__Sound__Sequence__create(size_t size);

/// Destroy array of msg/Sound messages.
/**
 * It calls
 * turtlebot3_msgs__msg__Sound__Sequence__fini()
 * on the array,
 * and frees the memory of the array.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_turtlebot3_msgs
void
turtlebot3_msgs__msg__Sound__Sequence__destroy(turtlebot3_msgs__msg__Sound__Sequence * array);

#ifdef __cplusplus
}
#endif

#endif  // TURTLEBOT3_MSGS__MSG__SOUND__FUNCTIONS_H_
