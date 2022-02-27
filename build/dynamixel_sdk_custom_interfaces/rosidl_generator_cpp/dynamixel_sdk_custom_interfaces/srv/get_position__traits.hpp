// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from dynamixel_sdk_custom_interfaces:srv/GetPosition.idl
// generated code does not contain a copyright notice

#ifndef DYNAMIXEL_SDK_CUSTOM_INTERFACES__SRV__GET_POSITION__TRAITS_HPP_
#define DYNAMIXEL_SDK_CUSTOM_INTERFACES__SRV__GET_POSITION__TRAITS_HPP_

#include "dynamixel_sdk_custom_interfaces/srv/get_position__struct.hpp"
#include <rosidl_generator_cpp/traits.hpp>
#include <stdint.h>
#include <type_traits>

namespace rosidl_generator_traits
{

template<>
inline const char * data_type<dynamixel_sdk_custom_interfaces::srv::GetPosition_Request>()
{
  return "dynamixel_sdk_custom_interfaces::srv::GetPosition_Request";
}

template<>
struct has_fixed_size<dynamixel_sdk_custom_interfaces::srv::GetPosition_Request>
  : std::integral_constant<bool, true> {};

template<>
struct has_bounded_size<dynamixel_sdk_custom_interfaces::srv::GetPosition_Request>
  : std::integral_constant<bool, true> {};

}  // namespace rosidl_generator_traits

namespace rosidl_generator_traits
{

template<>
inline const char * data_type<dynamixel_sdk_custom_interfaces::srv::GetPosition_Response>()
{
  return "dynamixel_sdk_custom_interfaces::srv::GetPosition_Response";
}

template<>
struct has_fixed_size<dynamixel_sdk_custom_interfaces::srv::GetPosition_Response>
  : std::integral_constant<bool, true> {};

template<>
struct has_bounded_size<dynamixel_sdk_custom_interfaces::srv::GetPosition_Response>
  : std::integral_constant<bool, true> {};

}  // namespace rosidl_generator_traits

namespace rosidl_generator_traits
{

template<>
inline const char * data_type<dynamixel_sdk_custom_interfaces::srv::GetPosition>()
{
  return "dynamixel_sdk_custom_interfaces::srv::GetPosition";
}

template<>
struct has_fixed_size<dynamixel_sdk_custom_interfaces::srv::GetPosition>
  : std::integral_constant<
    bool,
    has_fixed_size<dynamixel_sdk_custom_interfaces::srv::GetPosition_Request>::value &&
    has_fixed_size<dynamixel_sdk_custom_interfaces::srv::GetPosition_Response>::value
  >
{
};

template<>
struct has_bounded_size<dynamixel_sdk_custom_interfaces::srv::GetPosition>
  : std::integral_constant<
    bool,
    has_bounded_size<dynamixel_sdk_custom_interfaces::srv::GetPosition_Request>::value &&
    has_bounded_size<dynamixel_sdk_custom_interfaces::srv::GetPosition_Response>::value
  >
{
};

}  // namespace rosidl_generator_traits

#endif  // DYNAMIXEL_SDK_CUSTOM_INTERFACES__SRV__GET_POSITION__TRAITS_HPP_
