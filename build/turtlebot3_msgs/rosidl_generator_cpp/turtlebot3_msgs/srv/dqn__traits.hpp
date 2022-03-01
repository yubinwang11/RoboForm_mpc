// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from turtlebot3_msgs:srv/Dqn.idl
// generated code does not contain a copyright notice

#ifndef TURTLEBOT3_MSGS__SRV__DQN__TRAITS_HPP_
#define TURTLEBOT3_MSGS__SRV__DQN__TRAITS_HPP_

#include "turtlebot3_msgs/srv/dqn__struct.hpp"
#include <rosidl_generator_cpp/traits.hpp>
#include <stdint.h>
#include <type_traits>

namespace rosidl_generator_traits
{

template<>
inline const char * data_type<turtlebot3_msgs::srv::Dqn_Request>()
{
  return "turtlebot3_msgs::srv::Dqn_Request";
}

template<>
struct has_fixed_size<turtlebot3_msgs::srv::Dqn_Request>
  : std::integral_constant<bool, true> {};

template<>
struct has_bounded_size<turtlebot3_msgs::srv::Dqn_Request>
  : std::integral_constant<bool, true> {};

}  // namespace rosidl_generator_traits

namespace rosidl_generator_traits
{

template<>
inline const char * data_type<turtlebot3_msgs::srv::Dqn_Response>()
{
  return "turtlebot3_msgs::srv::Dqn_Response";
}

template<>
struct has_fixed_size<turtlebot3_msgs::srv::Dqn_Response>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<turtlebot3_msgs::srv::Dqn_Response>
  : std::integral_constant<bool, false> {};

}  // namespace rosidl_generator_traits

namespace rosidl_generator_traits
{

template<>
inline const char * data_type<turtlebot3_msgs::srv::Dqn>()
{
  return "turtlebot3_msgs::srv::Dqn";
}

template<>
struct has_fixed_size<turtlebot3_msgs::srv::Dqn>
  : std::integral_constant<
    bool,
    has_fixed_size<turtlebot3_msgs::srv::Dqn_Request>::value &&
    has_fixed_size<turtlebot3_msgs::srv::Dqn_Response>::value
  >
{
};

template<>
struct has_bounded_size<turtlebot3_msgs::srv::Dqn>
  : std::integral_constant<
    bool,
    has_bounded_size<turtlebot3_msgs::srv::Dqn_Request>::value &&
    has_bounded_size<turtlebot3_msgs::srv::Dqn_Response>::value
  >
{
};

}  // namespace rosidl_generator_traits

#endif  // TURTLEBOT3_MSGS__SRV__DQN__TRAITS_HPP_
