/*
 * Copyright (C) 2020 Open Source Robotics Foundation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
*/

#ifndef SRC__RMF_FLEET_ADAPTER__PHASES__DISPENSEITEM_HPP
#define SRC__RMF_FLEET_ADAPTER__PHASES__DISPENSEITEM_HPP

#include "RxOperators.hpp"
#include "../Task.hpp"
#include "rmf_fleet_adapter/StandardNames.hpp"

#include <rmf_rxcpp/Transport.hpp>
#include <rmf_dispenser_msgs/msg/dispenser_request.hpp>
#include <rmf_dispenser_msgs/msg/dispenser_result.hpp>
#include <rmf_dispenser_msgs/msg/dispenser_state.hpp>

namespace rmf_fleet_adapter {
namespace phases {

struct DispenseItem
{
  class Action
  {
  public:

    Action(
      const std::shared_ptr<rmf_rxcpp::Transport>& transport,
      std::string request_guid,
      std::string target,
      std::string transporter_type,
      std::vector<rmf_dispenser_msgs::msg::DispenserRequestItem> items,
      rxcpp::observable<rmf_dispenser_msgs::msg::DispenserResult::SharedPtr> result_obs,
      rxcpp::observable<rmf_dispenser_msgs::msg::DispenserState::SharedPtr> state_obs,
      rclcpp::Publisher<rmf_dispenser_msgs::msg::DispenserRequest>::SharedPtr request_pub);

    inline const rxcpp::observable<Task::StatusMsg>& get_observable() const
    {
      return _obs;
    }

  private:

    std::weak_ptr<rmf_rxcpp::Transport> _transport;
    std::string _request_guid;
    std::string _target;
    std::string _transporter_type;
    std::vector<rmf_dispenser_msgs::msg::DispenserRequestItem> _items;
    rxcpp::observable<rmf_dispenser_msgs::msg::DispenserResult::SharedPtr> _result_obs;
    rxcpp::observable<rmf_dispenser_msgs::msg::DispenserState::SharedPtr> _state_obs;
    rclcpp::Publisher<rmf_dispenser_msgs::msg::DispenserRequest>::SharedPtr _request_pub;
    rxcpp::observable<Task::StatusMsg> _obs;
    rclcpp::TimerBase::SharedPtr _timer;
    bool _request_acknowledged = false;

    Task::StatusMsg _get_status(
      const rmf_dispenser_msgs::msg::DispenserResult::SharedPtr& dispenser_result,
      const rmf_dispenser_msgs::msg::DispenserState::SharedPtr& dispenser_state);

    void _do_publish();
  };

  class ActivePhase : public Task::ActivePhase
  {
  public:

    ActivePhase(
      const std::shared_ptr<rmf_rxcpp::Transport>& transport,
      std::string request_guid,
      std::string target,
      std::string transporter_type,
      std::vector<rmf_dispenser_msgs::msg::DispenserRequestItem> items,
      rxcpp::observable<rmf_dispenser_msgs::msg::DispenserResult::SharedPtr> result_obs,
      rxcpp::observable<rmf_dispenser_msgs::msg::DispenserState::SharedPtr> state_obs,
      rclcpp::Publisher<rmf_dispenser_msgs::msg::DispenserRequest>::SharedPtr request_pub);

    const rxcpp::observable<Task::StatusMsg>& observe() const override;

    rmf_traffic::Duration estimate_remaining_time() const override;

    void emergency_alarm(bool on) override;

    void cancel() override;

    const std::string& description() const override;

  private:

    std::weak_ptr<rmf_rxcpp::Transport> _transport;
    std::string _request_guid;
    std::string _target;
    std::string _transporter_type;
    std::vector<rmf_dispenser_msgs::msg::DispenserRequestItem> _items;
    rxcpp::observable<rmf_dispenser_msgs::msg::DispenserResult::SharedPtr> _result_obs;
    rxcpp::observable<rmf_dispenser_msgs::msg::DispenserState::SharedPtr> _state_obs;
    std::string _description;
    Action _action;
  };

  class PendingPhase : public Task::PendingPhase
  {
  public:

    PendingPhase(
      std::weak_ptr<rmf_rxcpp::Transport> transport,
      std::string request_guid,
      std::string target,
      std::string transporter_type,
      std::vector<rmf_dispenser_msgs::msg::DispenserRequestItem> items,
      rxcpp::observable<rmf_dispenser_msgs::msg::DispenserResult::SharedPtr> result_obs,
      rxcpp::observable<rmf_dispenser_msgs::msg::DispenserState::SharedPtr> state_obs,
      rclcpp::Publisher<rmf_dispenser_msgs::msg::DispenserRequest>::SharedPtr request_pub);

    std::shared_ptr<Task::ActivePhase> begin() override;

    rmf_traffic::Duration estimate_phase_duration() const override;

    const std::string& description() const override;

  private:

    std::weak_ptr<rmf_rxcpp::Transport> _transport;
    std::string _request_guid;
    std::string _target;
    std::string _transporter_type;
    std::vector<rmf_dispenser_msgs::msg::DispenserRequestItem> _items;
    rxcpp::observable<rmf_dispenser_msgs::msg::DispenserResult::SharedPtr> _result_obs;
    rxcpp::observable<rmf_dispenser_msgs::msg::DispenserState::SharedPtr> _state_obs;
    rclcpp::Publisher<rmf_dispenser_msgs::msg::DispenserRequest>::SharedPtr _request_pub;
    std::string _description;
  };
};

} // namespace phases
} // namespace rmf_fleet_adapter

#endif // SRC__RMF_FLEET_ADAPTER__PHASES__DISPENSEITEM_HPP
