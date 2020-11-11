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


#include <rmf_task/agv/TaskPlanner.hpp>
#include <rmf_task/agv/State.hpp>
#include <rmf_task/agv/StateConfig.hpp>
#include <rmf_task/requests/Delivery.hpp>
#include <rmf_task/requests/ChargeBattery.hpp>

#include <rmf_traffic/agv/Graph.hpp>
#include <rmf_traffic/Trajectory.hpp>
#include <rmf_traffic/Time.hpp>
#include <rmf_traffic/agv/Planner.hpp>
#include <rmf_traffic/Profile.hpp>
#include <rmf_traffic/agv/VehicleTraits.hpp>
#include <rmf_traffic/geometry/Circle.hpp>
#include <rmf_traffic/schedule/Viewer.hpp>
#include <rmf_traffic/schedule/Database.hpp>

#include <rmf_battery/agv/SimpleDevicePowerSink.hpp>
#include <rmf_battery/agv/SimpleMotionPowerSink.hpp>
#include <rmf_battery/agv/BatterySystem.hpp>

#include <rmf_utils/catch.hpp>

#include <iostream>
#include <vector>
#include <tuple>
#include <utility>
#include <random>

using Assignments = std::vector<std::vector<rmf_task::agv::TaskPlanner::Assignment>>;

std::pair<std::vector<std::vector<std::tuple<int,int,int>>>, std::vector<std::vector<int>>>
  generate_testcases(size_t max, std::vector<std::vector<size_t>> test_profile, size_t num_testcases);
std::pair<std::vector<std::tuple<int,int,int>>, std::vector<int>>
  generate_testcase(size_t max, std::vector<std::vector<size_t>> test_profile, std::mt19937& eng);
void run_tests(std::vector<std::vector<std::tuple<int,int,int>>> tests,
  std::vector<std::vector<int>> test_waypoints,
  const rmf_battery::agv::BatterySystem& battery_system,
  std::shared_ptr<rmf_traffic::agv::Planner> planner,
  std::shared_ptr<rmf_battery::agv::SimpleMotionPowerSink> motion_sink,
  std::shared_ptr<rmf_battery::agv::SimpleDevicePowerSink> device_sink,
  bool drain_battery,
  bool optimal=true);
std::pair<Assignments, double> compute_assignments(const std::vector<std::tuple<int,int,int>>& request_data,
  size_t initial_pt,
  size_t initial_pt2,
  size_t charging_pt,
  size_t charging_pt2,
  const rmf_battery::agv::BatterySystem& battery_system,
  std::shared_ptr<rmf_traffic::agv::Planner> planner,
  std::shared_ptr<rmf_battery::agv::SimpleMotionPowerSink> motion_sink,
  std::shared_ptr<rmf_battery::agv::SimpleDevicePowerSink> device_sink,
  bool drain_battery,
  bool optimal=true);

//==============================================================================
inline void display_solution(
  std::string parent,
  const rmf_task::agv::TaskPlanner::Assignments& assignments,
  const double cost)
{
  std::cout << parent << " cost: " << cost << std::endl;
  std::cout << parent << " assignments:" << std::endl;
  for (std::size_t i = 0; i < assignments.size(); ++i)
  {
    std:: cout << "--Agent: " << i << std::endl;
    for (const auto& a : assignments[i])
    {
      const auto& s = a.state();
      const double request_seconds = a.request()->earliest_start_time().time_since_epoch().count()/1e9;
      const double start_seconds = a.deployment_time().time_since_epoch().count()/1e9;
      const rmf_traffic::Time finish_time = s.finish_time();
      const double finish_seconds = finish_time.time_since_epoch().count()/1e9;
      std::cout << "    <" << a.request()->id() << ": " << request_seconds
                << ", " << start_seconds 
                << ", "<< finish_seconds << ", " << 100* s.battery_soc() 
                << "%>" << std::endl;
    }
  }
  std::cout << " ----------------------" << std::endl;
}

//==============================================================================
SCENARIO("Grid World")
{
  const int grid_size = 4;
  const double edge_length = 1000;
  const bool drain_battery = true;

  using SimpleMotionPowerSink = rmf_battery::agv::SimpleMotionPowerSink;
  using SimpleDevicePowerSink = rmf_battery::agv::SimpleDevicePowerSink;

  rmf_traffic::agv::Graph graph;
  auto add_bidir_lane = [&](const std::size_t w0, const std::size_t w1)
  {
      graph.add_lane(w0, w1);
      graph.add_lane(w1, w0);
  };

  const std::string map_name = "test_map";

  for (int i = 0; i < grid_size; ++i)
  {
    for (int j = 0; j < grid_size; ++j)
    {
      // const auto random = (double) rand() / RAND_MAX;
      const double random = 1.0;
      graph.add_waypoint(map_name, 
        {j*edge_length*random, -i*edge_length*random});
    }
  }

  for (int i = 0; i < grid_size*grid_size; ++i)
  {
    if ((i+1) % grid_size != 0)
      add_bidir_lane(i, i+1);
    if (i + grid_size < grid_size*grid_size)
      add_bidir_lane(i, i+4);
  }

  const auto shape = rmf_traffic::geometry::make_final_convex<
    rmf_traffic::geometry::Circle>(1.0);
  const rmf_traffic::Profile profile{shape, shape};
  const rmf_traffic::agv::VehicleTraits traits(
    {1.0, 0.7}, {0.6, 0.5}, profile);
  rmf_traffic::schedule::Database database;
  const auto default_options = rmf_traffic::agv::Planner::Options{
    nullptr};
    
  auto planner = std::make_shared<rmf_traffic::agv::Planner>(
      rmf_traffic::agv::Planner::Configuration{graph, traits},
      default_options);

  rmf_battery::agv::BatterySystem battery_system{24.0, 40.0, 8.8};
  REQUIRE(battery_system.valid());
  rmf_battery::agv::MechanicalSystem mechanical_system{70.0, 40.0, 0.22};
  REQUIRE(mechanical_system.valid());
  rmf_battery::agv::PowerSystem power_system{"processor", 20.0};
  REQUIRE(power_system.valid());

  std::shared_ptr<SimpleMotionPowerSink> motion_sink =
    std::make_shared<SimpleMotionPowerSink>(battery_system, mechanical_system);
  std::shared_ptr<SimpleDevicePowerSink> device_sink =
    std::make_shared<SimpleDevicePowerSink>(battery_system, power_system);

  /*WHEN("Planning for 3 requests and 2 agents")
  {
    const auto now = std::chrono::steady_clock::now();
    const double default_orientation = 0.0;

    rmf_traffic::agv::Plan::Start first_location{now, 13, default_orientation};
    rmf_traffic::agv::Plan::Start second_location{now, 2, default_orientation};

    std::vector<rmf_task::agv::State> initial_states =
    {
      rmf_task::agv::State{first_location, 13, 1.0},
      rmf_task::agv::State{second_location, 2, 1.0}
    };

    std::vector<rmf_task::agv::StateConfig> state_configs =
    {
      rmf_task::agv::StateConfig{0.2},
      rmf_task::agv::StateConfig{0.2}
    };

    std::vector<rmf_task::Request::SharedPtr> requests =
    {
      rmf_task::requests::Delivery::make(
        1,
        0,
        3,
        motion_sink,
        device_sink,
        planner,
        now + rmf_traffic::time::from_seconds(0),
        drain_battery),

      rmf_task::requests::Delivery::make(
        2,
        15,
        2,
        motion_sink,
        device_sink,
        planner,
        now + rmf_traffic::time::from_seconds(0),
        drain_battery),

      rmf_task::requests::Delivery::make(
        3,
        7,
        9,
        motion_sink,
        device_sink,
        planner,
        now + rmf_traffic::time::from_seconds(0),
        drain_battery)
    };

    std::shared_ptr<rmf_task::agv::TaskPlanner::Configuration>  task_config =
      std::make_shared<rmf_task::agv::TaskPlanner::Configuration>(
        battery_system,
        motion_sink,
        device_sink,
        planner);
    rmf_task::agv::TaskPlanner task_planner(task_config);

    const auto greedy_assignments = task_planner.greedy_plan(
      now, initial_states, state_configs, requests);
    const double greedy_cost = task_planner.compute_cost(greedy_assignments);

    const auto optimal_assignments = task_planner.optimal_plan(
      now, initial_states, state_configs, requests, nullptr);
    const double optimal_cost = task_planner.compute_cost(optimal_assignments);
    
    display_solution("Greedy", greedy_assignments, greedy_cost);
    display_solution("Optimal", optimal_assignments, optimal_cost);

    REQUIRE(optimal_cost <= greedy_cost);
  }

  WHEN("Planning for 11 requests and 2 agents")
  {
    const auto now = std::chrono::steady_clock::now();
    const double default_orientation = 0.0;

    rmf_traffic::agv::Plan::Start first_location{now, 13, default_orientation};
    rmf_traffic::agv::Plan::Start second_location{now, 2, default_orientation};

    std::vector<rmf_task::agv::State> initial_states =
    {
      rmf_task::agv::State{first_location, 13, 1.0},
      rmf_task::agv::State{second_location, 2, 1.0}
    };

    std::vector<rmf_task::agv::StateConfig> state_configs =
    {
      rmf_task::agv::StateConfig{0.2},
      rmf_task::agv::StateConfig{0.2}
    };

    std::vector<rmf_task::Request::SharedPtr> requests =
    {
      rmf_task::requests::Delivery::make(
        1,
        0,
        3,
        motion_sink,
        device_sink,
        planner,
        now + rmf_traffic::time::from_seconds(0),
        drain_battery),

      rmf_task::requests::Delivery::make(
        2,
        15,
        2,
        motion_sink,
        device_sink,
        planner,
        now + rmf_traffic::time::from_seconds(0),
        drain_battery),

      rmf_task::requests::Delivery::make(
        3,
        7,
        9,
        motion_sink,
        device_sink,
        planner,
        now + rmf_traffic::time::from_seconds(0),
        drain_battery),

      rmf_task::requests::Delivery::make(
        4,
        8,
        11,
        motion_sink,
        device_sink,
        planner,
        now + rmf_traffic::time::from_seconds(50000),
        drain_battery),

      rmf_task::requests::Delivery::make(
        5,
        10,
        0,
        motion_sink,
        device_sink,
        planner,
        now + rmf_traffic::time::from_seconds(50000),
        drain_battery),

      rmf_task::requests::Delivery::make(
        6,
        4,
        8,
        motion_sink,
        device_sink,
        planner,
        now + rmf_traffic::time::from_seconds(60000),
        drain_battery),

      rmf_task::requests::Delivery::make(
        7,
        8,
        14,
        motion_sink,
        device_sink,
        planner,
        now + rmf_traffic::time::from_seconds(60000),
        drain_battery),

      rmf_task::requests::Delivery::make(
        8,
        5,
        11,
        motion_sink,
        device_sink,
        planner,
        now + rmf_traffic::time::from_seconds(60000),
        drain_battery),

      rmf_task::requests::Delivery::make(
        9,
        9,
        0,
        motion_sink,
        device_sink,
        planner,
        now + rmf_traffic::time::from_seconds(60000),
        drain_battery),

      rmf_task::requests::Delivery::make(
        10,
        1,
        3,
        motion_sink,
        device_sink,
        planner,
        now + rmf_traffic::time::from_seconds(60000),
        drain_battery),

      rmf_task::requests::Delivery::make(
        11,
        0,
        12,
        motion_sink,
        device_sink,
        planner,
        now + rmf_traffic::time::from_seconds(60000),
        drain_battery)
    };

    std::shared_ptr<rmf_task::agv::TaskPlanner::Configuration>  task_config =
      std::make_shared<rmf_task::agv::TaskPlanner::Configuration>(
        battery_system,
        motion_sink,
        device_sink,
        planner);
    rmf_task::agv::TaskPlanner task_planner(task_config);

    const auto greedy_assignments = task_planner.greedy_plan(
      now, initial_states, state_configs, requests);
    const double greedy_cost = task_planner.compute_cost(greedy_assignments);

    const auto optimal_assignments = task_planner.optimal_plan(
      now, initial_states, state_configs, requests, nullptr);
    const double optimal_cost = task_planner.compute_cost(optimal_assignments);
  
    //display_solution("Greedy", greedy_assignments, greedy_cost);
    display_solution("Optimal", optimal_assignments, optimal_cost);

    //REQUIRE(optimal_cost <= greedy_cost);
  }*/

  std::vector<std::tuple<int,int,int>> test1 {
    {2,3,0},
    {15,4,0},
    {7,12,0},
    {8,11,50000},
    {10,0,50000},
    {1,8,60000},
    {8,7,60000},
    {5,11,60000},
    {9,12,60000},
    {1,5,60000},
    {0,10,60000}};

  std::vector<int> test1_waypoints {13, 2, 13, 2};

  std::vector<std::tuple<int,int,int>> test2 {
    {6,3,0},
    {10,7,0},
    {2,12,0},
    {8,11,50000},
    {10,6,70000},
    {2,9,70000},
    {3,4,70000},
    {5,11,70000},
    {9,1,70000},
    {1,5,70000},
    {13,10,70000}};

  std::vector<int> test2_waypoints {13, 2, 9, 2};

  std::vector<std::tuple<int,int,int>> test3 {
    {2,3,0},
    {8,4,0},
    {4,1,0},
    {3,7,50000},
    {14,2,50000},
    {3,8,60000},
    {5,7,60000},
    {5,1,60000},
    {10,12,60000},
    {1,5,60000},
    {8,10,60000}};
  
  std::vector<int> test3_waypoints {13, 2, 13, 2};

  std::vector<std::tuple<int,int,int>> test4 {
    {9,12,0},
    {5,9,0},
    {2,4,0},
    {14,7,50000},
    {14,2,50000},
    {3,1,60000},
    {8,2,60000},
    {5,11,60000},
    {9,4,60000},
    {6,7,60000},
    {12,4,60000}};

  std::vector<int> test4_waypoints {13, 2, 13, 2};

  // Identical to earlier test above
  std::vector<std::tuple<int,int,int>> test5 {
    {0,3,0},
    {2,15,0},
    {7,9,0},
    {8,11,50000},
    {5,10,50000},
    {4,8,60000},
    {8,14,60000},
    {5,11,60000},
    {9,0,60000},
    {1,3,60000},
    {0,12,60000}};

  std::vector<int> test5_waypoints {13, 2, 13, 2};

  bool test_optimal = true; // set to true if you want to invoke the optimal solver

  // Hardcoded tests - 11 tasks, different start times
  std::vector<std::vector<std::tuple<int,int,int>>> tests {test1, test2, test3, test4, test5};
  std::vector<std::vector<int>> test_waypoints {test1_waypoints, test2_waypoints, test3_waypoints, test4_waypoints, test5_waypoints};
  run_tests(tests, test_waypoints, battery_system, planner, motion_sink, device_sink, drain_battery, test_optimal);

  // Randomly generated tests

  //7 tasks, start time 0
  std::pair<std::vector<std::vector<std::tuple<int,int,int>>>,
    std::vector<std::vector<int>>> auto_gen_testcases_1 =
    generate_testcases(7, {{7,0}}, 20);

  //7 tasks, start time 30000
  std::pair<std::vector<std::vector<std::tuple<int,int,int>>>,
    std::vector<std::vector<int>>> auto_gen_testcases_2 =
    generate_testcases(7, {{7,30000}}, 20);

  //11 tasks, different start times
  std::pair<std::vector<std::vector<std::tuple<int,int,int>>>,
    std::vector<std::vector<int>>> auto_gen_testcases_3 =
    generate_testcases(15, {{4,0},{3,50000},{4,70000}}, 20);

  //29 tasks, different start times
  std::pair<std::vector<std::vector<std::tuple<int,int,int>>>,
    std::vector<std::vector<int>>> auto_gen_testcases_4 =
    generate_testcases(15, {{4,0},{3,50000},{4,70000}, {3,90000}, {5, 130000},{2,170000}, {4, 190000}, {3, 220000}, {1, 250000}}, 2);

  //run_tests(auto_gen_testcases_3.first, auto_gen_testcases_3.second, battery_system,
  //  planner, motion_sink, device_sink, drain_battery, test_optimal);
}

// Randomly generates a testcase, where `max` is the number of waypoints on the map,
// and test_profile is a vector specifying the number of tasks at each time
// Returns a pair consisting of a vector of delivery tasks and a list of charger waypoints
std::pair<std::vector<std::tuple<int,int,int>>, std::vector<int>> generate_testcase(
  size_t max, std::vector<std::vector<size_t>> test_profile, std::mt19937& eng)
{
  std::uniform_int_distribution<> dist(1,max);

  std::vector<std::tuple<int,int,int>> tasks; // tuple of from, to locations and time
  for(size_t i = 0; i < test_profile.size(); ++i){
    for(size_t j = 0; j < test_profile[i][0]; ++j){
      int from = dist(eng);
      int to = dist(eng);
      while(to == from){
        to = dist(eng);
      }
      int time = test_profile[i][1];
      std::tuple<int,int,int> task{from, to, time};
      tasks.push_back(task);
    }
  }

  std::vector<int> waypoints {13, 2, 13, 2}; // hardcoded for now
  return make_pair(tasks, waypoints);
}

std::pair<std::vector<std::vector<std::tuple<int,int,int>>>,
  std::vector<std::vector<int>>> generate_testcases(
  size_t max, std::vector<std::vector<size_t>> num_tasks, size_t num_testcases){
  // create source of randomness, and initialize it with non-deterministic seed
  std::random_device r;
  std::seed_seq seed{r(), r(), r(), r(), r(), r(), r(), r()};
  std::mt19937 eng{1242};
  //std::mt19937 eng{seed};

  std::vector<std::vector<std::tuple<int,int,int>>> testcases;
  std::vector<std::vector<int>> waypoints;
  for(size_t i = 0; i < num_testcases; ++i){
    auto testcase = generate_testcase(max, num_tasks, eng);
    testcases.push_back(testcase.first);
    waypoints.push_back(testcase.second);
  }
  return make_pair(testcases, waypoints);
}

void run_tests(std::vector<std::vector<std::tuple<int,int,int>>> tests,
  std::vector<std::vector<int>> test_waypoints,
  const rmf_battery::agv::BatterySystem& battery_system,
  std::shared_ptr<rmf_traffic::agv::Planner> planner,
  std::shared_ptr<rmf_battery::agv::SimpleMotionPowerSink> motion_sink,
  std::shared_ptr<rmf_battery::agv::SimpleDevicePowerSink> device_sink,
  bool drain_battery,
  bool optimal)
{
  std::vector<std::pair<Assignments, double>> assignments;

  for(size_t i = 0; i < tests.size(); ++i){
    auto& waypoints = test_waypoints[i];
    auto& test = tests[i];
    std::pair<Assignments, double> assignment = compute_assignments(test, waypoints[0],
        waypoints[1], waypoints[2], waypoints[3], battery_system, planner,
        motion_sink, device_sink, drain_battery, optimal);
    assignments.push_back(std::move(assignment));
  }

  for(size_t i = 0; i < assignments.size(); ++i){
    std::cout << "i: " << i << std::endl;
    display_solution("Result ", assignments[i].first, assignments[i].second);
  }

  double avg = 0;
  for(size_t i = 0; i < assignments.size(); ++i){
    std::cout << assignments[i].second << " , ";
    avg += assignments[i].second;
  }
  std::cout << std::endl;
  std::cout << "Average cost: " << avg/(double)assignments.size() << std::endl;

}

std::pair<Assignments, double> compute_assignments(const std::vector<std::tuple<int,int,int>>& request_data,
  size_t initial_pt,
  size_t initial_pt2,
  size_t charging_pt,
  size_t charging_pt2,
  const rmf_battery::agv::BatterySystem& battery_system,
  std::shared_ptr<rmf_traffic::agv::Planner> planner,
  std::shared_ptr<rmf_battery::agv::SimpleMotionPowerSink> motion_sink,
  std::shared_ptr<rmf_battery::agv::SimpleDevicePowerSink> device_sink,
  bool drain_battery,
  bool optimal)
{
  const auto now = std::chrono::steady_clock::now();
  const double default_orientation = 0.0;

  rmf_traffic::agv::Plan::Start first_location{now, initial_pt, default_orientation};
  rmf_traffic::agv::Plan::Start second_location{now, initial_pt2, default_orientation};

  std::vector<rmf_task::agv::State> initial_states =
  {
    rmf_task::agv::State{first_location, charging_pt, 1.0},
    rmf_task::agv::State{second_location, charging_pt2, 1.0}
  };

  std::vector<rmf_task::agv::StateConfig> state_configs =
  {
    rmf_task::agv::StateConfig{0.2},
    rmf_task::agv::StateConfig{0.2}
  };

  std::vector<rmf_task::Request::SharedPtr> requests;
  for(size_t i = 0; i < request_data.size(); ++i){
    auto new_task = rmf_task::requests::Delivery::make(
      i,
      std::get<0>(request_data[i]),
      std::get<1>(request_data[i]),
      motion_sink,
      device_sink,
      planner,
      now + rmf_traffic::time::from_seconds(std::get<2>(request_data[i])),
      drain_battery);
    requests.push_back(std::move(new_task));
  }

  std::shared_ptr<rmf_task::agv::TaskPlanner::Configuration>  task_config =
    std::make_shared<rmf_task::agv::TaskPlanner::Configuration>(
      battery_system,
      motion_sink,
      device_sink,
      planner);
  rmf_task::agv::TaskPlanner task_planner(task_config);

  if(optimal)
  {
    const auto assignments = task_planner.optimal_plan(
      now, initial_states, state_configs, requests, nullptr);
    const double cost = task_planner.compute_cost(assignments);
    return(std::make_pair(assignments, cost));
  }
  else
  {
    const auto assignments = task_planner.greedy_plan(
      now, initial_states, state_configs, requests);
    const double cost = task_planner.compute_cost(assignments);
    return(std::make_pair(assignments, cost));
  }
}