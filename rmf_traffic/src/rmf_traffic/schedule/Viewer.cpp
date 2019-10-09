/*
 * Copyright (C) 2019 Open Source Robotics Foundation
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

#include "ViewerInternal.hpp"

#include "../detail/internal_bidirectional_iterator.hpp"
#include "../DetectConflictInternal.hpp"

#include <rmf_traffic/schedule/Viewer.hpp>
#include <rmf_traffic/schedule/Database.hpp>

namespace rmf_traffic {
namespace schedule {

namespace internal {

//==============================================================================
void ViewRelevanceInspector::after(const std::size_t* _after)
{
  after_version = _after;
}

//==============================================================================
void ViewRelevanceInspector::reserve(std::size_t size)
{
  trajectories.reserve(size);
}

//==============================================================================
void ViewRelevanceInspector::inspect(
    const ConstEntryPtr& entry,
    const rmf_traffic::internal::Spacetime& spacetime_region)
{
  if(entry->succeeded_by)
    return;

  if(after_version && entry->version < *after_version)
    return;

  if(rmf_traffic::internal::detect_conflicts(
       entry->trajectory, spacetime_region, nullptr))
    trajectories.push_back(&entry->trajectory);
}

//==============================================================================
void ViewRelevanceInspector::inspect(
    const ConstEntryPtr& entry,
    const Time* lower_time_bound,
    const Time* upper_time_bound)
{
  if(entry->succeeded_by)
    return;

  if(after_version && entry->version < *after_version)
    return;

  const Trajectory& trajectory = entry->trajectory;
  assert(trajectory.start_time() != nullptr);

  if(lower_time_bound && *trajectory.finish_time() < *lower_time_bound)
    return;

  if(upper_time_bound && *upper_time_bound < *trajectory.start_time())
    return;

  trajectories.push_back(&entry->trajectory);
}

} // namespace internal

//==============================================================================
class Viewer::View::Implementation
{
public:

  std::vector<const Trajectory*> entries;

  static View make_view(std::vector<const Trajectory*> entries)
  {
    View view;
    view._pimpl = rmf_utils::make_impl<Implementation>(
          Implementation{std::move(entries)});
    return view;
  }
};

//==============================================================================
class Viewer::View::IterImpl
{
public:

  internal::DeepIterator iter;

};

//==============================================================================
auto Viewer::View::begin() const -> const_iterator
{
  return const_iterator{IterImpl{
      internal::DeepIterator{_pimpl->entries.begin()}}};
}

//==============================================================================
auto Viewer::View::end() const -> const_iterator
{
  return const_iterator{IterImpl{
      internal::DeepIterator{_pimpl->entries.end()}}};
}

//==============================================================================
std::size_t Viewer::View::size() const
{
  return _pimpl->entries.size();
}

//==============================================================================
Viewer::View Viewer::query(const Query& parameters) const
{
  return View::Implementation::make_view(
        std::move(_pimpl->inspect<internal::ViewRelevanceInspector>(
                    parameters).trajectories));
}

} // namespace schedule

namespace detail {

template class bidirectional_iterator<
    const Trajectory,
    schedule::Viewer::View::IterImpl,
    schedule::Viewer::View
>;

} // namespace detail
} // namespace rmf_traffic
