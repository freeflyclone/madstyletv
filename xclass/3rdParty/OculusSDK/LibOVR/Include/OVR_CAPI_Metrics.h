/************************************************************************************

Filename    :   OVR_CAPI_Metrics.h
Content     :   Public API metrics structures
Created     :   June 12, 2017
Copyright   :   Copyright 2017 Oculus VR, LLC All Rights reserved.
Author      :   Keivaun Waugh

Licensed under the Oculus VR Rift SDK License Version 3.3 (the "License");
you may not use the Oculus VR Rift SDK except in compliance with the License,
which is provided at the time of installation or download, or which
otherwise accompanies this software in either electronic or hard copy form.

You may obtain a copy of the License at

http://www.oculusvr.com/licenses/LICENSE-3.3

Unless required by applicable law or agreed to in writing, the Oculus VR SDK
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

************************************************************************************/

#pragma once

#include <atomic>
#include <string>
#include <array>
#include "../Src/OVR_CAPI_Prototypes.h"

// -----------------------------------------------------------------------------------
// Structures to keep track of API call counts. Eventually will include call latency.

#define GENERATE_STAT_ENUM(ENUM) stat_##ENUM
#define GENERATE_STAT_STRING(STRING) #STRING

#define METRICS(RETURN, NAME, VERSION, PARAMS) GENERATE_STAT(NAME)
// ignore old functions. they may share the same name which will cause enumre
// definition problems
#define NO_METRICS(RETURN, NAME, VERSION, PARAMS)
#define API_TABLE                           \
  OVR_LIST_PUBLIC_APIS(METRICS, NO_METRICS) \
  OVR_LIST_PRIVATE_APIS(METRICS, NO_METRICS)

#define GENERATE_STAT(ENUM) GENERATE_STAT_ENUM(ENUM),
enum PublicAPI { API_TABLE API_Count };
#undef GENERATE_STAT

#define GENERATE_STAT(STRING) GENERATE_STAT_STRING(STRING),
static std::string PublicAPIName[API_Count] = {API_TABLE};
#undef GENERATE_STAT

#undef METRICS
#undef NO_METRICS
#undef API_TABLE

// Structure to keep track of metrics for a single API
// Add to this in the future to keep track of more individual API metrics
struct IndivAPIStats {
  // array[0] is inactive metrics. array[1] is active metric
  // This allows the arrays to be indexed by the inFocus boolean
  std::array<std::atomic_uint64_t, 2> CallCount = {0};
  std::array<std::atomic_uint64_t, 2> MaxLatencyNs = {0};
  std::array<std::atomic_uint64_t, 2> TotalLatencyNs = {0};
  void reset() {
    CallCount[0] = {0};
    CallCount[1] = {0};
    MaxLatencyNs[0] = {0};
    MaxLatencyNs[1] = {0};
    TotalLatencyNs[0] = {0};
    TotalLatencyNs[1] = {0};
  }
};

/// Returns the number of API function calls made for each public API
///
/// \param[in] callCountsFocus Filled with API call counts while app is in focus
/// \param[in] callCountsNoFocus Filled with API call count while app is out of focus
/// \param[in] maxLatencyNsFocus Filled with max latency while app is in focus
/// \param[in] maxLatencyNsNoFocus Filled with max latency while app is out of focus
/// \param[in] avgLatencyNsFocus Filled with avg latency while app is in focus
/// \param[in] avgLatencyNsNoFocus Filled with avg latency while app is out of focus
///
void getAPIMetrics(
    uint64_t* callCountsFocus,
    uint64_t* callCountsNoFocus,
    uint64_t* maxLatencyNsFocus,
    uint64_t* maxLatencyNsNoFocus,
    uint64_t* avgLatencyNsFocus,
    uint64_t* avgLatencyNsNoFocus);

// Keep track of whether the client app is in focus or not
void updateAPIFocusStatus(bool inFocus);
