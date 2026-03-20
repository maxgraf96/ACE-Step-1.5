#pragma once

#include <array>

#include <JuceHeader.h>

#include "PluginEnums.h"
#include "PluginState.h"

namespace acestep::vst3
{
struct PluginBackendResultSlot final
{
    juce::String label;
    juce::String remoteFileUrl;
};

struct PluginHealthCheckResult final
{
    BackendStatus status = BackendStatus::offline;
    juce::String errorMessage;
};

struct PluginGenerationStartResult final
{
    bool succeeded = false;
    juce::String taskId;
    juce::String errorMessage;
};

struct PluginGenerationPollResult final
{
    JobStatus status = JobStatus::failed;
    juce::String progressText;
    juce::String errorMessage;
    std::array<PluginBackendResultSlot, static_cast<size_t>(kResultSlotCount)> resultSlots;
};

class PluginBackendClient final
{
public:
    [[nodiscard]] PluginHealthCheckResult checkHealth(const juce::String& baseUrl) const;
    [[nodiscard]] PluginGenerationStartResult startGeneration(const PluginState& state) const;
    [[nodiscard]] PluginGenerationPollResult pollGeneration(const juce::String& baseUrl,
                                                            const juce::String& taskId) const;
};
}  // namespace acestep::vst3
