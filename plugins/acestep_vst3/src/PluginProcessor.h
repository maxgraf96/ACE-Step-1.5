#pragma once

#include <atomic>
#include <optional>

#include <JuceHeader.h>

#include "PluginBackendClient.h"
#include "PluginConfig.h"
#include "PluginState.h"

namespace acestep::vst3
{
class ACEStepVST3AudioProcessor final : public juce::AudioProcessor
{
public:
    ACEStepVST3AudioProcessor();
    ~ACEStepVST3AudioProcessor() override;

    void prepareToPlay(double sampleRate, int samplesPerBlock) override;
    void releaseResources() override;
    bool isBusesLayoutSupported(const BusesLayout& layouts) const override;
    void processBlock(juce::AudioBuffer<float>& buffer, juce::MidiBuffer& midiMessages) override;

    juce::AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override;

    const juce::String getName() const override;
    bool acceptsMidi() const override;
    bool producesMidi() const override;
    bool isMidiEffect() const override;
    bool isSynth() const;
    double getTailLengthSeconds() const override;

    int getNumPrograms() override;
    int getCurrentProgram() override;
    void setCurrentProgram(int index) override;
    const juce::String getProgramName(int index) override;
    void changeProgramName(int index, const juce::String& newName) override;
    void getStateInformation(juce::MemoryBlock& destData) override;
    void setStateInformation(const void* data, int sizeInBytes) override;

    const PluginState& getState() const noexcept;
    PluginState& getMutableState() noexcept;
    void requestGeneration();
    void selectResultSlot(int index);
    void pumpBackendWorkflow();

private:
    enum class BackendTaskKind
    {
        none,
        healthCheck,
        submitGeneration,
        pollGeneration,
    };

    struct BackendTaskResult final
    {
        BackendTaskKind kind = BackendTaskKind::none;
        PluginHealthCheckResult health;
        PluginGenerationStartResult generationStart;
        PluginGenerationPollResult generationPoll;
    };

    void scheduleHealthCheck();
    void scheduleGenerationStart();
    void scheduleGenerationPoll();
    void applyCompletedTask(const BackendTaskResult& taskResult);
    void clearGeneratedResults();

    PluginState state_;
    PluginBackendClient backendClient_;
    juce::ThreadPool backendThreadPool_ {1};
    juce::CriticalSection backendTaskLock_;
    std::optional<BackendTaskResult> completedBackendTask_;
    std::atomic<bool> backendTaskRunning_ {false};
    juce::uint32 lastHealthCheckAtMs_ = 0;
    juce::uint32 lastPollRequestAtMs_ = 0;
    juce::String lastHealthCheckedBaseUrl_;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(ACEStepVST3AudioProcessor)
};
}  // namespace acestep::vst3
