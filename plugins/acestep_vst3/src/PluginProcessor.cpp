#include "PluginProcessor.h"

#include "PluginEditor.h"

namespace acestep::vst3
{
ACEStepVST3AudioProcessor::ACEStepVST3AudioProcessor()
    : juce::AudioProcessor(
          BusesProperties().withOutput("Output", juce::AudioChannelSet::stereo(), true))
{
}

ACEStepVST3AudioProcessor::~ACEStepVST3AudioProcessor() = default;

void ACEStepVST3AudioProcessor::prepareToPlay(double sampleRate, int samplesPerBlock)
{
    juce::ignoreUnused(sampleRate, samplesPerBlock);
}

void ACEStepVST3AudioProcessor::releaseResources() {}

bool ACEStepVST3AudioProcessor::isBusesLayoutSupported(const BusesLayout& layouts) const
{
    if (layouts.getMainOutputChannelSet() != juce::AudioChannelSet::stereo())
    {
        return false;
    }

    return layouts.getMainInputChannelSet().isDisabled();
}

void ACEStepVST3AudioProcessor::processBlock(juce::AudioBuffer<float>& buffer,
                                             juce::MidiBuffer& midiMessages)
{
    juce::ignoreUnused(midiMessages);
    buffer.clear();
}

juce::AudioProcessorEditor* ACEStepVST3AudioProcessor::createEditor()
{
    return new ACEStepVST3AudioProcessorEditor(*this);
}

bool ACEStepVST3AudioProcessor::hasEditor() const
{
    return true;
}

const juce::String ACEStepVST3AudioProcessor::getName() const
{
    return kPluginName;
}

bool ACEStepVST3AudioProcessor::acceptsMidi() const
{
    return true;
}

bool ACEStepVST3AudioProcessor::producesMidi() const
{
    return false;
}

bool ACEStepVST3AudioProcessor::isMidiEffect() const
{
    return false;
}

bool ACEStepVST3AudioProcessor::isSynth() const
{
    return true;
}

double ACEStepVST3AudioProcessor::getTailLengthSeconds() const
{
    return 0.0;
}

int ACEStepVST3AudioProcessor::getNumPrograms()
{
    return 1;
}

int ACEStepVST3AudioProcessor::getCurrentProgram()
{
    return 0;
}

void ACEStepVST3AudioProcessor::setCurrentProgram(int index)
{
    juce::ignoreUnused(index);
}

const juce::String ACEStepVST3AudioProcessor::getProgramName(int index)
{
    juce::ignoreUnused(index);
    return {};
}

void ACEStepVST3AudioProcessor::changeProgramName(int index, const juce::String& newName)
{
    juce::ignoreUnused(index, newName);
}

void ACEStepVST3AudioProcessor::getStateInformation(juce::MemoryBlock& destData)
{
    if (auto xml = createStateXml(state_))
    {
        copyXmlToBinary(*xml, destData);
    }
}

void ACEStepVST3AudioProcessor::setStateInformation(const void* data, int sizeInBytes)
{
    std::unique_ptr<juce::XmlElement> xml(getXmlFromBinary(data, sizeInBytes));
    if (xml != nullptr)
    {
        if (auto parsedState = parseStateXml(*xml))
        {
            state_ = *parsedState;
        }
    }
}

const PluginState& ACEStepVST3AudioProcessor::getState() const noexcept
{
    return state_;
}

PluginState& ACEStepVST3AudioProcessor::getMutableState() noexcept
{
    return state_;
}

void ACEStepVST3AudioProcessor::requestGeneration()
{
    if (state_.prompt.trim().isEmpty())
    {
        state_.jobStatus = JobStatus::failed;
        state_.progressText = {};
        state_.errorMessage = "Prompt is required before generation.";
        return;
    }

    clearGeneratedResults();
    state_.jobStatus = JobStatus::submitting;
    state_.progressText = "Submitting request...";
    state_.errorMessage = {};
}

void ACEStepVST3AudioProcessor::selectResultSlot(int index)
{
    state_.selectedResultSlot = juce::jlimit(0, kResultSlotCount - 1, index);
}

void ACEStepVST3AudioProcessor::pumpBackendWorkflow()
{
    std::optional<BackendTaskResult> completedTask;
    {
        const juce::ScopedLock lock(backendTaskLock_);
        if (completedBackendTask_.has_value())
        {
            completedTask = std::move(completedBackendTask_);
            completedBackendTask_.reset();
        }
    }

    if (completedTask.has_value())
    {
        applyCompletedTask(*completedTask);
    }

    if (backendTaskRunning_.load())
    {
        return;
    }

    const auto now = juce::Time::getMillisecondCounter();
    if (state_.jobStatus == JobStatus::submitting)
    {
        scheduleGenerationStart();
        return;
    }

    if (state_.jobStatus == JobStatus::queuedOrRunning && state_.currentTaskId.isNotEmpty()
        && now - lastPollRequestAtMs_ >= 1500)
    {
        scheduleGenerationPoll();
        return;
    }

    if (state_.jobStatus == JobStatus::idle || state_.jobStatus == JobStatus::failed
        || state_.jobStatus == JobStatus::succeeded)
    {
        if (lastHealthCheckedBaseUrl_ != state_.backendBaseUrl || now - lastHealthCheckAtMs_ >= 5000)
        {
            scheduleHealthCheck();
        }
    }
}

void ACEStepVST3AudioProcessor::scheduleHealthCheck()
{
    if (backendTaskRunning_.exchange(true))
    {
        return;
    }

    lastHealthCheckedBaseUrl_ = state_.backendBaseUrl;
    lastHealthCheckAtMs_ = juce::Time::getMillisecondCounter();
    const auto baseUrl = state_.backendBaseUrl;
    backendThreadPool_.addJob(std::function<juce::ThreadPoolJob::JobStatus()>([this, baseUrl]() {
        BackendTaskResult taskResult;
        taskResult.kind = BackendTaskKind::healthCheck;
        taskResult.health = backendClient_.checkHealth(baseUrl);
        const juce::ScopedLock lock(backendTaskLock_);
        completedBackendTask_ = std::move(taskResult);
        backendTaskRunning_.store(false);
        return juce::ThreadPoolJob::jobHasFinished;
    }));
}

void ACEStepVST3AudioProcessor::scheduleGenerationStart()
{
    if (backendTaskRunning_.exchange(true))
    {
        return;
    }

    const auto stateSnapshot = state_;
    backendThreadPool_.addJob(std::function<juce::ThreadPoolJob::JobStatus()>([this, stateSnapshot]() {
        BackendTaskResult taskResult;
        taskResult.kind = BackendTaskKind::submitGeneration;
        taskResult.generationStart = backendClient_.startGeneration(stateSnapshot);
        const juce::ScopedLock lock(backendTaskLock_);
        completedBackendTask_ = std::move(taskResult);
        backendTaskRunning_.store(false);
        return juce::ThreadPoolJob::jobHasFinished;
    }));
}

void ACEStepVST3AudioProcessor::scheduleGenerationPoll()
{
    if (backendTaskRunning_.exchange(true))
    {
        return;
    }

    const auto baseUrl = state_.backendBaseUrl;
    const auto taskId = state_.currentTaskId;
    lastPollRequestAtMs_ = juce::Time::getMillisecondCounter();
    backendThreadPool_.addJob(std::function<juce::ThreadPoolJob::JobStatus()>([this, baseUrl, taskId]() {
        BackendTaskResult taskResult;
        taskResult.kind = BackendTaskKind::pollGeneration;
        taskResult.generationPoll = backendClient_.pollGeneration(baseUrl, taskId);
        const juce::ScopedLock lock(backendTaskLock_);
        completedBackendTask_ = std::move(taskResult);
        backendTaskRunning_.store(false);
        return juce::ThreadPoolJob::jobHasFinished;
    }));
}

void ACEStepVST3AudioProcessor::applyCompletedTask(const BackendTaskResult& taskResult)
{
    switch (taskResult.kind)
    {
        case BackendTaskKind::healthCheck:
            state_.backendStatus = taskResult.health.status;
            if (taskResult.health.status == BackendStatus::ready
                && state_.jobStatus != JobStatus::failed)
            {
                state_.errorMessage = {};
            }
            else if (taskResult.health.status != BackendStatus::ready
                     && state_.jobStatus == JobStatus::idle)
            {
                state_.errorMessage = taskResult.health.errorMessage;
            }
            return;
        case BackendTaskKind::submitGeneration:
            if (!taskResult.generationStart.succeeded)
            {
                state_.jobStatus = JobStatus::failed;
                state_.progressText = {};
                state_.errorMessage = taskResult.generationStart.errorMessage;
                state_.currentTaskId = {};
                return;
            }

            state_.backendStatus = BackendStatus::ready;
            state_.jobStatus = JobStatus::queuedOrRunning;
            state_.currentTaskId = taskResult.generationStart.taskId;
            state_.progressText = "Task started: " + state_.currentTaskId;
            state_.errorMessage = {};
            return;
        case BackendTaskKind::pollGeneration:
            state_.jobStatus = taskResult.generationPoll.status;
            state_.progressText = taskResult.generationPoll.progressText;
            if (taskResult.generationPoll.status == JobStatus::failed)
            {
                state_.errorMessage = taskResult.generationPoll.errorMessage;
                state_.currentTaskId = {};
                return;
            }

            if (taskResult.generationPoll.status != JobStatus::succeeded)
            {
                return;
            }

            state_.currentTaskId = {};
            state_.errorMessage = {};
            for (int index = 0; index < kResultSlotCount; ++index)
            {
                const auto& slot = taskResult.generationPoll.resultSlots[static_cast<size_t>(index)];
                state_.resultSlots[static_cast<size_t>(index)] = slot.label;
                state_.resultFileUrls[static_cast<size_t>(index)] = slot.remoteFileUrl;
            }
            state_.selectedResultSlot = 0;
            if (state_.resultFileUrls[0].isEmpty())
            {
                state_.errorMessage = "Task finished but no audio file was returned.";
            }
            return;
        case BackendTaskKind::none:
            return;
    }
}

void ACEStepVST3AudioProcessor::clearGeneratedResults()
{
    state_.currentTaskId = {};
    state_.progressText = {};
    state_.selectedResultSlot = 0;
    for (int index = 0; index < kResultSlotCount; ++index)
    {
        state_.resultSlots[static_cast<size_t>(index)] = {};
        state_.resultFileUrls[static_cast<size_t>(index)] = {};
    }
}
}  // namespace acestep::vst3

juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new acestep::vst3::ACEStepVST3AudioProcessor();
}
