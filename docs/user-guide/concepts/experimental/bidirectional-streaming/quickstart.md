# Quickstart

This quickstart guide shows you how to create your first bidirectional streaming agent for real-time audio and text conversations. You'll learn how to set up audio I/O, handle streaming events, use tools during conversations, and work with different model providers.

After completing this guide, you can build voice assistants, interactive chatbots, multi-modal applications, and integrate bidirectional streaming with web servers or custom I/O channels.

## Prerequisites

Before starting, ensure you have:

- Python 3.12+ installed
- Audio hardware (microphone and speakers) for voice conversations
- Model provider credentials configured (AWS, OpenAI, or Google)

## Install the SDK

Bidirectional streaming is included in the Strands Agents SDK as an experimental feature. Install the SDK with bidirectional streaming support:

### For All Providers

To install with support for all bidirectional streaming providers:

```bash
pip install "strands-agents[bidi-all]"
```

This includes:

- Core bidirectional streaming functionality
- PyAudio for audio I/O
- Support for Nova Sonic (AWS Bedrock)
- Support for OpenAI Realtime API
- Support for Gemini Live API

### For Specific Providers

You can also install support for specific providers only:

**Nova Sonic (AWS Bedrock) only:**
```bash
pip install "strands-agents[bidi]"
```

**OpenAI Realtime API:**
```bash
pip install "strands-agents[bidi,bidi-openai]"
```

**Gemini Live API:**
```bash
pip install "strands-agents[bidi,bidi-gemini]"
```

### Platform-Specific Audio Setup

On macOS, you may need to install PortAudio first:

```bash
brew install portaudio
pip install "strands-agents[bidi-all]"
```

On Linux (Ubuntu/Debian):

```bash
sudo apt-get install portaudio19-dev python3-pyaudio
pip install "strands-agents[bidi-all]"
```

On Windows, PyAudio typically installs without additional dependencies.

## Configuring Credentials

Bidirectional streaming supports multiple model providers. Choose one based on your needs:

### Amazon Bedrock (Nova Sonic)

Nova Sonic is Amazon's bidirectional streaming model. Configure AWS credentials:

```bash
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
export AWS_DEFAULT_REGION=us-east-1
```

Enable Nova Sonic model access in the [Amazon Bedrock console](https://docs.aws.amazon.com/bedrock/latest/userguide/model-access-modify.html).

### OpenAI (Realtime API)

For OpenAI's Realtime API, set your API key:

```bash
export OPENAI_API_KEY=your_api_key
```

### Google (Gemini Live)

For Gemini Live API, set your API key:

```bash
export GOOGLE_API_KEY=your_api_key
```

## Your First Voice Conversation

Now let's create a simple voice-enabled agent that can have real-time conversations:

```python
import asyncio
from strands.experimental.bidi import BidiAgent, BidiAudioIO
from strands.experimental.bidi.models import BidiNovaSonicModel

# Create a bidirectional streaming model
model = BidiNovaSonicModel()

# Create the agent
agent = BidiAgent(
    model=model,
    system_prompt="You are a helpful voice assistant. Keep responses concise and natural."
)

# Setup audio I/O for microphone and speakers
audio_io = BidiAudioIO()

# Run the conversation
async def main():
    await agent.run(
        inputs=[audio_io.input()],
        outputs=[audio_io.output()]
    )

asyncio.run(main())
```

And that's it! We now have a voice-enabled agent that can:

- Listen to your voice through the microphone
- Process speech in real-time
- Respond with natural voice output
- Handle interruptions when you start speaking

!!! note "Stopping the Conversation"
    The `run()` method runs indefinitely. See [Controlling Conversation Lifecycle](#controlling-conversation-lifecycle) for proper ways to stop conversations.

## Adding Text I/O

Combine audio with text input/output for debugging or multi-modal interactions:

```python
import asyncio
from strands.experimental.bidi import BidiAgent, BidiAudioIO, BidiTextIO
from strands.experimental.bidi.models import BidiNovaSonicModel

model = BidiNovaSonicModel()
agent = BidiAgent(
    model=model,
    system_prompt="You are a helpful assistant."
)

# Setup both audio and text I/O
audio_io = BidiAudioIO()
text_io = BidiTextIO()

async def main():
    await agent.run(
        inputs=[audio_io.input()],
        outputs=[audio_io.output(), text_io.output()]  # Both audio and text
    )

asyncio.run(main())
```

Now you'll see transcripts printed to the console while audio plays through your speakers.

## Controlling Conversation Lifecycle

Both `run()` and `receive()` run indefinitely by default. Here are the proper ways to control when conversations start and stop:

### Using run() with Keyboard Interrupt

The simplest approach for testing is to use `Ctrl+C`:

```python
import asyncio
from strands.experimental.bidi import BidiAgent, BidiAudioIO
from strands.experimental.bidi.models import BidiNovaSonicModel

async def main():
    model = BidiNovaSonicModel()
    agent = BidiAgent(model=model)
    audio_io = BidiAudioIO()
    
    try:
        # Runs indefinitely until interrupted
        await agent.run(
            inputs=[audio_io.input()],
            outputs=[audio_io.output()]
        )
    except KeyboardInterrupt:
        print("\nConversation interrupted by user")
    finally:
        # stop() should only be called after run() exits
        await agent.stop()

asyncio.run(main())
```

### Using receive() with Exit Conditions

When manually processing events, implement explicit exit conditions:

```python
import asyncio
from strands.experimental.bidi import BidiAgent
from strands.experimental.bidi.models import BidiNovaSonicModel
from strands.experimental.bidi.types.events import (
    BidiResponseCompleteEvent,
    BidiConnectionCloseEvent
)

async def main():
    model = BidiNovaSonicModel()
    agent = BidiAgent(model=model)
    
    await agent.start()
    
    # Send initial message
    await agent.send("Tell me three facts about Python")
    
    # Process events with exit condition
    async for event in agent.receive():
        if isinstance(event, BidiResponseCompleteEvent):
            print("Response complete, exiting...")
            break  # Exit the receive loop
        
        elif isinstance(event, BidiConnectionCloseEvent):
            print("Connection closed")
            break
    
    # stop() should only be called after exiting receive loop
    await agent.stop()

asyncio.run(main())
```

### Using a Message Counter

For multi-turn conversations with a limit:

```python
import asyncio
from strands.experimental.bidi import BidiAgent
from strands.experimental.bidi.models import BidiNovaSonicModel
from strands.experimental.bidi.types.events import BidiResponseCompleteEvent

async def main():
    model = BidiNovaSonicModel()
    agent = BidiAgent(model=model)
    
    await agent.start()
    
    max_turns = 3
    turn_count = 0
    
    questions = [
        "What is Python?",
        "What is asyncio?",
        "What are coroutines?"
    ]
    
    # Send first question
    await agent.send(questions[turn_count])
    
    async for event in agent.receive():
        if isinstance(event, BidiResponseCompleteEvent):
            turn_count += 1
            
            if turn_count >= max_turns:
                print(f"Completed {max_turns} turns, exiting...")
                break
            
            # Send next question
            await agent.send(questions[turn_count])
    
    # stop() should only be called after exiting receive loop
    await agent.stop()

asyncio.run(main())
```

### Using Context Manager (Automatic Cleanup)

The context manager automatically calls `stop()` when exiting:

```python
import asyncio
from strands.experimental.bidi import BidiAgent
from strands.experimental.bidi.models import BidiNovaSonicModel
from strands.experimental.bidi.types.events import BidiResponseCompleteEvent

async def main():
    model = BidiNovaSonicModel()
    
    # Context manager handles start/stop automatically
    async with BidiAgent(model=model) as agent:
        await agent.send("Tell me a joke")
        
        async for event in agent.receive():
            if isinstance(event, BidiResponseCompleteEvent):
                break  # Exit receive loop
    
    # stop() called automatically when exiting context

asyncio.run(main())
```

!!! warning "Important: Call stop() After Exiting Loops"
    Always call `agent.stop()` **after** exiting the `run()` or `receive()` loop, never during. Calling `stop()` while still receiving events can cause errors. The context manager pattern handles this automatically.

## Adding Tools to Your Agent

Just like standard Strands agents, bidirectional agents can use tools during conversations:

```python
import asyncio
from strands import tool
from strands.experimental.bidi import BidiAgent, BidiAudioIO
from strands.experimental.bidi.models import BidiNovaSonicModel
from strands_tools import calculator, current_time

# Define a custom tool
@tool
def get_weather(location: str) -> str:
    """
    Get the current weather for a location.
    
    Args:
        location: City name or location
    
    Returns:
        Weather information
    """
    # In a real application, call a weather API
    return f"The weather in {location} is sunny and 72°F"

# Create agent with tools
model = BidiNovaSonicModel()
agent = BidiAgent(
    model=model,
    tools=[calculator, current_time, get_weather],
    system_prompt="You are a helpful assistant with access to tools."
)

audio_io = BidiAudioIO()

async def main():
    await agent.run(
        inputs=[audio_io.input()],
        outputs=[audio_io.output()]
    )

asyncio.run(main())
```

You can now ask questions like:

- "What time is it?"
- "Calculate 25 times 48"
- "What's the weather in San Francisco?"

The agent automatically determines when to use tools and executes them concurrently without blocking the conversation.

## Understanding Bidirectional Events

Bidirectional streaming produces various events during a conversation. For more control, you can process these events manually:

```python
import asyncio
from strands.experimental.bidi import BidiAgent
from strands.experimental.bidi.models import BidiNovaSonicModel
from strands.experimental.bidi.types.events import (
    BidiConnectionStartEvent,
    BidiAudioStreamEvent,
    BidiTranscriptStreamEvent,
    BidiInterruptionEvent,
    BidiResponseCompleteEvent,
    BidiConnectionCloseEvent
)

model = BidiNovaSonicModel()
agent = BidiAgent(model=model)

async def main():
    # Start the agent
    await agent.start()
    
    # Send a text message
    await agent.send("Tell me a short joke")
    
    # Process events as they arrive
    async for event in agent.receive():
        if isinstance(event, BidiConnectionStartEvent):
            print(f"Connected to {event.model}")
        
        elif isinstance(event, BidiAudioStreamEvent):
            # Audio chunk received (base64 encoded)
            print(f"Audio chunk: {len(event.audio)} bytes")
        
        elif isinstance(event, BidiTranscriptStreamEvent):
            # Transcript of speech (user or assistant)
            print(f"{event.role}: {event.text}")
        
        elif isinstance(event, BidiInterruptionEvent):
            print(f"Interrupted: {event.reason}")
        
        elif isinstance(event, BidiResponseCompleteEvent):
            print(f"Response complete: {event.stop_reason}")
            break
        
        elif isinstance(event, BidiConnectionCloseEvent):
            print(f"Connection closed: {event.reason}")
            break
    
    # Cleanup
    await agent.stop()

asyncio.run(main())
```

This manual approach gives you full control over event processing. It's particularly useful for:

- Custom UI updates
- Logging and analytics
- Integration with web frameworks
- Building custom I/O channels

## Model Providers

### Amazon Nova Sonic (Default)

Nova Sonic is optimized for low-latency voice conversations:

```python
from strands.experimental.bidi.models import BidiNovaSonicModel

model = BidiNovaSonicModel(
    model_id="amazon.nova-sonic-v1:0",
    provider_config={
        "audio": {
            "input_rate": 16000,
            "output_rate": 16000,
            "voice": "matthew"  # or "ruth"
        }
    }
)
```

**Features:**

- 8-minute connection timeout (auto-reconnects)
- 16kHz audio input/output
- Built-in voice activity detection
- AWS Bedrock integration

### OpenAI Realtime API

OpenAI's Realtime API provides high-quality voice interactions:

```python
from strands.experimental.bidi.models import BidiOpenAIRealtimeModel

model = BidiOpenAIRealtimeModel(
    model_id="gpt-realtime",
    provider_config={
        "audio": {
            "voice": "alloy"  # alloy, echo, fable, onyx, nova, shimmer
        }
    },
    client_config={
        "api_key": "your_api_key"  # or set OPENAI_API_KEY env var
    }
)
```

**Features:**

- 60-minute connection timeout
- 24kHz audio input/output
- Multiple voice options
- Configurable VAD settings

### Google Gemini Live

Gemini Live offers multimodal streaming with session resumption:

```python
from strands.experimental.bidi.models import BidiGeminiLiveModel

model = BidiGeminiLiveModel(
    model_id="gemini-2.5-flash-native-audio-preview-09-2025",
    provider_config={
        "audio": {
            "voice": "Puck"  # Puck, Charon, Kore, Fenrir, Aoede
        }
    },
    client_config={
        "api_key": "your_api_key"  # or set GOOGLE_API_KEY env var
    }
)
```

**Features:**

- Session resumption after timeouts
- 16kHz input, 24kHz output
- Multiple voice options
- Native multimodal support

## Configuring Audio Settings

Customize audio configuration for both the model and I/O:

```python
from strands.experimental.bidi import BidiAgent, BidiAudioIO
from strands.experimental.bidi.models import BidiGeminiLiveModel

# Configure model audio settings
model = BidiGeminiLiveModel(
    provider_config={
        "audio": {
            "input_rate": 48000,   # Higher quality input
            "output_rate": 24000,  # Standard output
            "voice": "Puck"
        }
    }
)

# Configure I/O buffer settings
audio_io = BidiAudioIO(
    input_buffer_size=10,           # Max input queue size
    output_buffer_size=20,          # Max output queue size
    input_frames_per_buffer=512,   # Input chunk size
    output_frames_per_buffer=512   # Output chunk size
)

agent = BidiAgent(model=model)

async def main():
    await agent.run(
        inputs=[audio_io.input()],
        outputs=[audio_io.output()]
    )

asyncio.run(main())
```

The I/O automatically configures hardware to match the model's audio requirements.

## Handling Interruptions

Bidirectional agents automatically handle interruptions when users start speaking:

```python
import asyncio
from strands.experimental.bidi import BidiAgent, BidiAudioIO
from strands.experimental.bidi.models import BidiNovaSonicModel
from strands.experimental.bidi.types.events import BidiInterruptionEvent

model = BidiNovaSonicModel()
agent = BidiAgent(model=model)
audio_io = BidiAudioIO()

async def main():
    await agent.start()
    
    # Start receiving events
    async for event in agent.receive():
        if isinstance(event, BidiInterruptionEvent):
            print(f"User interrupted: {event.reason}")
            # Audio output automatically cleared
            # Model stops generating
            # Ready for new input

asyncio.run(main())
```

Interruptions are detected via voice activity detection (VAD) and handled automatically:

1. User starts speaking
2. Model stops generating
3. Audio output buffer cleared
4. Model ready for new input

## Manual Start and Stop

If you need more control over the agent lifecycle, you can manually call `start()` and `stop()`:

```python
import asyncio
from strands.experimental.bidi import BidiAgent
from strands.experimental.bidi.models import BidiNovaSonicModel
from strands.experimental.bidi.types.events import BidiResponseCompleteEvent

async def main():
    model = BidiNovaSonicModel()
    agent = BidiAgent(model=model)
    
    # Manually start the agent
    await agent.start()
    
    try:
        await agent.send("What is Python?")
        
        async for event in agent.receive():
            if isinstance(event, BidiResponseCompleteEvent):
                break
    finally:
        # Always stop after exiting receive loop
        await agent.stop()

asyncio.run(main())
```

See [Controlling Conversation Lifecycle](#controlling-conversation-lifecycle) for more patterns and best practices.

## Passing Context to Tools

Pass custom context to tools during execution:

```python
import asyncio
from strands import tool
from strands.experimental.bidi import BidiAgent, BidiAudioIO
from strands.experimental.bidi.models import BidiNovaSonicModel

@tool
def get_user_info(invocation_state: dict) -> str:
    """Get information about the current user."""
    user_id = invocation_state["user_id"]
    user_name = invocation_state["user_name"]
    return f"User {user_name} (ID: {user_id})"

model = BidiNovaSonicModel()
agent = BidiAgent(
    model=model,
    tools=[get_user_info]
)

audio_io = BidiAudioIO()

async def main():
    # Pass context when starting
    await agent.start(invocation_state={
        "user_id": "user_123",
        "user_name": "Alice",
        "session_id": "session_456"
    })
    
    await agent.run(
        inputs=[audio_io.input()],
        outputs=[audio_io.output()]
    )

asyncio.run(main())
```

The `invocation_state` is available to all tools during execution.

## Graceful Shutdown

Use the experimental `stop_conversation` tool to allow users to end conversations naturally:

```python
import asyncio
from strands.experimental.bidi import BidiAgent, BidiAudioIO
from strands.experimental.bidi.models import BidiNovaSonicModel
from strands.experimental.bidi.tools import stop_conversation

model = BidiNovaSonicModel()
agent = BidiAgent(
    model=model,
    tools=[stop_conversation],
    system_prompt="You are a helpful assistant. When the user says 'stop conversation', use the stop_conversation tool."
)

audio_io = BidiAudioIO()

async def main():
    await agent.run(
        inputs=[audio_io.input()],
        outputs=[audio_io.output()]
    )
    # Conversation ends when user says "stop conversation"

asyncio.run(main())
```

The agent will gracefully close the connection when the user explicitly requests it.

## Debug Logs

To enable debug logs in your agent, configure the `strands` logger:

```python
import asyncio
import logging
from strands.experimental.bidi import BidiAgent, BidiAudioIO
from strands.experimental.bidi.models import BidiNovaSonicModel

# Enable debug logs
logging.getLogger("strands").setLevel(logging.DEBUG)
logging.basicConfig(
    format="%(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler()]
)

model = BidiNovaSonicModel()
agent = BidiAgent(model=model)
audio_io = BidiAudioIO()

async def main():
    await agent.run(
        inputs=[audio_io.input()],
        outputs=[audio_io.output()]
    )

asyncio.run(main())
```

Debug logs show:

- Connection lifecycle events
- Audio buffer operations
- Tool execution details
- Event processing flow

## Common Issues

### No Audio Output

If you don't hear audio:

```python
# List available audio devices
import pyaudio
p = pyaudio.PyAudio()
for i in range(p.get_device_count()):
    info = p.get_device_info_by_index(i)
    print(f"{i}: {info['name']}")

# Specify output device explicitly
audio_io = BidiAudioIO(output_device_index=2)
```

### Microphone Not Working

If the agent doesn't respond to speech:

```python
# Specify input device explicitly
audio_io = BidiAudioIO(input_device_index=1)

# Check system permissions (macOS)
# System Preferences → Security & Privacy → Microphone
```

### Connection Timeouts

If you experience frequent disconnections:

```python
# Use OpenAI for longer timeout (60 min vs Nova's 8 min)
from strands.experimental.bidi.models import BidiOpenAIRealtimeModel
model = BidiOpenAIRealtimeModel()

# Or handle restarts gracefully
async for event in agent.receive():
    if isinstance(event, BidiConnectionRestartEvent):
        print("Reconnecting...")
        continue
```

## Next Steps

Ready to learn more? Check out these resources:

- [Agent](agent.md) - Deep dive into BidiAgent configuration and lifecycle
- [Events](events.md) - Complete guide to bidirectional streaming events
- [I/O Channels](io/io.md) - Understanding and customizing input/output channels
- **Model Providers:**
  - [Nova Sonic](models/nova_sonic.md) - Amazon Bedrock's bidirectional streaming model
  - [OpenAI Realtime](models/openai_realtime.md) - OpenAI's Realtime API
  - [Gemini Live](models/gemini_live.md) - Google's Gemini Live API
- [API Reference](../../../../api-reference/experimental.md) - Complete API documentation

