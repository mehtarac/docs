# BidiAgent [Experimental]

!!! warning "Experimental Feature"
    This feature is experimental and may change in future versions. Use with caution in production environments.

The `BidiAgent` is a specialized agent designed for real-time bidirectional streaming conversations. Unlike the standard `Agent` that follows a request-response pattern, `BidiAgent` maintains persistent connections that enable continuous audio and text streaming, real-time interruptions, and concurrent tool execution.

```mermaid
flowchart TB
    subgraph User
        A[Microphone] --> B[Audio Input]
        C[Text Input] --> D[Input Events]
        B --> D
    end
    
    subgraph BidiAgent
        D --> E[Agent Loop]
        E --> F[Model Connection]
        F --> G[Tool Execution]
        G --> F
        F --> H[Output Events]
    end
    
    subgraph Output
        H --> I[Audio Output]
        H --> J[Text Output]
        I --> K[Speakers]
        J --> L[Console/UI]
    end
```


## Agent vs BidiAgent

While both `Agent` and `BidiAgent` share the same core purpose of enabling AI-powered interactions, they differ significantly in their architecture and use cases.

### Standard Agent (Request-Response)

The standard `Agent` follows a traditional request-response pattern:

```python
from strands import Agent
from strands_tools import calculator

agent = Agent(tools=[calculator])

# Single request-response cycle
result = agent("Calculate 25 * 48")
print(result.message)  # "The result is 1200"
```

**Characteristics:**

- **Synchronous interaction**: One request, one response
- **Discrete cycles**: Each invocation is independent
- **Message-based**: Operates on complete messages
- **Tool execution**: Sequential, blocking the response

### BidiAgent (Bidirectional Streaming)

`BidiAgent` maintains a persistent, bidirectional connection:

```python
import asyncio
from strands.experimental.bidi import BidiAgent, BidiAudioIO
from strands.experimental.bidi.models import BidiNovaSonicModel

model = BidiNovaSonicModel()
agent = BidiAgent(model=model, tools=[calculator])
audio_io = BidiAudioIO()

async def main():
    # Persistent connection with continuous streaming
    await agent.run(
        inputs=[audio_io.input()],
        outputs=[audio_io.output()]
    )

asyncio.run(main())
```

**Characteristics:**

- **Asynchronous streaming**: Continuous input/output
- **Persistent connection**: Single connection for multiple turns
- **Event-based**: Operates on streaming events
- **Tool execution**: Concurrent, non-blocking

### When to Use Each

**Use `Agent` when:**

- Building chatbots or CLI applications
- Processing discrete requests
- Implementing API endpoints
- Working with text-only interactions
- Simplicity is preferred

**Use `BidiAgent` when:**

- Building voice assistants
- Requiring real-time audio streaming
- Needing natural conversation interruptions
- Implementing live transcription
- Building interactive, multi-modal applications


## The Bidirectional Agent Loop

The bidirectional agent loop is fundamentally different from the standard agent loop. Instead of processing discrete messages, it continuously streams events in both directions while managing connection state and concurrent operations.

### Architecture Overview

```mermaid
flowchart TB
    A[Agent Start] --> B[Model Connection]
    B --> C[Agent Loop]
    C --> D[Model Task]
    C --> E[Event Queue]
    D --> E
    E --> F[receive]
    D --> G[Tool Detection]
    G --> H[Tool Tasks]
    H --> E
    F --> I[User Code]
    I --> J[send]
    J --> K[Model]
    K --> D
```

### Event Flow

#### Startup Sequence

1. **Agent Initialization**
   ```python
   agent = BidiAgent(model=model, tools=[calculator])
   ```
   - Creates tool registry
   - Initializes agent state
   - Sets up hook registry

2. **Connection Start**
   ```python
   await agent.start()
   ```
   - Calls `model.start(system_prompt, tools, messages)`
   - Establishes WebSocket/SDK connection
   - Sends conversation history if provided
   - Spawns background task for model communication
   - Enables sending capability

3. **Event Processing**
   ```python
   async for event in agent.receive():
       # Process events
   ```
   - Dequeues events from internal queue
   - Yields to user code
   - Continues until stopped

#### Tool Execution

Tools execute concurrently without blocking the conversation. When a tool is invoked:

1. The tool executor streams events as the tool runs
2. Tool events are queued to the event loop
3. Tool use and result messages are added atomically to conversation history
4. Results are automatically sent back to the model

The special `stop_conversation` tool triggers agent shutdown instead of sending results back to the model.

### Connection Lifecycle

#### Normal Operation

```
User → send() → Model → receive() → Model Task → Event Queue → receive() → User
                  ↓
              Tool Use
                  ↓
            Tool Task → Event Queue → receive() → User
                  ↓
            Tool Result → Model
```

## Interruptions

One of the most powerful features of `BidiAgent` is its ability to handle real-time interruptions. When a user starts speaking while the model is generating a response, the agent automatically detects this and stops the current response, allowing for natural, human-like conversations.

### How Interruptions Work

Interruptions are detected through Voice Activity Detection (VAD) built into the model providers:

```mermaid
flowchart LR
    A[User Starts Speaking] --> B[Model Detects Speech]
    B --> C[BidiInterruptionEvent]
    C --> D[Clear Audio Buffer]
    C --> E[Stop Response]
    E --> F[BidiResponseCompleteEvent]
    B --> G[Transcribe Speech]
    G --> H[BidiTranscriptStreamEvent]
    F --> I[Ready for New Input]
    H --> I
```

### Handling Interruptions

The interruption flow: Model's VAD detects user speech → `BidiInterruptionEvent` sent → Audio buffer cleared → Response terminated → User's speech transcribed → Model ready for new input.

#### Automatic Handling (Default)

When using `BidiAudioIO`, interruptions are handled automatically:

```python
import asyncio
from strands.experimental.bidi import BidiAgent, BidiAudioIO
from strands.experimental.bidi.models import BidiNovaSonicModel

model = BidiNovaSonicModel()
agent = BidiAgent(model=model)
audio_io = BidiAudioIO()

async def main():
    # Interruptions handled automatically
    await agent.run(
        inputs=[audio_io.input()],
        outputs=[audio_io.output()]
    )

asyncio.run(main())
```

The `BidiAudioIO` output automatically clears the audio buffer, stops playback immediately, and resumes normal operation for the next response.

#### Manual Handling

For custom behavior, process interruption events manually:

```python
import asyncio
from strands.experimental.bidi import BidiAgent
from strands.experimental.bidi.models import BidiNovaSonicModel
from strands.experimental.bidi.types.events import (
    BidiInterruptionEvent,
    BidiResponseCompleteEvent
)

model = BidiNovaSonicModel()
agent = BidiAgent(model=model)

async def main():
    await agent.start()
    await agent.send("Tell me a long story")
    
    async for event in agent.receive():
        if isinstance(event, BidiInterruptionEvent):
            print(f"Interrupted: {event.reason}")
            # Custom handling:
            # - Update UI to show interruption
            # - Log analytics
            # - Clear custom buffers
            
        elif isinstance(event, BidiResponseCompleteEvent):
            if event.stop_reason == "interrupted":
                print("Response was interrupted by user")
            break
    
    await agent.stop()

asyncio.run(main())
```

### Interruption Events

#### Key Events

**BidiInterruptionEvent** - Emitted when interruption detected:
- `reason`: `"user_speech"` (most common) or `"error"`

**BidiResponseCompleteEvent** - Includes interruption status:
- `stop_reason`: `"complete"`, `"interrupted"`, `"error"`, or `"tool_use"`

### Interruption Hooks

Use hooks to track interruptions across your application:

```python
from strands.experimental.bidi import BidiAgent
from strands.experimental.bidi.hooks.events import BidiInterruptionEvent as BidiInterruptionHookEvent

class InterruptionTracker:
    def __init__(self):
        self.interruption_count = 0
    
    async def on_interruption(self, event: BidiInterruptionHookEvent):
        self.interruption_count += 1
        print(f"Interruption #{self.interruption_count}: {event.reason}")
        
        # Log to analytics
        # Update UI
        # Track user behavior

tracker = InterruptionTracker()
agent = BidiAgent(
    model=model,
    hooks=[tracker]
)
```

### Provider-Specific Behavior

**Nova Sonic**: Built-in VAD, always active, fast detection optimized for low latency.

**OpenAI Realtime**: Server-side VAD with configurable threshold and silence duration:

```python
from strands.experimental.bidi.models import BidiOpenAIRealtimeModel

model = BidiOpenAIRealtimeModel(
    provider_config={
        "turn_detection": {
            "type": "server_vad",
            "threshold": 0.5,  # Sensitivity (0.0-1.0)
            "prefix_padding_ms": 300,  # Audio before speech
            "silence_duration_ms": 500  # Silence to end turn
        }
    }
)
```

**Gemini Live**: Built-in VAD, fast detection integrated with transcription.

### Best Practices

1. **Always Clear Buffers**: When handling interruptions manually, clear audio/output buffers immediately
2. **Update UI Promptly**: Show visual feedback to maintain user awareness
3. **Track Interruption Patterns**: Monitor frequency to identify issues with response length or relevance
4. **Test VAD Settings**: Adjust sensitivity for your environment
5. **Handle Edge Cases**: Consider rapid interruptions, network delays, and simultaneous tool execution

### Common Issues

#### Interruptions Not Working

If interruptions aren't being detected:

```python
# Check VAD configuration (OpenAI)
model = BidiOpenAIRealtimeModel(
    provider_config={
        "turn_detection": {
            "type": "server_vad",
            "threshold": 0.3,  # Lower = more sensitive
            "silence_duration_ms": 300  # Shorter = faster detection
        }
    }
)

# Verify microphone is working
audio_io = BidiAudioIO(input_device_index=1)  # Specify device

# Check system permissions (macOS)
# System Preferences → Security & Privacy → Microphone
```

#### Audio Continues After Interruption

If audio keeps playing after interruption:

```python
# Ensure BidiAudioIO is handling interruptions
async def __call__(self, event: BidiOutputEvent):
    if isinstance(event, BidiInterruptionEvent):
        self._buffer.clear()  # Critical!
        print("Buffer cleared due to interruption")
```

#### Frequent False Interruptions

If the model is interrupted too easily:

```python
# Increase VAD threshold (OpenAI)
model = BidiOpenAIRealtimeModel(
    provider_config={
        "turn_detection": {
            "threshold": 0.7,  # Higher = less sensitive
            "prefix_padding_ms": 500,  # More context
            "silence_duration_ms": 700  # Longer silence required
        }
    }
)
```


## Configuration

`BidiAgent` supports extensive configuration to customize behavior for your specific use case.

### Basic Configuration

```python
from strands.experimental.bidi import BidiAgent
from strands.experimental.bidi.models import BidiNovaSonicModel

model = BidiNovaSonicModel()

agent = BidiAgent(
    model=model,
    tools=[calculator, weather],
    system_prompt="You are a helpful voice assistant.",
    messages=[],  # Optional conversation history
    agent_id="voice_assistant_1",
    name="Voice Assistant",
    description="A voice-enabled AI assistant"
)
```

### Model Configuration

Each model provider has specific configuration options:

```python
from strands.experimental.bidi.models import BidiNovaSonicModel

model = BidiNovaSonicModel(
    model_id="amazon.nova-sonic-v1:0",
    provider_config={
        "audio": {
            "input_rate": 16000,
            "output_rate": 16000,
            "voice": "matthew",  # or "ruth"
            "channels": 1,
            "format": "pcm"
        }
    },
    client_config={
        "boto_session": boto3.Session(),
        "region": "us-east-1"
    }
)
```

See [Model Providers](models/nova_sonic.md) for provider-specific options.

### Tool Configuration

Tools work the same as in standard agents:

```python
from strands import tool
from strands.experimental.bidi import BidiAgent

@tool
def get_weather(location: str) -> str:
    """Get weather for a location."""
    return f"Weather in {location}: Sunny, 72°F"

agent = BidiAgent(
    model=model,
    tools=[get_weather],
    record_direct_tool_call=True,  # Record tool calls in history
    load_tools_from_directory=False  # Auto-load from ./tools/
)
```

### State Management

Pass custom state to tools:

```python
from strands.experimental.bidi import BidiAgent
from strands import tool

@tool
def get_user_data(invocation_state: dict) -> str:
    """Access user context."""
    user_id = invocation_state["user_id"]
    return f"User: {user_id}"

agent = BidiAgent(model=model, tools=[get_user_data])

# Pass state when starting
await agent.start(invocation_state={
    "user_id": "user_123",
    "session_id": "session_456",
    "database": db_connection
})
```

### Tool Executor

Control how tools execute:

```python
from strands.experimental.bidi import BidiAgent
from strands.tools.executors import ConcurrentToolExecutor

# Default: concurrent execution
agent = BidiAgent(
    model=model,
    tools=[tool1, tool2, tool3],
    tool_executor=ConcurrentToolExecutor()
)

# Custom executor with limits
from strands.tools.executors import ConcurrentToolExecutor

class RateLimitedExecutor(ConcurrentToolExecutor):
    def __init__(self, max_concurrent=3):
        super().__init__()
        self._semaphore = asyncio.Semaphore(max_concurrent)
    
    async def _stream(self, *args, **kwargs):
        async with self._semaphore:
            async for event in super()._stream(*args, **kwargs):
                yield event

agent = BidiAgent(
    model=model,
    tools=[tool1, tool2, tool3],
    tool_executor=RateLimitedExecutor(max_concurrent=2)
)
```

### Hooks

Register hooks for lifecycle events. See [Hooks](hooks.md) for complete documentation.

```python
from strands.experimental.bidi import BidiAgent
from strands.experimental.bidi.hooks.events import (
    BidiAgentInitializedEvent,
    BidiMessageAddedEvent,
    BidiInterruptionEvent
)

class ConversationLogger:
    async def on_agent_initialized(self, event: BidiAgentInitializedEvent):
        print(f"Agent {event.agent.agent_id} initialized")
    
    async def on_message_added(self, event: BidiMessageAddedEvent):
        message = event.message
        print(f"{message['role']}: {message['content']}")
    
    async def on_interruption(self, event: BidiInterruptionEvent):
        print(f"Interrupted: {event.reason}")

agent = BidiAgent(
    model=model,
    hooks=[ConversationLogger()]
)
```

### Session Management

Integrate with session management for persistence. See [Session Management](session-management.md) for complete documentation.

```python
from strands.experimental.bidi import BidiAgent
from strands.session.file_session_manager import FileSessionManager

session_manager = FileSessionManager(session_id="user_123_session")

agent = BidiAgent(
    model=model,
    session_manager=session_manager
)

# Messages automatically persisted
await agent.start()
```


## Lifecycle Management

Understanding the `BidiAgent` lifecycle is crucial for proper resource management and error handling.

### Lifecycle States

```mermaid
stateDiagram-v2
    [*] --> Created: BidiAgent
    Created --> Started: start
    Started --> Running: run or receive
    Running --> Running: send and receive events
    Running --> Stopped: stop
    Stopped --> [*]
    
    Running --> Restarting: Timeout
    Restarting --> Running: Reconnected
```

### State Transitions

#### 1. Creation

```python
agent = BidiAgent(model=model, tools=[calculator])
# Tool registry initialized, agent state created, hooks registered
# NOT connected to model yet
```

#### 2. Starting

```python
await agent.start(invocation_state={...})
# Model connection established, conversation history sent
# Background tasks spawned, ready to send/receive
```

#### 3. Running

```python
# Option A: Using run()
await agent.run(inputs=[...], outputs=[...])

# Option B: Manual send/receive
await agent.send("Hello")
async for event in agent.receive():
    # Process events - events streaming, tools executing, messages accumulating
    pass
```

#### 4. Stopping

```python
await agent.stop()
# Background tasks cancelled, model connection closed, resources cleaned up
```

### Lifecycle Patterns

#### Using run()

```python
agent = BidiAgent(model=model)
audio_io = BidiAudioIO()

await agent.run(
    inputs=[audio_io.input()],
    outputs=[audio_io.output()]
)
```

Simplest for I/O-based applications - handles start/stop automatically.

#### Context Manager

```python
agent = BidiAgent(model=model)

async with agent:
    await agent.send("Hello")
    async for event in agent.receive():
        if isinstance(event, BidiResponseCompleteEvent):
            break
```

Automatic `start()` and `stop()` with exception-safe cleanup. To pass `invocation_state`, call `start()` manually before entering the context.

#### Manual Lifecycle

```python
agent = BidiAgent(model=model)

try:
    await agent.start()
    await agent.send("Hello")
    
    async for event in agent.receive():
        if isinstance(event, BidiResponseCompleteEvent):
            break
finally:
    await agent.stop()
```

Explicit control with custom error handling and flexible timing.

### Connection Restart

When a model times out, the agent automatically restarts:

```python
async for event in agent.receive():
    if isinstance(event, BidiConnectionRestartEvent):
        print("Reconnecting...")
        # Connection restarting automatically
        # Conversation history preserved
        # Continue processing events normally
```

The restart process: Timeout detected → `BidiConnectionRestartEvent` emitted → Sending blocked → Hooks invoked → Model restarted with history → New receiver task spawned → Sending unblocked → Conversation continues seamlessly.

### Error Handling

#### Handling Errors in Events

```python
async for event in agent.receive():
    if isinstance(event, BidiErrorEvent):
        print(f"Error: {event.message}")
        # Access original exception
        original_error = event.error
        # Decide whether to continue or break
        break
```

#### Handling Connection Errors

```python
try:
    await agent.start()
    async for event in agent.receive():
        # Handle connection restart events
        if isinstance(event, BidiConnectionRestartEvent):
            print("Connection restarting, please wait...")
            continue  # Connection restarts automatically
        
        # Process other events
        pass
except Exception as e:
    print(f"Unexpected error: {e}")
finally:
    await agent.stop()
```

**Note:** Connection timeouts are handled automatically. The agent emits `BidiConnectionRestartEvent` when reconnecting.

#### Graceful Shutdown

```python
import signal

agent = BidiAgent(model=model)
audio_io = BidiAudioIO()

async def main():
    # Setup signal handler
    loop = asyncio.get_event_loop()
    
    def signal_handler():
        print("\nShutting down gracefully...")
        loop.create_task(agent.stop())
    
    loop.add_signal_handler(signal.SIGINT, signal_handler)
    loop.add_signal_handler(signal.SIGTERM, signal_handler)
    
    try:
        await agent.run(
            inputs=[audio_io.input()],
            outputs=[audio_io.output()]
        )
    except asyncio.CancelledError:
        print("Agent stopped")

asyncio.run(main())
```

### Resource Cleanup

The agent automatically cleans up background tasks, model connections, I/O channels, event queues, and invokes cleanup hooks.

### Best Practices

1. **Always Use try/finally**: Ensure `stop()` is called even on errors
2. **Prefer Context Managers**: Use `async with` for automatic cleanup
3. **Handle Restarts Gracefully**: Don't treat `BidiConnectionRestartEvent` as an error
4. **Monitor Lifecycle Hooks**: Use hooks to track state transitions
5. **Test Shutdown**: Verify cleanup works under various conditions
6. **Avoid Calling stop() During receive()**: Only call `stop()` after exiting the receive loop

## Next Steps

- [Events](events.md) - Complete guide to bidirectional streaming events
- [I/O Channels](io/io.md) - Building custom input/output channels
- [Model Providers](models/nova_sonic.md) - Provider-specific configuration
- [Quickstart](quickstart.md) - Getting started guide
- [API Reference](../../../../api-reference/experimental.md) - Complete API documentation