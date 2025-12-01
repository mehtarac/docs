# Session Management

Session management for `BidiAgent` provides a mechanism for persisting conversation history and agent state across bidirectional streaming sessions. This enables voice assistants and interactive applications to maintain context and continuity even when connections are restarted or the application is redeployed.

## Overview

A bidirectional streaming session represents all stateful information needed by the agent to function, including:

- Conversation history (messages with audio transcripts)
- Agent state (key-value storage)
- Connection state and configuration
- Tool execution history

Strands provides built-in session persistence capabilities that automatically capture and restore this information, allowing `BidiAgent` to seamlessly continue conversations where they left off, even after connection timeouts or application restarts.

## Basic Usage

Create a `BidiAgent` with a session manager and use it:

```python
from strands.experimental.bidi import BidiAgent, BidiAudioIO
from strands.experimental.bidi.models import BidiNovaSonicModel
from strands.session.file_session_manager import FileSessionManager

# Create a session manager with a unique session ID
session_manager = FileSessionManager(session_id="user_123_voice_session")

# Create the agent with session management
model = BidiNovaSonicModel()
agent = BidiAgent(
    model=model,
    session_manager=session_manager
)

# Use the agent - all messages are automatically persisted
audio_io = BidiAudioIO()
await agent.run(
    inputs=[audio_io.input()],
    outputs=[audio_io.output()]
)
```

The conversation history is automatically persisted and will be restored on the next session.

## Built-in Session Managers

Strands offers two built-in session managers for persisting bidirectional streaming sessions:

1. **FileSessionManager**: Stores sessions in the local filesystem
2. **S3SessionManager**: Stores sessions in Amazon S3 buckets

### FileSessionManager

The `FileSessionManager` provides a simple way to persist sessions to the local filesystem:

```python
from strands.experimental.bidi import BidiAgent
from strands.session.file_session_manager import FileSessionManager

# Create a session manager
session_manager = FileSessionManager(
    session_id="user_123_session",
    storage_dir="/path/to/sessions"  # Optional, defaults to temp directory
)

agent = BidiAgent(
    model=model,
    session_manager=session_manager
)
```

**Use cases:**
- Development and testing
- Single-server deployments
- Local voice assistants
- Prototyping

### S3SessionManager

The `S3SessionManager` stores sessions in Amazon S3 for distributed deployments:

```python
from strands.experimental.bidi import BidiAgent
from strands.session.s3_session_manager import S3SessionManager

# Create an S3 session manager
session_manager = S3SessionManager(
    session_id="user_123_session",
    bucket="my-voice-sessions",
    prefix="sessions/"  # Optional prefix for organization
)

agent = BidiAgent(
    model=model,
    session_manager=session_manager
)
```

**Use cases:**
- Production deployments
- Multi-server environments
- Serverless applications
- High availability requirements

## Session Lifecycle

### Session Creation

Sessions are created automatically when the agent starts:

```python
session_manager = FileSessionManager(session_id="new_session")
agent = BidiAgent(model=model, session_manager=session_manager)

# Session created on first start
await agent.start()
```

### Session Restoration

When an agent starts with an existing session ID, the conversation history is automatically restored:

```python
# First conversation
session_manager = FileSessionManager(session_id="user_123")
agent = BidiAgent(model=model, session_manager=session_manager)
await agent.start()
await agent.send("My name is Alice")
# ... conversation continues ...
await agent.stop()

# Later - conversation history restored
session_manager = FileSessionManager(session_id="user_123")
agent = BidiAgent(model=model, session_manager=session_manager)
await agent.start()  # Previous messages automatically loaded
await agent.send("What's my name?")  # Agent remembers: "Alice"
```

### Session Updates

Messages are persisted automatically as they're added:

```python
agent = BidiAgent(model=model, session_manager=session_manager)
await agent.start()

# Each message automatically saved
await agent.send("Hello")  # Saved
# Model response received and saved
# Tool execution saved
# All transcripts saved
```

## Connection Restart Behavior

When a connection times out and restarts, the session manager ensures continuity:

```python
agent = BidiAgent(model=model, session_manager=session_manager)
await agent.start()

async for event in agent.receive():
    if isinstance(event, BidiConnectionRestartEvent):
        # Connection restarting due to timeout
        # Session manager ensures:
        # 1. All messages up to this point are saved
        # 2. Full history sent to restarted connection
        # 3. Conversation continues seamlessly
        print("Reconnecting with full history preserved")
```

## Integration with Hooks

Session management works seamlessly with hooks:

```python
from strands.experimental.bidi.hooks.events import BidiMessageAddedEvent

class SessionLogger:
    async def on_message_added(self, event: BidiMessageAddedEvent):
        # Message already persisted by session manager
        print(f"Message persisted: {event.message['role']}")

agent = BidiAgent(
    model=model,
    session_manager=session_manager,
    hooks=[SessionLogger()]
)
```

The `BidiMessageAddedEvent` is emitted after the message is persisted, ensuring hooks see the saved state.

## Best Practices

### Session ID Management

Use meaningful, unique session IDs:

```python
# Good: User-specific session IDs
session_id = f"user_{user_id}_voice_{timestamp}"

# Good: Device-specific for voice assistants
session_id = f"device_{device_id}_session"

# Avoid: Generic or reused IDs
session_id = "session"  # ❌ Will overwrite previous sessions
```

### Session Cleanup

Implement session cleanup for old or completed sessions:

```python
from datetime import datetime, timedelta
from pathlib import Path

def cleanup_old_sessions(storage_dir: str, days: int = 30):
    """Remove sessions older than specified days."""
    cutoff = datetime.now() - timedelta(days=days)
    
    for session_file in Path(storage_dir).glob("*.json"):
        if session_file.stat().st_mtime < cutoff.timestamp():
            session_file.unlink()
            print(f"Removed old session: {session_file.name}")

# Run periodically
cleanup_old_sessions("/path/to/sessions", days=30)
```

### Error Handling

Handle session loading errors gracefully:

```python
try:
    session_manager = FileSessionManager(session_id=session_id)
    agent = BidiAgent(model=model, session_manager=session_manager)
    await agent.start()
except Exception as e:
    logger.error(f"Failed to load session: {e}")
    # Fall back to new session
    session_manager = FileSessionManager(session_id=f"{session_id}_new")
    agent = BidiAgent(model=model, session_manager=session_manager)
    await agent.start()
```

### Storage Considerations

**FileSessionManager:**
- Ensure sufficient disk space
- Use fast storage (SSD) for better performance
- Implement backup strategies
- Consider file system limits on number of files

**S3SessionManager:**
- Configure appropriate S3 lifecycle policies
- Use S3 versioning for recovery
- Consider costs for storage and API calls
- Implement proper IAM permissions

## Troubleshooting

### Session Not Loading

If sessions aren't loading:

```python
# Verify session file exists
from pathlib import Path

session_file = Path(storage_dir) / f"{session_id}.json"
if not session_file.exists():
    print(f"Session file not found: {session_file}")

# Check file permissions
if not session_file.is_readable():
    print(f"Cannot read session file: {session_file}")

# Enable debug logging
import logging
logging.getLogger("strands.session").setLevel(logging.DEBUG)
```

### Session Corruption

If a session file is corrupted:

```python
import json

try:
    with open(session_file) as f:
        data = json.load(f)
except json.JSONDecodeError as e:
    print(f"Corrupted session file: {e}")
    # Remove corrupted file and start fresh
    session_file.unlink()
```

### S3 Permission Issues

If S3 sessions fail:

```python
# Verify IAM permissions
# Required permissions:
# - s3:GetObject
# - s3:PutObject
# - s3:ListBucket

# Test S3 access
import boto3

s3 = boto3.client('s3')
try:
    s3.head_bucket(Bucket='my-sessions')
    print("S3 bucket accessible")
except Exception as e:
    print(f"S3 access error: {e}")
```

## Next Steps

- [Agent](agent.md) - Learn about BidiAgent configuration and lifecycle
- [Hooks](hooks.md) - Extend agent functionality with hooks
- [Events](events.md) - Complete guide to bidirectional streaming events
- [API Reference](../../../../api-reference/experimental.md) - Complete API documentation
