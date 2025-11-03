# Security Policy

## Supported Versions

We release patches for security vulnerabilities in the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 0.2.x   | :white_check_mark: |
| < 0.2.0 | :x:                |

## Reporting a Vulnerability

We take the security of strands-mlx seriously. If you discover a security vulnerability, please follow these steps:

### 🔒 Private Disclosure

**DO NOT** open a public GitHub issue for security vulnerabilities.

Instead, please report security issues by emailing:

**📧 cagatay@strandsagents.com**

### What to Include

Please include the following information in your report:

- **Description** - Clear description of the vulnerability
- **Impact** - Potential impact and severity
- **Steps to Reproduce** - Detailed steps to reproduce the issue
- **Proof of Concept** - Code or commands demonstrating the vulnerability
- **Environment** - Python version, OS, hardware (Apple Silicon, etc.)
- **Suggested Fix** - If you have ideas on how to fix it

### What to Expect

- **Acknowledgment** - We'll acknowledge receipt within 48 hours
- **Investigation** - We'll investigate and validate the report
- **Updates** - We'll keep you informed of our progress
- **Resolution** - We'll work on a fix and coordinate disclosure
- **Credit** - We'll credit you in the security advisory (if you wish)

---

## Security Best Practices

### Training Data Security

**⚠️ Training data may contain sensitive information:**

```python
# ❌ DON'T: Store sensitive data in training files
session = MLXSessionManager(session_id="passwords", storage_dir="./data")
agent("My password is secret123")  # Saved to JSONL!

# ✅ DO: Sanitize or exclude sensitive conversations
session = MLXSessionManager(
    session_id="safe_training",
    storage_dir="./secure_data"
)
# Be mindful of what data gets trained on
```

### Model Adapter Security

**🔐 Trained adapters can leak training data:**

- LoRA adapters contain model weights trained on your data
- Don't share adapters trained on private/sensitive information
- Review training data before sharing adapters on HuggingFace

### File System Access

**📁 Session managers write to disk:**

```python
# Be careful with file paths
session = MLXSessionManager(
    session_id="user_input",  # ⚠️ Sanitize user-provided IDs
    storage_dir="/safe/directory"  # ✅ Use absolute paths
)
```

### MLX Model Loading

**🤖 Only load models from trusted sources:**

```python
# ✅ Official mlx-community models
model = MLXModel(model_id="mlx-community/Qwen3-1.7B-4bit")

# ⚠️ Verify adapter sources before loading
model = MLXModel(
    model_id="mlx-community/Qwen3-1.7B-4bit",
    adapter_path="untrusted/adapter"  # Could contain malicious code
)
```

---

## Known Security Considerations

### 1. Training Data Persistence

- **Issue:** Conversation data saved to JSONL files
- **Impact:** Sensitive information may persist on disk
- **Mitigation:** 
  - Use secure directories with proper permissions
  - Implement data retention policies
  - Sanitize sensitive data before training

### 2. Model Adapter Distribution

- **Issue:** Adapters can memorize training data
- **Impact:** Private information could be extracted from adapters
- **Mitigation:**
  - Review training data before sharing adapters
  - Use differential privacy techniques (future feature)
  - Limit adapter distribution

### 3. Tool Execution in Agents

- **Issue:** Agents can execute tools with system access
- **Impact:** Malicious prompts could abuse tool capabilities
- **Mitigation:**
  - Carefully select which tools to enable
  - Validate tool inputs
  - Use principle of least privilege

### 4. Vision Model Inputs

- **Issue:** Image/audio/video files processed by models
- **Impact:** Malicious media could exploit vulnerabilities
- **Mitigation:**
  - Validate media file formats
  - Use trusted media sources
  - Implement file size limits

---

## Security Updates

Security updates will be released as patch versions (e.g., 0.2.5 → 0.2.6) and announced via:

- **GitHub Security Advisories**
- **PyPI release notes**
- **Repository CHANGELOG.md**

---

## Dependencies

strands-mlx relies on several dependencies. We monitor these for vulnerabilities:

- **mlx** - Apple's ML framework
- **mlx-lm** - Language model support
- **mlx-vlm** - Vision/audio/video support (optional)
- **strands-agents** - Core agent framework

Run security checks:

```bash
# Check for known vulnerabilities
pip install safety
safety check

# Or with uv
uv pip install safety
safety check
```

---

## Responsible Disclosure

We believe in responsible disclosure. If you report a vulnerability:

1. We'll work with you to understand and validate the issue
2. We'll develop and test a fix
3. We'll coordinate public disclosure timing
4. We'll credit you in the security advisory (unless you prefer anonymity)

Thank you for helping keep strands-mlx and its users safe! 🛡️

---

## Contact

- **Security Issues:** cagatay@strandsagents.com
- **General Issues:** [GitHub Issues](https://github.com/cagataycali/strands-mlx/issues)
- **Discussions:** [GitHub Discussions](https://github.com/cagataycali/strands-mlx/discussions)
