#!/usr/bin/env python3
"""
Phone Agent CLI - AI-powered HarmonyOS automation (HDC).

Usage:
    python main.py [OPTIONS]

Environment Variables:
    PHONE_AGENT_BASE_URL: Model API base URL (default: http://localhost:8000/v1)
    PHONE_AGENT_MODEL: Model name (default: autoglm-phone-9b)
    PHONE_AGENT_API_KEY: API key for model authentication (default: EMPTY)
    PHONE_AGENT_MAX_STEPS: Maximum steps per task (default: 100)
    PHONE_AGENT_DEVICE_ID: HDC device ID for multi-device setups
"""

import argparse
import os
import shutil
import subprocess
import sys

from openai import OpenAI

from phone_agent import PhoneAgent
from phone_agent.agent import AgentConfig
from phone_agent.config import list_supported_apps
from phone_agent.device_factory import DeviceType, get_device_factory, set_device_type
from phone_agent.model import ModelConfig


def check_system_requirements(device_id: str | None = None) -> bool:
    """
    Check HarmonyOS requirements before running the agent.

    Checks:
    1. HDC installed and accessible
    2. At least one device connected

    Args:
        device_id: Optional target device ID.

    Returns:
        True if all checks pass, False otherwise.
    """
    print("Checking HarmonyOS environment...")
    print("-" * 50)

    all_passed = True

    # Check 1: HDC installed
    print("1. Checking HDC installation...", end=" ")
    tool_cmd = "hdc"
    if shutil.which(tool_cmd) is None:
        print("FAILED")
        print("   Error: HDC is not installed or not in PATH.")
        print("   Solution:")
        print("     - Download from HarmonyOS SDK or https://gitee.com/openharmony/docs")
        print("     - Add the hdc binary to your PATH")
        all_passed = False
    else:
        try:
            result = subprocess.run(
                [tool_cmd, "-v"], capture_output=True, text=True, timeout=10
            )
            version_line = result.stdout.strip().split("\n")[0]
            print(f"OK ({version_line if version_line else 'installed'})")
        except FileNotFoundError:
            print("FAILED")
            print("   Error: hdc command not found.")
            all_passed = False
        except subprocess.TimeoutExpired:
            print("FAILED")
            print("   Error: hdc command timed out.")
            all_passed = False

    if not all_passed:
        print("-" * 50)
        print("System check failed. Please fix the issues above.")
        return False

    # Check 2: Device connected
    print("2. Checking connected devices...", end=" ")
    try:
        result = subprocess.run(
            ["hdc", "list", "targets"], capture_output=True, text=True, timeout=10
        )
        lines = result.stdout.strip().split("\n")
        devices = [line for line in lines if line.strip()]

        if not devices:
            print("FAILED")
            print("   Error: No HarmonyOS devices connected.")
            print("   Solution:")
            print("     1. Enable USB debugging on your HarmonyOS device")
            print("     2. Connect via USB and authorize the connection")
            print("     3. Or connect remotely: python main.py --connect <ip>:<port>")
            all_passed = False
        else:
            device_ids = [d.strip() for d in devices]
            print(
                f"OK ({len(devices)} device(s): {', '.join(device_ids[:2])}{'...' if len(device_ids) > 2 else ''})"
            )
    except subprocess.TimeoutExpired:
        print("FAILED")
        print("   Error: hdc command timed out.")
        all_passed = False
    except Exception as e:
        print("FAILED")
        print(f"   Error: {e}")
        all_passed = False

    print("-" * 50)

    if all_passed:
        print("All system checks passed!\n")
    else:
        print("System check failed. Please fix the issues above.")

    return all_passed


def check_model_api(base_url: str, model_name: str, api_key: str = "EMPTY") -> bool:
    """
    Check if the model API is accessible and the specified model exists.

    Checks:
    1. Network connectivity to the API endpoint
    2. Model exists in the available models list

    Args:
        base_url: The API base URL
        model_name: The model name to check
        api_key: The API key for authentication

    Returns:
        True if all checks pass, False otherwise.
    """
    print("Checking model API...")
    print("-" * 50)

    all_passed = True

    # Check 1: Network connectivity using chat API
    print(f"1. Checking API connectivity ({base_url})...", end=" ")
    try:
        # Create OpenAI client
        client = OpenAI(base_url=base_url, api_key=api_key, timeout=30.0)

        # Use chat completion to test connectivity (more universally supported than /models)
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=5,
            temperature=0.0,
            stream=False,
        )

        # Check if we got a valid response
        if response.choices and len(response.choices) > 0:
            print("OK")
        else:
            print("FAILED")
            print("   Error: Received empty response from API")
            all_passed = False

    except Exception as e:
        print("FAILED")
        error_msg = str(e)

        # Provide more specific error messages
        if "Connection refused" in error_msg or "Connection error" in error_msg:
            print(f"   Error: Cannot connect to {base_url}")
            print("   Solution:")
            print("     1. Check if the model server is running")
            print("     2. Verify the base URL is correct")
            print(f"     3. Try: curl {base_url}/chat/completions")
        elif "timed out" in error_msg.lower() or "timeout" in error_msg.lower():
            print(f"   Error: Connection to {base_url} timed out")
            print("   Solution:")
            print("     1. Check your network connection")
            print("     2. Verify the server is responding")
        elif (
            "Name or service not known" in error_msg
            or "nodename nor servname" in error_msg
        ):
            print(f"   Error: Cannot resolve hostname")
            print("   Solution:")
            print("     1. Check the URL is correct")
            print("     2. Verify DNS settings")
        else:
            print(f"   Error: {error_msg}")

        all_passed = False

    print("-" * 50)

    if all_passed:
        print("Model API checks passed!\n")
    else:
        print("Model API check failed. Please fix the issues above.")

    return all_passed


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Phone Agent - AI-powered HarmonyOS automation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run with default settings (HarmonyOS via HDC)
    python main.py "打开微信发送一条消息"

    # Reproduce leak case from DevEcoTesting artifacts by timestamp (ms)
    python main.py --leak-ts-ms 1769048341000

    # Specify model endpoint
    python main.py --base-url http://localhost:8000/v1

    # Use API key for authentication
    python main.py --apikey sk-xxxxx

    # Run with specific device
    python main.py --device-id FMR0XXXXXX

    # Connect to remote device
    python main.py --connect 192.168.1.100:5555

    # List connected devices
    python main.py --list-devices

    # Enable TCP/IP on USB device and get connection info
    python main.py --enable-tcpip

    # List supported HarmonyOS apps
    python main.py --list-apps
        """,
    )

    # Model options
    parser.add_argument(
        "--base-url",
        type=str,
        default=os.getenv("PHONE_AGENT_BASE_URL", "http://localhost:8000/v1"),
        help="Model API base URL",
    )

    parser.add_argument(
        "--model",
        type=str,
        default=os.getenv("PHONE_AGENT_MODEL", "autoglm-phone-9b"),
        help="Model name",
    )

    parser.add_argument(
        "--apikey",
        type=str,
        default=os.getenv("PHONE_AGENT_API_KEY", "EMPTY"),
        help="API key for model authentication",
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=int(os.getenv("PHONE_AGENT_MAX_STEPS", "100")),
        help="Maximum steps per task",
    )

    # Device options
    parser.add_argument(
        "--device-id",
        "-d",
        type=str,
        default=os.getenv("PHONE_AGENT_DEVICE_ID"),
        help="HDC device ID",
    )

    parser.add_argument(
        "--connect",
        "-c",
        type=str,
        metavar="ADDRESS",
        help="Connect to remote device (e.g., 192.168.1.100:5555)",
    )

    parser.add_argument(
        "--disconnect",
        type=str,
        nargs="?",
        const="all",
        metavar="ADDRESS",
        help="Disconnect from remote device (or 'all' to disconnect all)",
    )

    parser.add_argument(
        "--list-devices", action="store_true", help="List connected devices and exit"
    )

    parser.add_argument(
        "--enable-tcpip",
        type=int,
        nargs="?",
        const=5555,
        metavar="PORT",
        help="Enable TCP/IP debugging on USB device (default port: 5555)",
    )

    # Other options
    parser.add_argument(
        "--quiet", "-q", action="store_true", help="Suppress verbose output"
    )

    parser.add_argument(
        "--list-apps", action="store_true", help="List supported apps and exit"
    )

    # Leak reproduction options (DevEcoTesting)
    parser.add_argument(
        "--leak-ts-ms",
        type=int,
        default=0,
        help="Leak timestamp in epoch milliseconds; when set, runs leak reproduction workflow using DevEcoTesting artifacts",
    )
    parser.add_argument(
        "--devecotesting-root",
        type=str,
        default="devecotesting",
        help="DevEcoTesting artifact root directory (default: devecotesting)",
    )
    parser.add_argument("--pre-window-s", type=int, default=30)
    parser.add_argument("--post-window-s", type=int, default=10)
    parser.add_argument("--max-actions", type=int, default=6)
    parser.add_argument("--max-screenshots", type=int, default=2)
    parser.add_argument("--suspect-k", type=int, default=3)

    parser.add_argument(
        "task",
        nargs="?",
        type=str,
        help="Task to execute (interactive mode if not provided)",
    )

    return parser.parse_args()


def handle_device_commands(args) -> bool:
    """
    Handle device-related commands.

    Returns:
        True if a device command was handled (should exit), False otherwise.
    """
    device_factory = get_device_factory()
    ConnectionClass = device_factory.get_connection_class()
    conn = ConnectionClass()

    # Handle --list-devices
    if args.list_devices:
        devices = device_factory.list_devices()
        if not devices:
            print("No devices connected.")
        else:
            print("Connected devices:")
            print("-" * 60)
            for device in devices:
                status_icon = "OK" if device.status == "device" else "NO"
                conn_type = device.connection_type.value
                model_info = f" ({device.model})" if device.model else ""
                print(
                    f"  {status_icon} {device.device_id:<30} [{conn_type}]{model_info}"
                )
        return True

    # Handle --connect
    if args.connect:
        print(f"Connecting to {args.connect}...")
        success, message = conn.connect(args.connect)
        print(f"{'OK' if success else 'NO'} {message}")
        if success:
            # Set as default device
            args.device_id = args.connect
        return not success  # Continue if connection succeeded

    # Handle --disconnect
    if args.disconnect:
        if args.disconnect == "all":
            print("Disconnecting all remote devices...")
            success, message = conn.disconnect()
        else:
            print(f"Disconnecting from {args.disconnect}...")
            success, message = conn.disconnect(args.disconnect)
        print(f"{'OK' if success else 'NO'} {message}")
        return True

    # Handle --enable-tcpip
    if args.enable_tcpip:
        port = args.enable_tcpip
        print(f"Enabling TCP/IP debugging on port {port}...")

        success, message = conn.enable_tcpip(port, args.device_id)
        print(f"{'OK' if success else 'NO'} {message}")

        if success:
            # Try to get device IP
            ip = conn.get_device_ip(args.device_id)
            if ip:
                print(f"\nYou can now connect remotely using:")
                print(f"  python main.py --connect {ip}:{port}")
                print(f"\nOr via HDC directly:")
                print(f"  hdc tconn {ip}:{port}")
            else:
                print("\nCould not determine device IP. Check device WiFi settings.")
        return True

    return False


def main():
    """Main entry point."""
    args = parse_args()
    # Always operate in HDC mode
    set_device_type(DeviceType.HDC)

    # Enable verbose HDC output to aid troubleshooting
    from phone_agent.hdc import set_hdc_verbose

    set_hdc_verbose(False)

    # Handle --list-apps (no system check needed)
    if args.list_apps:
        print("Supported HarmonyOS apps:")
        for app in sorted(list_supported_apps()):
            print(f"  - {app}")
        return

    # Handle device commands (these may need partial system checks)
    if handle_device_commands(args):
        return

    # Run system requirements check before proceeding
    if not check_system_requirements(args.device_id):
        sys.exit(1)

    # Check model API connectivity and model availability
    if not check_model_api(args.base_url, args.model, args.apikey):
        sys.exit(1)

    # Resolve device id early so all HDC commands consistently include `-t <device>`.
    device_factory = get_device_factory()
    devices = device_factory.list_devices()
    resolved_device_id: str | None = args.device_id
    if not resolved_device_id and devices:
        resolved_device_id = devices[0].device_id

    # Create configurations and agent
    model_config = ModelConfig(
        base_url=args.base_url,
        model_name=args.model,
        api_key=args.apikey,
        lang="cn",
    )

    agent_config = AgentConfig(
        max_steps=args.max_steps,
        device_id=resolved_device_id,
        verbose=not args.quiet,
        lang="cn",
    )

    agent = PhoneAgent(
        model_config=model_config,
        agent_config=agent_config,
    )

    # Print header
    print("=" * 50)
    print("Phone Agent - AI-powered HarmonyOS automation")
    print("=" * 50)
    print(f"Model: {model_config.model_name}")
    print(f"Base URL: {model_config.base_url}")
    print(f"Max Steps: {agent_config.max_steps}")
    print(f"Language: {agent_config.lang}")

    # Show device info
    if agent_config.device_id:
        print(f"Device: {agent_config.device_id}")
    elif devices:
        # Should not happen due to early resolution, but keep as a fallback.
        agent_config.device_id = devices[0].device_id
        agent.action_handler.device_id = agent_config.device_id
        print(f"Device: {agent_config.device_id} (auto-detected)")

    print("=" * 50)

    # Leak reproduction flow (timestamp-driven, no manual task required)
    if args.leak_ts_ms:
        from workflow.devecotesting_case_builder import BuildOptions, build_leak_case_from_devecotesting
        from workflow.leak_system_prompt import LEAK_SYSTEM_PROMPT
        from workflow.prompt_builder import (
            build_leak_case_extra_messages,
            build_leak_case_task_hint,
        )
        from workflow.screen_match import similarity_file_vs_base64_jpeg
        from workflow.sequence_extract import extract_repro_sequence_from_finish_message

        options = BuildOptions(
            pre_window_s=args.pre_window_s,
            post_window_s=args.post_window_s,
            max_actions=args.max_actions,
            max_screenshots=args.max_screenshots,
            suspect_k=args.suspect_k,
        )

        case = build_leak_case_from_devecotesting(
            args.leak_ts_ms, args.devecotesting_root, options=options
        )
        extra_messages = build_leak_case_extra_messages(case)
        task_hint = build_leak_case_task_hint(case)

        # Leak-mode: use a shorter system prompt and reduce completion token budget.
        if case.target_app_name or case.target_bundle_name:
            agent.agent_config.system_prompt = (
                LEAK_SYSTEM_PROMPT
                + f"\n目标APP（来自 DevEcoTesting layout）: {case.target_app_name or '未知'} / {case.target_bundle_name or ''}\n"
            )
        else:
            agent.agent_config.system_prompt = LEAK_SYSTEM_PROMPT
        agent.model_config.max_tokens = min(agent.model_config.max_tokens, 1200)
        agent.agent_config.max_same_screen_steps = min(agent.agent_config.max_same_screen_steps, 8)

        print(f"\nLeak Case: {case.case_id}")
        print(f"Leak ts(ms): {case.leak_ts_ms}")
        if case.target_app_name or case.target_bundle_name:
            print(f"Target app: {case.target_app_name} ({case.target_bundle_name})")
        print(f"Actions: {len(case.actions)}")
        print(f"Screenshots: {len(case.screenshots)}\n")

        # Provide a numeric screen match score to help the model stay anchored to the target scene.
        pre_ref = case.key_screenshots.get("pre_leak")
        if pre_ref is not None:
            def _hook(current_app: str, screen_base64: str):
                try:
                    score = similarity_file_vs_base64_jpeg(pre_ref.path, screen_base64)
                except Exception:
                    score = None
                return {
                    "leak_mode": True,
                    "target_app": case.target_app_name,
                    "pre_leak_ts_ms": pre_ref.ts_ms,
                    "pre_leak_match_score": score,
                }

            agent.agent_config.screen_info_hook = _hook

        result = agent.run(task_hint, extra_messages=extra_messages)
        print(f"\nResult: {result}")

        seq = extract_repro_sequence_from_finish_message(result)
        if seq is not None:
            print("\nDetected <LEAK_SEQUENCE_READY> with JSON payload.")
            repeat_env = os.getenv("PHONE_AGENT_REPLAY_COUNT", "10")
            try:
                repeat_count = max(1, int(repeat_env))
            except ValueError:
                repeat_count = 3

            try:
                replay_msgs = agent.execute_repro_sequence(seq, repeat_count=repeat_count, inter_step_wait_s=0.8)
                print(f"\nReplayed sequence repeat_count={repeat_count}:")
                for msg in replay_msgs:
                    print(f"  - {msg}")
            except Exception as e:
                print(f"\nReplay failed: {e}")
        return

    # Run with provided task or enter interactive mode
    if args.task:
        print(f"\nTask: {args.task}\n")
        result = agent.run(args.task)
        print(f"\nResult: {result}")
    else:
        # Interactive mode
        print("\nEntering interactive mode. Type 'quit' to exit.\n")

        while True:
            try:
                task = input("Enter your task: ").strip()

                if task.lower() in ("quit", "exit", "q"):
                    print("Goodbye!")
                    break

                if not task:
                    continue

                print()
                result = agent.run(task)
                print(f"\nResult: {result}\n")
                agent.reset()

            except KeyboardInterrupt:
                print("\n\nInterrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\nError: {e}\n")


if __name__ == "__main__":
    main()
