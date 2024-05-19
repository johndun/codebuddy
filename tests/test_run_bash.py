from codebuddy.utils import run_bash


def test_run_bash_success():
    # Test a simple command
    result = run_bash('echo "Hello, World!"')
    assert result == "Hello, World!\n"


def test_run_bash_failure():
    # Test a command that fails
    result = run_bash("exit 1")
    assert result is None


def test_run_bash_command_not_found():
    # Test a command that does not exist
    result = run_bash("nonexistent_command")
    assert result is None
