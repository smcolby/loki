"""Tests for the stop subcommand — docker compose down."""

from click.testing import CliRunner

from loki.cli import cli


def test_stop_calls_docker_compose_down(mocker):
    """The stop command invokes docker compose down via subprocess."""
    mock_run = mocker.patch("loki.cli.subprocess.run", autospec=True)

    CliRunner().invoke(cli, ["stop"])

    mock_run.assert_called_once_with(["docker", "compose", "down"], check=False)


def test_stop_prints_message(mocker):
    """The stop command prints a status message before stopping the stack."""
    mocker.patch("loki.cli.subprocess.run", autospec=True)

    result = CliRunner().invoke(cli, ["stop"])

    assert "Stopping" in result.output
