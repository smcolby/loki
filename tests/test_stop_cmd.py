"""Tests for the stop subcommand — docker compose down and avahi-publish teardown."""

from click.testing import CliRunner

from loki.cli import cli


def test_stop_calls_docker_compose_down(mocker):
    """The stop command invokes docker compose down via subprocess."""
    mock_run = mocker.patch("loki.cli.subprocess.run", autospec=True)

    CliRunner().invoke(cli, ["stop"])

    cmd = mock_run.call_args.args[0]
    assert cmd[:2] == ["docker", "compose"]
    assert "--project-directory" in cmd
    assert cmd[-1] == "down"
    assert mock_run.call_args.kwargs == {"check": False}


def test_stop_prints_message(mocker):
    """The stop command prints a status message before stopping the stack."""
    mocker.patch("loki.cli.subprocess.run", autospec=True)

    result = CliRunner().invoke(cli, ["stop"])

    assert "Stopping" in result.output


def test_stop_kills_avahi_process(mocker):
    """The stop command calls stop_avahi_publish to clean up the mDNS process."""
    mocker.patch("loki.cli.subprocess.run", autospec=True)
    mock_stop = mocker.patch("loki.cli.stop_avahi_publish")

    CliRunner().invoke(cli, ["stop"])

    mock_stop.assert_called_once()


def test_stop_avahi_called_before_compose_down(mocker):
    """The stop command terminates avahi-publish before stopping Docker Compose."""
    call_order = []
    mocker.patch(
        "loki.cli.subprocess.run",
        autospec=True,
        side_effect=lambda *a, **kw: call_order.append("compose"),
    )
    mocker.patch(
        "loki.cli.stop_avahi_publish",
        side_effect=lambda *a, **kw: call_order.append("avahi"),
    )

    CliRunner().invoke(cli, ["stop"])

    assert call_order.index("avahi") < call_order.index("compose")
