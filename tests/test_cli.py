import json

import pytest

from inference_slo.cli import main


@pytest.mark.parametrize("argv", [[], ["demo"], ["sweep"], ["classes"]])
def test_commands_exit_zero_and_print_something(argv, capsys):
    assert main(argv) == 0
    assert capsys.readouterr().out.strip()


@pytest.mark.parametrize("argv", [["demo"], ["sweep"], ["classes"]])
def test_json_output_parses(argv, capsys):
    assert main(argv + ["--json"]) == 0
    assert json.loads(capsys.readouterr().out)


def test_demo_reports_all_three_policies(capsys):
    main(["demo"])
    out = capsys.readouterr().out
    for policy in ("admit-all", "queue-depth", "slo-aware"):
        assert policy in out


def test_rate_is_honoured_before_the_subcommand(capsys):
    """A flag on the parent parser must not be clobbered by the subparser default."""
    assert main(["--rate", "4", "demo"]) == 0
    assert "4 rps offered" in capsys.readouterr().out


def test_rate_is_honoured_after_the_subcommand(capsys):
    assert main(["demo", "--rate", "4"]) == 0
    assert "4 rps offered" in capsys.readouterr().out


def test_unknown_command_is_rejected():
    with pytest.raises(SystemExit):
        main(["nonsense"])
