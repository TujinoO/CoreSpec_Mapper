from __future__ import annotations

from unittest.mock import patch
import unittest

from corespec_mapper import cli


class V5CliTests(unittest.TestCase):
    def test_parser_exposes_v5_commands_without_a_spectral_library_path(self):
        parser = cli.build_parser()
        audit = parser.parse_args(["v5-audit", "--config", "project.json", "--output", "audit.json"])
        self.assertEqual(audit.command, "v5-audit")
        self.assertFalse(hasattr(audit, "spectral_library_root"))

        run = parser.parse_args([
            "v5-run",
            "--config", "project.json",
            "--output-root", "outputs",
            "--run-id", "run_001",
            "--start-line", "10",
            "--stop-line", "20",
        ])
        self.assertEqual(run.command, "v5-run")
        self.assertEqual((run.start_line, run.stop_line), (10, 20))

        validate = parser.parse_args([
            "v5-validate", "--config", "regression.json", "--output", "report.json"
        ])
        self.assertEqual(validate.command, "v5-validate")

        self.assertEqual(parser.parse_args(["desktop"]).command, "desktop")
        self.assertEqual(parser.parse_args(["desktop-v4"]).command, "desktop-v4")

    def test_v5_audit_and_library_commands_preserve_service_contract(self):
        config = {"inputs": {"analysis_image": "cube.dat"}}
        audit = {
            "library_ensemble": {"selected": 2},
            "selected_references": [{"mineral_id": "calcite"}],
            "mineral_support": [{"mineral_id": "calcite", "level": "supported"}],
        }
        with (
            patch.object(cli, "load_config", return_value=config) as load,
            patch.object(cli, "audit_v5_project", return_value=audit) as service,
            patch.object(cli, "save_json") as save,
        ):
            self.assertEqual(cli.main(["v5-audit", "--config", "project.json", "--output", "audit.json"]), 0)
            load.assert_called_with("project.json")
            service.assert_called_once_with(config)
            save.assert_called_once_with(audit, "audit.json")

            service.reset_mock()
            save.reset_mock()
            self.assertEqual(cli.main(["v5-library", "--config", "project.json", "--output", "library.json"]), 0)
            service.assert_called_once_with(config)
            save.assert_called_once_with(
                {
                    "library_ensemble": audit["library_ensemble"],
                    "selected_references": audit["selected_references"],
                    "mineral_support": audit["mineral_support"],
                },
                "library.json",
            )

    def test_v5_run_forwards_non_overwrite_identifier_and_subset(self):
        config = {"inputs": {"analysis_image": "cube.dat"}}
        result = {"run_directory": "outputs/Project/run_001"}
        with (
            patch.object(cli, "load_config", return_value=config),
            patch.object(cli, "run_v5_project", return_value=result) as service,
            patch("builtins.print") as output,
        ):
            self.assertEqual(
                cli.main([
                    "v5-run",
                    "--config", "project.json",
                    "--output-root", "outputs",
                    "--run-id", "run_001",
                    "--start-line", "10",
                    "--stop-line", "20",
                ]),
                0,
            )
            service.assert_called_once_with(
                config,
                "outputs",
                run_id="run_001",
                start_line=10,
                stop_line=20,
            )
            output.assert_called_once_with(result["run_directory"])

    def test_v5_validate_returns_nonzero_when_release_gate_fails(self):
        with patch.object(cli, "validate_v5_regression_file", return_value={"passed": False, "failures": ["mask"]}):
            self.assertEqual(
                cli.main(["v5-validate", "--config", "regression.json", "--output", "report.json"]),
                2,
            )


if __name__ == "__main__":
    unittest.main()
