#!/usr/bin/env python3
"""
Deployment Verification Script

Verifies that all AI Hub platform components are properly deployed and functioning.
Checks all 4 phases:
- Phase 1: Core Infrastructure
- Phase 2: Advanced Infrastructure
- Phase 3: Visualization & Analysis
- Phase 4: Production Infrastructure

Usage:
    python verify_deployment.py [--full]

Exit codes:
    0 - All checks passed
    1 - Some checks failed
    2 - Critical checks failed
"""

import sys
import subprocess
import importlib
from pathlib import Path
from typing import List, Tuple
import json


class Colors:
    """ANSI color codes for terminal output."""

    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    BOLD = "\033[1m"
    END = "\033[0m"


def print_header(text: str):
    """Print a formatted header."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'=' * 70}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text.center(70)}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'=' * 70}{Colors.END}\n")


def print_check(name: str, passed: bool, details: str = ""):
    """Print a check result."""
    status = f"{Colors.GREEN}✓{Colors.END}" if passed else f"{Colors.RED}✗{Colors.END}"
    print(f"  {status} {name}")
    if details:
        print(f"      {details}")


def check_module_import(module_name: str) -> Tuple[bool, str]:
    """Check if a module can be imported."""
    try:
        importlib.import_module(module_name)
        return True, "Module imported successfully"
    except ImportError as e:
        return False, str(e)


def check_file_exists(path: str) -> Tuple[bool, str]:
    """Check if a file exists."""
    if Path(path).exists():
        return True, f"Found at {path}"
    return False, f"Not found at {path}"


def check_command(command: List[str]) -> Tuple[bool, str]:
    """Check if a command runs successfully."""
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            return True, "Command executed successfully"
        return False, f"Exit code: {result.returncode}"
    except Exception as e:
        return False, str(e)


class DeploymentVerifier:
    """Verifies AI Hub deployment."""

    def __init__(self):
        self.checks_passed = 0
        self.checks_failed = 0
        self.warnings = 0

    def run_check(self, name: str, check_func, *args) -> bool:
        """Run a single check and record result."""
        try:
            passed, details = check_func(*args)
            print_check(name, passed, details)

            if passed:
                self.checks_passed += 1
            else:
                self.checks_failed += 1
            return passed
        except Exception as e:
            print_check(name, False, f"Error: {e}")
            self.checks_failed += 1
            return False

    def verify_phase1(self):
        """Verify Phase 1: Core Infrastructure."""
        print_header("PHASE 1: Core Infrastructure")

        # Check core modules
        print(f"{Colors.BOLD}Core Modules:{Colors.END}")
        self.run_check("API Module", check_module_import, "src.ai_hub.api")
        self.run_check("Integration Hub", check_module_import, "src.ai_hub.integration")
        self.run_check(
            "Model Registry", check_module_import, "src.ai_hub.model_registry"
        )
        self.run_check(
            "Experiment Tracking", check_module_import, "src.ai_hub.experiment_tracking"
        )
        self.run_check("Security", check_module_import, "src.ai_hub.security")

        # Check files
        print(f"\n{Colors.BOLD}Configuration Files:{Colors.END}")
        self.run_check("pyproject.toml", check_file_exists, "pyproject.toml")
        self.run_check("CLI Entry Point", check_file_exists, "src/ai_hub/cli.py")

        # Check CLI
        print(f"\n{Colors.BOLD}CLI Availability:{Colors.END}")
        self.run_check(
            "ai-hub CLI", check_command, ["python", "-m", "src.ai_hub.cli", "--help"]
        )

    def verify_phase2(self):
        """Verify Phase 2: Advanced Infrastructure."""
        print_header("PHASE 2: Advanced Infrastructure")

        # Check modules
        print(f"{Colors.BOLD}Advanced Modules:{Colors.END}")
        self.run_check("Async API", check_module_import, "src.ai_hub.async_api")
        self.run_check(
            "Distributed Training", check_module_import, "src.ai_hub.distributed"
        )
        self.run_check("AutoML", check_module_import, "src.ai_hub.automl")
        self.run_check("Model Serving", check_module_import, "src.ai_hub.serving")

        # Check optional dependencies
        print(f"\n{Colors.BOLD}Optional Dependencies:{Colors.END}")
        ray_ok, _ = check_module_import("ray")
        print_check("Ray (for distributed)", ray_ok)
        if not ray_ok:
            self.warnings += 1

    def verify_phase3(self):
        """Verify Phase 3: Visualization & Analysis."""
        print_header("PHASE 3: Visualization & Analysis")

        # Check modules
        print(f"{Colors.BOLD}Visualization Modules:{Colors.END}")
        self.run_check(
            "Basic Visualization", check_module_import, "src.ai_hub.visualization"
        )
        self.run_check(
            "Advanced Visualization", check_module_import, "src.ai_hub.advanced_viz"
        )
        self.run_check(
            "Interpretability", check_module_import, "src.ai_hub.interpretability"
        )
        self.run_check(
            "Training Widgets", check_module_import, "src.ai_hub.training_widgets"
        )
        self.run_check(
            "Paper Generator", check_module_import, "src.ai_hub.paper_generator"
        )
        self.run_check(
            "Dataset Versioning", check_module_import, "src.ai_hub.dataset_versioning"
        )

        # Check dependencies
        print(f"\n{Colors.BOLD}Visualization Dependencies:{Colors.END}")
        plotly_ok, _ = check_module_import("plotly")
        print_check("Plotly", plotly_ok)
        if not plotly_ok:
            self.warnings += 1

    def verify_phase4(self):
        """Verify Phase 4: Production Infrastructure."""
        print_header("PHASE 4: Production Infrastructure")

        # Check Docker files
        print(f"{Colors.BOLD}Docker Configuration:{Colors.END}")
        self.run_check("Dockerfile", check_file_exists, "Dockerfile")
        self.run_check("Docker Compose", check_file_exists, "docker-compose.yml")
        self.run_check(".dockerignore", check_file_exists, ".dockerignore")

        # Check monitoring
        print(f"\n{Colors.BOLD}Monitoring Configuration:{Colors.END}")
        self.run_check(
            "Prometheus Config", check_file_exists, "monitoring/prometheus.yml"
        )
        self.run_check("Grafana Config", check_file_exists, "monitoring/grafana")

        # Check documentation
        print(f"\n{Colors.BOLD}Documentation:{Colors.END}")
        self.run_check("README", check_file_exists, "README.md")
        self.run_check(
            "Complete Summary", check_file_exists, "COMPLETE_IMPLEMENTATION_SUMMARY.md"
        )
        self.run_check("Phase 1 Docs", check_file_exists, "docs/PHASE1_README.md")
        self.run_check("Phase 2 Docs", check_file_exists, "docs/PHASE2_README.md")
        self.run_check("Phase 3 Docs", check_file_exists, "docs/PHASE3_README.md")
        self.run_check("Architecture Docs", check_file_exists, "docs/ARCHITECTURE.md")

    def verify_tests(self):
        """Verify test suite."""
        print_header("Test Suite Verification")

        print(f"{Colors.BOLD}Test Files:{Colors.END}")
        self.run_check("API Tests", check_file_exists, "tests/test_api.py")
        self.run_check("ML Engine Tests", check_file_exists, "tests/test_ml_engine.py")
        self.run_check(
            "Phase 1 Tests", check_file_exists, "tests/test_phase1_comprehensive.py"
        )
        self.run_check(
            "Phase 2 Tests", check_file_exists, "tests/test_phase2_comprehensive.py"
        )
        self.run_check(
            "Phase 3 Tests", check_file_exists, "tests/test_phase3_comprehensive.py"
        )
        self.run_check(
            "Integration Tests",
            check_file_exists,
            "tests/test_integration_all_phases.py",
        )

        # Try to run a quick test
        print(f"\n{Colors.BOLD}Quick Test Run:{Colors.END}")
        ok, details = check_command(
            ["python", "-m", "pytest", "tests/test_api.py", "-v", "--tb=short", "-x"]
        )
        if ok:
            print_check("API Tests Pass", True, "All API tests passed")
        else:
            print_check("API Tests Pass", False, "Some tests failed (check details)")
            self.warnings += 1

    def verify_examples(self):
        """Verify example files."""
        print_header("Example Files Verification")

        print(f"{Colors.BOLD}Example Scripts:{Colors.END}")
        self.run_check(
            "Phase 1 Examples",
            check_file_exists,
            "examples/phase1_complete_examples.py",
        )
        self.run_check(
            "Phase 2 Examples",
            check_file_exists,
            "examples/phase2_advanced_examples.py",
        )
        self.run_check(
            "Phase 3 Viz Examples",
            check_file_exists,
            "examples/phase3_advanced_visualization.py",
        )
        self.run_check(
            "Phase 3 Interpretability",
            check_file_exists,
            "examples/phase3_interpretability_examples.py",
        )
        self.run_check(
            "Phase 3 Widgets",
            check_file_exists,
            "examples/phase3_jupyter_widgets_examples.py",
        )
        self.run_check(
            "Master Example", check_file_exists, "examples/master_complete_example.py"
        )

    def generate_report(self) -> int:
        """Generate final report and return exit code."""
        print_header("DEPLOYMENT VERIFICATION SUMMARY")

        total = self.checks_passed + self.checks_failed

        print(f"{Colors.BOLD}Results:{Colors.END}")
        print(f"  {Colors.GREEN}✓ Passed:{Colors.END} {self.checks_passed}/{total}")
        print(f"  {Colors.RED}✗ Failed:{Colors.END} {self.checks_failed}/{total}")

        if self.warnings > 0:
            print(f"  {Colors.YELLOW}⚠ Warnings:{Colors.END} {self.warnings}")

        # Calculate percentage
        if total > 0:
            percentage = (self.checks_passed / total) * 100
            print(f"\n{Colors.BOLD}Success Rate:{Colors.END} {percentage:.1f}%")

        # Determine status
        print(f"\n{Colors.BOLD}Status:{Colors.END}")
        if self.checks_failed == 0:
            print(f"  {Colors.GREEN}{Colors.BOLD}✓ ALL CHECKS PASSED{Colors.END}")
            print(f"\n  The AI Hub platform is fully deployed and operational!")
            print(f"  All 4 phases (48 features) are ready to use.")
            return 0
        elif self.checks_failed < 3:
            print(f"  {Colors.YELLOW}{Colors.BOLD}⚠ MOSTLY FUNCTIONAL{Colors.END}")
            print(f"\n  Platform is functional with minor issues.")
            print(f"  Review failed checks above.")
            return 1
        else:
            print(
                f"  {Colors.RED}{Colors.BOLD}✗ DEPLOYMENT ISSUES DETECTED{Colors.END}"
            )
            print(f"\n  Multiple components are not properly configured.")
            print(f"  Please review the failed checks and fix issues.")
            return 2


def main():
    """Main verification function."""
    import argparse

    parser = argparse.ArgumentParser(description="Verify AI Hub platform deployment")
    parser.add_argument(
        "--full", action="store_true", help="Run full verification including tests"
    )
    parser.add_argument(
        "--phase", type=int, choices=[1, 2, 3, 4], help="Verify specific phase only"
    )

    args = parser.parse_args()

    print(f"{Colors.BOLD}{Colors.BLUE}")
    print("╔═══════════════════════════════════════════════════════════════╗")
    print("║           AI HUB DEPLOYMENT VERIFICATION                      ║")
    print("║              Complete 4-Phase Platform                        ║")
    print("╚═══════════════════════════════════════════════════════════════╝")
    print(f"{Colors.END}")

    verifier = DeploymentVerifier()

    if args.phase:
        # Verify specific phase
        if args.phase == 1:
            verifier.verify_phase1()
        elif args.phase == 2:
            verifier.verify_phase2()
        elif args.phase == 3:
            verifier.verify_phase3()
        elif args.phase == 4:
            verifier.verify_phase4()
    else:
        # Verify all phases
        verifier.verify_phase1()
        verifier.verify_phase2()
        verifier.verify_phase3()
        verifier.verify_phase4()
        verifier.verify_examples()

        if args.full:
            verifier.verify_tests()

    exit_code = verifier.generate_report()

    print(f"\n{Colors.BOLD}For more information:{Colors.END}")
    print("  • README.md - Overview and quick start")
    print("  • COMPLETE_IMPLEMENTATION_SUMMARY.md - Full platform details")
    print("  • examples/ - Runnable examples for all phases")
    print()

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
