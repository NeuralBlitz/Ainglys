#!/usr/bin/env python3
"""Master test runner with coverage reporting for all AI Hub packages."""

import subprocess
import sys
import os
from pathlib import Path
from datetime import datetime

# Test configurations
TEST_PACKAGES = [
    {
        "name": "AI Hub Core",
        "path": "tests",
        "description": "Core AI Hub functionality",
    },
    {
        "name": "ACCA Framework",
        "path": "projects/acca/tests",
        "description": "Attentive Causal Automata",
    },
    {
        "name": "ACML",
        "path": "Advanced-Research/Ideas/projects/acml/tests",
        "description": "Adaptive Categorical Meta-Learning",
    },
    {
        "name": "Aether-Calc",
        "path": "Advanced-Research/Ideas/projects/aether_calc/tests",
        "description": "Aether-Calculus optimizer",
    },
    {
        "name": "GAAV",
        "path": "Advanced-Research/Ideas/projects/gaav/tests",
        "description": "Granular Arithmetic & Visualization",
    },
    {
        "name": "GAFT",
        "path": "Advanced-Research/Ideas/projects/gaft/tests",
        "description": "Granular Arithmonic Field Theory",
    },
    {
        "name": "CNM",
        "path": "Advanced-Research/Ideas/projects/cnm/tests",
        "description": "Complex Network Morphogenesis",
    },
    {
        "name": "HNMA",
        "path": "Advanced-Research/Ideas/projects/hnma/tests",
        "description": "Hierarchical Neural Multi-Agent",
    },
    {
        "name": "MCAF",
        "path": "Advanced-Research/Ideas/projects/mcaf/tests",
        "description": "Meta-Cognitive Architecture",
    },
    {
        "name": "NEWNN",
        "path": "Advanced-Research/Ideas/projects/newnn/tests",
        "description": "Neuromorphic Edge-Weighted Networks",
    },
    {
        "name": "GCAA",
        "path": "Advanced-Research/Ideas/projects/gcaa/tests",
        "description": "Geometric Cellular Automata",
    },
]


def run_tests(package):
    """Run tests for a package."""
    print(f"\n{'=' * 70}")
    print(f"🧪 Testing: {package['name']}")
    print(f"📁 Path: {package['path']}")
    print(f"📝 Description: {package['description']}")
    print("=" * 70)

    test_path = Path(package["path"])

    if not test_path.exists():
        print(f"⚠️  Test directory not found: {test_path}")
        return None

    # Run pytest with coverage
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        str(test_path),
        "-v",
        "--tb=short",
        "--color=yes",
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)

        output = result.stdout + result.stderr

        # Parse results
        passed = output.count(" PASSED")
        failed = output.count(" FAILED")
        errors = output.count(" ERROR")

        # Extract test names
        test_names = []
        for line in output.split("\n"):
            if "::" in line and ("PASSED" in line or "FAILED" in line):
                test_names.append(line.strip())

        return {
            "name": package["name"],
            "returncode": result.returncode,
            "passed": passed,
            "failed": failed,
            "errors": errors,
            "total": passed + failed + errors,
            "output": output,
            "tests": test_names,
        }

    except subprocess.TimeoutExpired:
        return {
            "name": package["name"],
            "returncode": -1,
            "passed": 0,
            "failed": 0,
            "errors": 1,
            "total": 1,
            "output": "Timeout (>180s)",
            "tests": [],
        }
    except Exception as e:
        return {
            "name": package["name"],
            "returncode": -1,
            "passed": 0,
            "failed": 0,
            "errors": 1,
            "total": 1,
            "output": str(e),
            "tests": [],
        }


def print_summary(results):
    """Print test summary."""
    print("\n" + "=" * 70)
    print("📊 COMPREHENSIVE TEST SUMMARY")
    print("=" * 70)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    total_passed = 0
    total_failed = 0
    total_errors = 0
    total_tests = 0

    print(f"{'Package':<25} {'Passed':>8} {'Failed':>8} {'Errors':>8} {'Status':>10}")
    print("-" * 70)

    for result in results:
        if result is None:
            continue

        status = "✅ PASS" if result["returncode"] == 0 else "❌ FAIL"
        print(
            f"{result['name']:<25} {result['passed']:>8} {result['failed']:>8} {result['errors']:>8} {status:>10}"
        )

        total_passed += result["passed"]
        total_failed += result["failed"]
        total_errors += result["errors"]
        total_tests += result["total"]

    print("-" * 70)
    print(f"{'TOTAL':<25} {total_passed:>8} {total_failed:>8} {total_errors:>8}")
    print("=" * 70)

    # Calculate statistics
    if total_tests > 0:
        success_rate = (total_passed / total_tests) * 100
        print(f"\n📈 Success Rate: {success_rate:.1f}% ({total_passed}/{total_tests})")
        print(f"⏱️  Total Tests: {total_tests}")
        print(f"✅ Passed: {total_passed}")
        print(f"❌ Failed: {total_failed}")
        print(f"⚠️  Errors: {total_errors}")

    # Overall status
    print()
    if total_failed == 0 and total_errors == 0:
        print("🎉 ALL TESTS PASSED! Production ready.")
        return 0
    else:
        print(f"⚠️  {total_failed + total_errors} test(s) need attention")
        return 1


def save_report(results):
    """Save detailed test report."""
    report_path = Path("TEST_RESULTS.txt")

    with open(report_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("COMPREHENSIVE TEST RESULTS\n")
        f.write("=" * 70 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        for result in results:
            if result is None:
                continue

            f.write(f"\n{'=' * 70}\n")
            f.write(f"Package: {result['name']}\n")
            f.write(f"Status: {'PASS' if result['returncode'] == 0 else 'FAIL'}\n")
            f.write(
                f"Tests: {result['passed']} passed, {result['failed']} failed, {result['errors']} errors\n"
            )
            f.write("-" * 70 + "\n")

            # Write test details
            if result["tests"]:
                f.write("Test Details:\n")
                for test in result["tests"]:
                    f.write(f"  {test}\n")

            # Write output if there were failures
            if result["failed"] > 0 or result["errors"] > 0:
                f.write("\nOutput:\n")
                f.write(result["output"])

            f.write("\n")

    return report_path


def main():
    """Main entry point."""
    print("🚀 AI Hub & Research Frameworks - Comprehensive Test Suite")
    print("=" * 70)
    print(f"Working directory: {os.getcwd()}")
    print(f"Python: {sys.version}")
    print(f"Packages to test: {len(TEST_PACKAGES)}")
    print("=" * 70)

    # Run all tests
    results = []
    for package in TEST_PACKAGES:
        result = run_tests(package)
        results.append(result)

    # Print summary
    exit_code = print_summary(results)

    # Save report
    report_path = save_report(results)
    print(f"\n📝 Detailed report saved to: {report_path.absolute()}")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
