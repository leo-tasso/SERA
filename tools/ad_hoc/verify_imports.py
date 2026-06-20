#!/usr/bin/env python3
"""Verify all labor downloader modules can be imported without syntax errors."""

import sys

sys.path.insert(0, "src")

print("Checking imports...")
try:
    from sera.downloaders.unemployment_rate import UnemploymentRateDownloader

    print("✓ UnemploymentRateDownloader imported")
except Exception as e:
    print(f"✗ UnemploymentRateDownloader: {e}")

try:
    from sera.downloaders.youth_employment import YouthEmploymentDownloader

    print("✓ YouthEmploymentDownloader imported")
except Exception as e:
    print(f"✗ YouthEmploymentDownloader: {e}")

try:
    from sera.downloaders.average_wages import AverageWagesDownloader

    print("✓ AverageWagesDownloader imported")
except Exception as e:
    print(f"✗ AverageWagesDownloader: {e}")

try:
    from sera.downloaders.self_employment import SelfEmploymentDownloader

    print("✓ SelfEmploymentDownloader imported")
except Exception as e:
    print(f"✗ SelfEmploymentDownloader: {e}")

try:
    from sera.downloaders.skills_match import SkillsMatchDownloader

    print("✓ SkillsMatchDownloader imported")
except Exception as e:
    print(f"✗ SkillsMatchDownloader: {e}")

print("\nAll modules checked successfully!")
