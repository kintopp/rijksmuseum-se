#!/usr/bin/env bash
# Apply local patches to vendored dependencies after `uv sync`.
#
# Usage:
#   uv sync && bash patches/apply-patches.sh
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SITE_PACKAGES="$REPO_ROOT/.venv/lib/python3.12/site-packages"
PATCHES_DIR="$REPO_ROOT/patches"

applied=0
skipped=0
failed=0

for patch in "$PATCHES_DIR"/*.patch; do
    [ -f "$patch" ] || continue
    name="$(basename "$patch")"

    # Extract the target package directory from the patch (b/package_name/...)
    pkg_dir=$(grep '^+++ b/' "$patch" | head -1 | sed 's|^+++ b/||; s|/.*||')

    if [ ! -d "$SITE_PACKAGES/$pkg_dir" ]; then
        echo "SKIP  $name — $pkg_dir not installed"
        skipped=$((skipped + 1))
        continue
    fi

    # --forward: skip if already applied (don't reverse). --batch: no prompts.
    if patch -p1 --forward --batch -d "$SITE_PACKAGES" < "$patch" 2>/dev/null; then
        echo "OK    $name"
        applied=$((applied + 1))
    else
        # exit code 1 from --forward means already applied
        if patch -p1 -R --dry-run --batch -d "$SITE_PACKAGES" < "$patch" >/dev/null 2>&1; then
            echo "SKIP  $name — already applied"
            skipped=$((skipped + 1))
        else
            echo "FAIL  $name — patch does not apply cleanly"
            failed=$((failed + 1))
        fi
    fi
done

echo ""
echo "Applied: $applied, Skipped: $skipped, Failed: $failed"
[ "$failed" -eq 0 ]
