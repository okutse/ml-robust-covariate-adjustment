#!/usr/bin/env bash
set -euo pipefail

cat <<'USAGE'
Usage: scripts/git_prune_simulation_history.sh [--dry-run]

This helper rewrites Git history to remove intermediate simulation artifacts
that are not intended to be tracked (per-replicate caches, checkpoints, etc.).

It requires `git-filter-repo` to be installed and available on PATH. This
script does NOT run the rewrite unless you remove the --dry-run flag and
confirm the operation. Rewriting Git history is destructive for shared
repositories: coordinate with your team and back up the repository first.
USAGE

DRY_RUN=true
if [[ "${1:-}" == "--dry-run" ]]; then
  shift || true
  DRY_RUN=true
else
  DRY_RUN=false
fi

if ! command -v git-filter-repo >/dev/null 2>&1; then
  echo "git-filter-repo is not installed. Install it from https://github.com/newren/git-filter-repo" >&2
  exit 1
fi

echo "This script will rewrite Git history to remove intermediate simulation artifacts."
echo "A backup branch 'pre-prune-backup' will be created. Push it to remote before proceeding."
read -r -p "Proceed to create backup branch? [y/N] " ans
if [[ "$ans" != "y" && "$ans" != "Y" ]]; then
  echo "Aborting. No changes made."
  exit 0
fi

git branch -f pre-prune-backup
echo "Created branch 'pre-prune-backup'. Please push it to the remote (git push origin pre-prune-backup) before continuing."

echo "The following patterns will be removed from history:" 
cat <<'PAT'
  - simulations/**/replicate_*.csv
  - simulations/**/replicate_*_checkpoint.csv
  - simulations/**/replicate_checkpoint.csv
  - simulations/**/archives/**
  - logs/**   (optional; adjust .gitignore if you want to keep logs tracked)
PAT

if [[ "$DRY_RUN" == "true" ]]; then
  echo "Dry run requested; no history will be rewritten. To perform the rewrite, re-run without --dry-run." >&2
  exit 0
fi

read -r -p "Ready to rewrite history and remove the above patterns? This is destructive. Continue? [y/N] " ans2
if [[ "$ans2" != "y" && "$ans2" != "Y" ]]; then
  echo "Aborting. No changes made."
  exit 0
fi

# Build git-filter-repo arguments to remove the patterns above
args=(--invert-paths)
args+=(--paths-from-file=- <<'PATHS'
simulations/**/replicate_*.csv
simulations/**/replicate_*_checkpoint.csv
simulations/**/replicate_checkpoint.csv
simulations/**/archives/**
logs/**
PATHS
)

echo "Running git-filter-repo to remove files. This may take time..."
git filter-repo --invert-paths --path-glob 'simulations/**/replicate_*.csv' \
  --path-glob 'simulations/**/replicate_*_checkpoint.csv' \
  --path-glob 'simulations/**/replicate_checkpoint.csv' \
  --path-glob 'simulations/**/archives/**' \
  --path-glob 'logs/**'

echo "History rewrite complete. Inspect the repository and then force-push to update remotes:" 
echo "  git push --force origin --all"
echo "Also push tags:" 
echo "  git push --force origin --tags"
