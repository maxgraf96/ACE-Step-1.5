#!/usr/bin/env bash

set -euo pipefail

MAIN_BRANCH="main"
CUSTOM_BRANCH="custom-main"
UPSTREAM_REMOTE="upstream"
ORIGIN_REMOTE="origin"
PUSH=true
ALLOW_DIRTY=false

usage() {
    cat <<'EOF'
Usage: ./sync_upstream.sh [options]

Sync routine:
1) Fetch remotes
2) Fast-forward local main from upstream/main
3) Push main to origin
4) Merge main into custom-main
5) Push custom-main to origin

Options:
  --main <branch>           Main branch name (default: main)
  --custom <branch>         Custom branch name (default: custom-main)
  --upstream <remote>       Upstream remote name (default: upstream)
  --origin <remote>         Fork remote name (default: origin)
  --no-push                 Do not push branches
  --allow-dirty             Allow running with uncommitted changes
  -h, --help                Show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --main)
            MAIN_BRANCH="$2"
            shift 2
            ;;
        --custom)
            CUSTOM_BRANCH="$2"
            shift 2
            ;;
        --upstream)
            UPSTREAM_REMOTE="$2"
            shift 2
            ;;
        --origin)
            ORIGIN_REMOTE="$2"
            shift 2
            ;;
        --no-push)
            PUSH=false
            shift
            ;;
        --allow-dirty)
            ALLOW_DIRTY=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage
            exit 1
            ;;
    esac
done

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "Error: not inside a git repository." >&2
    exit 1
fi

if [[ "$ALLOW_DIRTY" != true ]] && [[ -n "$(git status --porcelain)" ]]; then
    echo "Error: working tree has uncommitted changes." >&2
    echo "Commit/stash first, or run with --allow-dirty." >&2
    exit 1
fi

if ! git remote get-url "$UPSTREAM_REMOTE" >/dev/null 2>&1; then
    echo "Error: remote '$UPSTREAM_REMOTE' not found." >&2
    exit 1
fi

if ! git remote get-url "$ORIGIN_REMOTE" >/dev/null 2>&1; then
    echo "Error: remote '$ORIGIN_REMOTE' not found." >&2
    exit 1
fi

if ! git show-ref --verify --quiet "refs/heads/$MAIN_BRANCH"; then
    echo "Error: local branch '$MAIN_BRANCH' not found." >&2
    exit 1
fi

if ! git show-ref --verify --quiet "refs/heads/$CUSTOM_BRANCH"; then
    echo "Error: local branch '$CUSTOM_BRANCH' not found." >&2
    exit 1
fi

START_BRANCH="$(git rev-parse --abbrev-ref HEAD)"

echo "[1/7] Fetching $UPSTREAM_REMOTE..."
git fetch "$UPSTREAM_REMOTE"

echo "[2/7] Fetching $ORIGIN_REMOTE..."
git fetch "$ORIGIN_REMOTE"

echo "[3/7] Updating $MAIN_BRANCH from $UPSTREAM_REMOTE/$MAIN_BRANCH..."
git switch "$MAIN_BRANCH"
git merge --ff-only "$UPSTREAM_REMOTE/$MAIN_BRANCH"

if [[ "$PUSH" == true ]]; then
    echo "[4/7] Pushing $MAIN_BRANCH to $ORIGIN_REMOTE..."
    git push "$ORIGIN_REMOTE" "$MAIN_BRANCH"
else
    echo "[4/7] Skipping push of $MAIN_BRANCH (--no-push)."
fi

echo "[5/7] Merging $MAIN_BRANCH into $CUSTOM_BRANCH..."
git switch "$CUSTOM_BRANCH"
git merge "$MAIN_BRANCH"

if [[ "$PUSH" == true ]]; then
    echo "[6/7] Pushing $CUSTOM_BRANCH to $ORIGIN_REMOTE..."
    git push "$ORIGIN_REMOTE" "$CUSTOM_BRANCH"
else
    echo "[6/7] Skipping push of $CUSTOM_BRANCH (--no-push)."
fi

echo "[7/7] Returning to starting branch: $START_BRANCH"
git switch "$START_BRANCH"

echo "Sync completed."
