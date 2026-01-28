# CLAUDE.md

## Debug Snapshots (llmdebug)

This project uses `llmdebug` for structured crash diagnostics. The pytest plugin auto-captures snapshots on test failures.

### On any failure:
1. **Read `.llmdebug/latest.json` first** - never patch before reading the snapshot
2. Analyze the snapshot:
   - **Exception type/message** - what went wrong
   - **Closest frame** - where it happened (index 0 is crash site)
   - **Locals** - variable values at crash time
   - **Array shapes** - look for empty arrays, shape mismatches, wrong dtypes
3. **Produce 2-4 ranked hypotheses** based on evidence before patching
4. Apply minimal fix for the most likely hypothesis
5. Re-run to verify

### Key signals to look for:
- `shape: [0, ...]` - empty array, upstream data issue
- `None` where object expected - initialization/ordering bug
- Type mismatch in locals - wrong function called or bad return value
- Index vs length - `i=10` with `len(arr)=10` means off-by-one

### When the snapshot isn't enough:
If the crash frame locals show the symptom but not the cause:
1. **Add targeted instrumentation** (not random fixes):
   - Wrap suspect code in `with snapshot_section("stage_x")`
   - Focus on the boundary where values go wrong
2. Re-run to generate a better snapshot
3. Repeat the hypothesis→patch loop

### Don't:
- Guess without reading the snapshot first
- Make multiple speculative changes - one hypothesis at a time
- Refactor until tests pass - prefer small, targeted fixes
- Ignore array shapes - they're often the key clue

### Debugging workflow summary:
```
fail → read snapshot → ranked hypotheses → minimal patch → rerun
```

This replaces:
```
fail → guess patch → rerun → repeat (the LLM roulette)
```

## Release Workflow

This project uses **python-semantic-release (PSR)** for automated releases.

### Conventional Commits

Use conventional commit messages - PSR parses them to determine version bumps:

| Commit Type | Example | Version Bump |
|-------------|---------|--------------|
| `feat:` | `feat: add GPU memory tracking` | Minor (0.1.4 → 0.2.0) |
| `fix:` | `fix: handle empty arrays` | Patch (0.1.4 → 0.1.5) |
| `perf:` | `perf: optimize serialization` | Patch |
| `feat!:` or `BREAKING CHANGE:` | `feat!: change API` | Major (0.1.4 → 1.0.0) |
| `docs:`, `test:`, `chore:` | `docs: update README` | None |

Format: `<type>(<optional scope>): <description>`

### How Releases Work

1. Push commits to `main` with conventional commit messages
2. GitHub Actions runs CI (tests, lint, typecheck)
3. Release workflow analyzes commits since last tag
4. If releasable commits found: bumps version, updates CHANGELOG.md, creates tag, publishes to PyPI

### Manual Commands

```bash
# Preview what would be released (dry run)
uv run semantic-release version --noop

# Check current version
uv run semantic-release version --print
```

### CI/CD Files

- `.github/workflows/ci.yml` - Runs on push/PR: tests, ruff, pyright
- `.github/workflows/release.yml` - Runs on main: auto-release to PyPI
