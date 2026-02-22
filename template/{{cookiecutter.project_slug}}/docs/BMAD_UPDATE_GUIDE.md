# BMAD Update Guide

This guide explains how to update the BMAD framework in projects generated from this template.

## Overview

BMAD is installed via the [BMAD installer](https://github.com/bmad-sim/bmad) into the `_bmad/` directory. The template includes a minimal BMAD configuration (`_bmad/bmm/config.yaml`) that is parameterized during project creation.

## Directory Structure

```
_bmad/
  bmm/              # BMM module (your project's BMAD configuration)
    config.yaml     # Project-specific BMAD config
  core/             # BMAD core (managed by BMAD updates -- do not edit)
  [other modules]   # Additional BMAD modules
```

## Updating BMAD

### Step 1: Check current BMAD version

Look at `_bmad/core/` for version indicators or check the BMAD installer documentation.

### Step 2: Run the BMAD installer

Follow the [BMAD installation guide](https://github.com/bmad-sim/bmad) to update. The installer updates `_bmad/core/` while preserving your `_bmad/bmm/` customizations.

### Step 3: Verify your customizations

After updating, verify that:
- `_bmad/bmm/config.yaml` still has your project settings
- Any custom agents in `_bmad/bmm/agents/` are preserved
- Any custom workflows in `_bmad/bmm/workflows/` still work

### Step 4: Test

Run your project's quality pipeline to ensure nothing is broken:

```sh
nox
```

## Preserving Customizations

Files in `_bmad/bmm/` are YOUR project's customizations. The BMAD installer should not overwrite these. If you need to customize BMAD agents or workflows:

1. Copy the agent/workflow from `_bmad/core/` to `_bmad/bmm/`
2. Make your modifications in the `bmm` copy
3. The `bmm` version takes precedence over `core`

## Template Updates vs BMAD Updates

- **Template updates** (via `cruft update`): Update project scaffolding (pyproject.toml, CI, docs)
- **BMAD updates** (via BMAD installer): Update BMAD framework itself

These are independent processes. You can update one without the other.
