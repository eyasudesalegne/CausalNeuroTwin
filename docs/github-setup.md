# GitHub repository setup

## Create the remote repository

Create an empty public or private repository named `CausalNeuroTwin` under the `eyasudesalegne` account. Do not ask GitHub to generate a README, licence, or `.gitignore`, because these files already exist.

From the extracted Phase 01 directory:

```bash
git init
git branch -M main
git add .
git commit -m "chore: establish Phase 01 repository foundation"
git remote add origin https://github.com/eyasudesalegne/CausalNeuroTwin.git
git push -u origin main
```

## Repository settings

Configure a ruleset for `main`:

- require a pull request before merging;
- require at least one approval;
- require conversation resolution;
- require the CI checks to pass;
- block force pushes;
- block branch deletion;
- require CODEOWNERS review when additional maintainers are added.

Enable:

- Dependabot alerts and updates;
- secret scanning where available;
- private vulnerability reporting;
- issue templates;
- Discussions only when the project needs a public support channel.

## Initial release policy

Do not tag `v0.1.0-alpha.1` until the first GitHub Actions run passes. The release should be described as a repository foundation, not a scientific prototype.
