# Colab Quickstart

## Best workflow

Do not copy and paste the whole project into Colab.

Recommended pattern:

1. keep this repository on GitHub
2. keep notebooks in the repository
3. open the notebook from GitHub in Colab
4. let the notebook clone the repository into the Colab runtime
5. install the package and requirements from the cloned repository

This keeps GitHub as the source of truth and lets Colab act as the execution environment.

## Important local note

At the moment, this local repository does not have a Git remote configured.

Before the Colab workflow will be smooth, connect this repo to GitHub with the correct repository URL and push `main`.

Example:

```bash
cd "/Users/angelica.ramirez/Documents/New project"
git remote add origin https://github.com/YOUR-USERNAME/YOUR-REPO.git
git push -u origin main
```

## How to open the project in Colab

### Option A: Best option

Use GitHub-backed notebooks.

1. Push this repository to GitHub.
2. In Colab, choose `File -> Open notebook`.
3. Open the `GitHub` tab.
4. Paste the repository URL or search for the repository.
5. Open `notebooks/01_data_access_and_masks.ipynb`.
6. In the first setup cell, replace the placeholder `REPO_URL` with your actual GitHub repository URL.
7. Run the setup cell to clone the repo into the Colab runtime and install dependencies.

### Option B: Acceptable fallback

If the GitHub tab is being stubborn, upload the notebook file to Colab.

1. In Colab, choose `File -> Upload notebook`.
2. Upload `notebooks/01_data_access_and_masks.ipynb`.
3. Replace the placeholder `REPO_URL` in the notebook.
4. Run the setup cell so the rest of the repository is cloned into the runtime.

This still avoids manual copy-paste of the whole project.

## How to save notebook changes

If Colab opened the notebook from GitHub, use the GitHub save flow in Colab to save notebook changes back to the repository. In current Colab builds, this is commonly exposed as `File -> Save a copy in GitHub`. If the wording in your Colab UI differs, use the GitHub-related save option or download the `.ipynb` and commit it from the repo.

## Suggested day-to-day workflow

1. edit and run notebooks in Colab
2. keep reusable code in `src/jpcz_catalog/`
3. save notebook changes back to GitHub
4. pull those changes into this local repo when needed
5. avoid storing large data files in the repo

## What should live in Colab versus GitHub

Good for Colab:

- notebook execution
- exploratory plots
- ERA5 access in the cloud

Good for GitHub:

- notebooks
- helper modules in `src/`
- docs
- small verification summaries

