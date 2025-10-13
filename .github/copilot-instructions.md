# Copilot Instructions for LLM Engineering Course Website

This is an MkDocs-based documentation site for an LLM (Large Language Model) engineering course. The site is hosted on GitHub Pages at `llm-engg.github.io`.

## Project Structure

```
mkdocs.yml          # MkDocs configuration (currently minimal)
schedule.csv        # Course schedule data (currently empty)
docs/
  index.md          # Main documentation homepage
.venv/              # Python virtual environment
```

## Development Workflow

### MkDocs Commands
- `mkdocs serve` - Start live-reloading development server (default: http://127.0.0.1:8000)
- `mkdocs build` - Build static site to `site/` directory
- `mkdocs gh-deploy` - Deploy to GitHub Pages (likely deployment method)

### Python Environment
- Uses `.venv/` virtual environment
- Activate with: `source .venv/bin/activate` (Linux/macOS) or `.venv\Scripts\activate` (Windows)
- MkDocs is the primary dependency

## Key Patterns & Conventions

### Documentation Structure
- All content lives in `docs/` directory
- Markdown files with `.md` extension
- `index.md` serves as homepage
- Standard MkDocs project layout expected

### Course-Specific Elements
- `schedule.csv` suggests course schedule functionality will be added
- This is for LLM engineering course for industry professionals
- Will contain course materials, assignments, and schedules

## Development Guidelines

### When Adding Course Content
- Place new markdown files in `docs/` directory
- Update `mkdocs.yml` navigation if adding new sections
- Consider how `schedule.csv` will integrate (may need custom MkDocs plugin or processing)

### Configuration Updates
- `mkdocs.yml` is currently minimal - will likely need navigation, theme, and plugin configuration
- Consider adding course-specific metadata and styling
- May need custom CSS/JS for course schedule display

### GitHub Pages Deployment
- Site builds from `main` branch
- Use `mkdocs gh-deploy` for deployment to `gh-pages` branch
- Ensure GitHub Pages is configured to serve from `gh-pages` branch

## Common Tasks

1. **Adding new course module**: Create `docs/module-X.md`, update navigation in `mkdocs.yml`
2. **Updating schedule**: Modify `schedule.csv` and ensure any schedule display logic is updated
3. **Local development**: Activate venv, run `mkdocs serve`, edit content with live reload
4. **Deployment**: Run `mkdocs gh-deploy` to publish changes
