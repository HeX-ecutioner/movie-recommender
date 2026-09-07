# Contributing to Cine Expert

Thank you for your interest in contributing to **Cine Expert**! We welcome contributions from the community—whether you are fixing a bug, improving documentation, enhancing algorithms, or adding new features.

---

## Code of Conduct

All contributors and maintainers are expected to adhere to our [Code of Conduct](CODE_OF_CONDUCT.md). Please read it before participating in our community.

---

## How Can I Contribute?

### Reporting Bugs
If you find a bug, please check existing issues first. If it has not been reported, open a new issue and include:
- A descriptive title.
- Clear steps to reproduce the issue.
- The expected vs. actual behavior.
- Relevant environment details (OS, Python version, browser).
- Console errors or server tracebacks.

### Suggesting Enhancements
Have an idea to improve recommendations or UI/UX? Open an issue tagged with `enhancement` detailing:
- The problem your idea solves.
- The proposed solution or behavior.
- Any alternative approaches considered.

### Pull Requests (PRs)
1. **Fork and Clone**:
   ```bash
   git clone https://github.com/HeX-ecutioner/cine-expert.git
   cd cine-expert
   ```
2. **Create a Topic Branch**:
   ```bash
   git checkout -b feature/my-new-feature
   # or
   git checkout -b fix/ambiguity-banner-styling
   ```
3. **Set Up Development Environment**:
   ```bash
   python -m venv venv
   # Windows
   .\venv\Scripts\activate
   # macOS/Linux
   source venv/bin/activate

   pip install -r requirements.txt
   ```
4. **Follow Project Guidelines**:
   - Write clean, readable, documented code.
   - Maintain documentation integrity: if changing core algorithms, API endpoints, or UI components, update the relevant files in `docs/`.
   - Ensure existing tests pass and add unit tests covering new logic.
5. **Run the Test Suite**:
   ```bash
   python -m unittest discover tests
   ```
6. **Commit and Push**:
   Use clear, descriptive commit messages:
   ```bash
   git commit -m "feat: improve collaborative filtering score stability"
   git push origin feature/my-new-feature
   ```
7. **Open a Pull Request**: Submit your PR against the `main` branch with a thorough explanation of changes made.

---

## Development Guidelines

### Backend (`api/` and `src/`)
- Adhere to PEP 8 style standards.
- Keep dependencies minimal to stay within Vercel serverless memory/bundle limits.
- Ensure all normalization, title resolution, and rating scoring edge cases (e.g. `NaN`, missing ratings, multiple release years) are safely handled.

### Frontend (`public/`)
- Cine Expert uses **Vanilla HTML5, CSS3, and ES6+ JavaScript** with zero build pipelines.
- Do not introduce heavyweight frameworks (e.g. React, Vue, Tailwind) unless agreed upon via an issue discussion.
- Utilize CSS Custom Properties defined in `public/styles.css` for consistent styling, colors, and dark/light theming.

---

## Questions?
Feel free to open an issue for discussion or reach out to the maintainers.
