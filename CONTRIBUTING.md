# Contributing

Thank you for your interest in contributing to S2FFT 🎉! We welcome contributions of all forms, including writing tutorials, improving documentation, creating bug reports or feature requests and updates to the package's code and infrastructure. 

If you don't feel like you currently have time to contribute but would like to show your support, [starring the repository](https://github.com/astro-informatics/s2fft/stargazers) or posting about it on social media is always appreciated!

## All contributors

We use the [All Contributors bot](https://allcontributors.org/) to record all forms of contribution to the project; you can see the list of all the wonderful people who have contributed to the project so far in the [README](README.md#contributors-). If you make a contribution to the project one of [the maintainers](https://github.com/orgs/astro-informatics/teams/s2fft-maintainers) should record your contribution [by adding a message tagging the bot](https://allcontributors.org/bot/usage/) in the relevant issue or pull request thread. If we forget to do this please remind us!

## Reporting bugs or requesting new features

If you have a question please first check if it is covered in the [documentation](https://astro-informatics.github.io/s2fft) or if there is an [existing issue](https://github.com/astro-informatics/s2fft/issues) which answers your query.

If there is not a relevant existing issue, to report a problem you are having with the package or request a new feature please [raise an issue](https://github.com/astro-informatics/s2fft/issues/new).

When reporting a bug; please describe the expected behaviour, what you actually observe, and provide sufficient information for someone else to reproduce the problem.
Ideally this should be in the form of a [_minimal reproducible example_](https://en.wikipedia.org/wiki/Minimal_reproducible_example) which reproduces the error while as being as small and simple as possible.

## Proposing changes to repository

We use a [fork](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo) and [pull-request](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-pull-requests) model for external contributions. Before opening a pull request that proposes substantial changes to the repository, for example adding a new feature or changing the public interface of the package, please first [raise an issue](https://github.com/astro-informatics/s2fft/issues/new) outlining the problem the proposed changes would address to allow for some discussion about the problem and proposed solution before significant time is invested.

If you have not made an open-source contribution via a pull request before you may find this [detailed guide](https://www.asmeurer.com/git-workflow/) by [asmeurer](https://github.com/asmeurer) helpful. A summary of the main steps is as follows:


1. [Fork the repository](https://github.com/astro-informatics/s2fft/fork) and create a local clone of the fork.
2. Create a new branch with a descriptive name in your fork clone.
3. Make the proposed changes on the branch, giving each commit a descriptive commit message.
4. Push the changes on the local branch to your fork on GitHub.
5. Create a [pull request](https://github.com/astro-informatics/s2fft/compare), specifying the fork branch as the source of the changes, giving the pull request a descriptive title and explaining what you are changing and why in the description. If the pull-request is resolving a specific issue, use [keywords](https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/using-keywords-in-issues-and-pull-requests) to link the appropriate issue.
6. Make sure all automated status checks pass on the pull request.
7. Await a review on the changes by one of [the project maintainers](https://github.com/orgs/astro-informatics/teams/s2fft-maintainers), and address any review comments.
8. Once all status checks pass and the changes have been approved by a maintainer the pull request can be (squash) merged.


## Python version support

We aim to broadly follow Scientific Python's [SPEC0 recommendation](https://scientific-python.org/specs/spec-0000/) - that is we will support Python versions for three years after their initial release and core scientific Python packages such as NumPy for two years after their initial release. 


## Code style and linting

The Python code in S2FFT uses the [Black code style](https://test-black.readthedocs.io/en/latest/the_black_code_style.html) and we use [Ruff](https://docs.astral.sh/ruff/) to lint and autoformat the code.

C++ code in S2FFT follows the [Google C++ code style](https://google.github.io/styleguide/cppguide.html) and we use [ClangFormat](https://clang.llvm.org/docs/ClangFormat.html) to autoformat the code.

We use pre-commit hooks to automatically check changes respect formatting and linting rules. You can install these hooks in your local repository by [installing pre-commit](https://pre-commit.com/#install) and running

```
pre-commit install
```

from the root of the repository.

The installed pre-commit hooks will run a series of checks on any staged changes when initiating a commit.  If there are problems with the changes this will be flagged in the terminal output. Where possible some problems may be automatically fixed - in this case the updated changes will need to be staged and committed again.

