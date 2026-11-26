# AI usage policy

As large language model (LLM) and other AI tools become more popular, we believe that it is important to have guidance in place to ensure that both those who are enthusiastic adopters of these new technologies and those who have reservations about their use (and everyone in between) can work together in a compatible way.

As AI-assisted development becomes more widespread, we continue to believe that the codebase itself should aim be understandable to human reviewers, and the key goal of this policy is to provide a framework for maintaining this aim.


# Transparency

Whenever a LLM or other AI tool is used, this use should be disclosed in the relevant pull request.
This disclosure should include:

- Details of the AI tool that was used
- Which parts of the pull requst the AI tool was used for


# Review

Whenever a LLM or other AI tool is used, the user opening the pull request should review all code and text produced by the tool.
In particular the user should check that:

- The material is correct
- Code is written in a way that its purpose can be understood by a human developer

Note that this review by the user should be completed before requesting a pull request review from a project maintainer.

In order to comply with this policy, users must ensure that LLMs are other AI tools do not autonomously open issues or pull requests without and human interactions or review.
While we do not explicitly forbid the use of LLMs and other AI tools when authoring issues and pull requests, we encourage developers to write the main text of these themselves, as this is a good was to ensure that the review phase of this policy is adhered to.


# Accountability

The user is accountable for any code and text produced by LLMs or other AI tools that they commit to the project, and should be able to explain the detail and purpose of these contributions, as well as ensuring that all copyright rules and licenses are adhered to.
